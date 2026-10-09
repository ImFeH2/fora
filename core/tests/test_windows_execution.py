from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import pytest

from fora.adapters.execution import platforms
from fora.adapters.execution.local import LocalExecution
from fora.core.errors import DomainError

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows execution")


def wait_for(path: Path) -> None:
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Process did not create {path}")
        time.sleep(0.01)


def command_tree(directory: Path, *, detached: bool = False) -> list[str]:
    child = (
        "import os,time; from pathlib import Path; "
        f"Path({str(directory / 'pid')!r}).write_text(str(os.getpid())); "
        f"Path({str(directory / 'ready')!r}).touch(); "
        "time.sleep(3); "
        f"Path({str(directory / 'late')!r}).write_text('late'); time.sleep(30)"
    )
    redirects = (
        ", stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL"
        if detached
        else ""
    )
    parent = (
        "import subprocess,sys; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]{redirects})"
    )
    return [sys._base_executable, "-c", parent]


@pytest.mark.parametrize("enforce", [False, True])
def test_timeout_terminates_descendants_after_parent_exit(
    tmp_path, monkeypatch, enforce
):
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.IsProcessInJob.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.BOOL),
    ]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    kernel.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        *[ctypes.POINTER(wintypes.FILETIME)] * 4,
    ]
    kernel.GetProcessTimes.restype = wintypes.BOOL

    def require(value):
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())

    def creation_time():
        values = [wintypes.FILETIME() for _ in range(4)]
        require(
            kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in values))
        )
        return values[0].dwHighDateTime, values[0].dwLowDateTime

    environment = LocalExecution([str(tmp_path)], enforce=enforce)
    backend = environment._backend
    original = backend.communicate
    handle = None
    identity = None
    observed = None
    started = None

    def communicate(process, data, timeout):
        nonlocal handle, identity, observed, started
        wait_for(tmp_path / "ready")
        pid = int((tmp_path / "pid").read_text())
        handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
        require(handle)
        identity = creation_time()
        observed = backend._results[process]
        belongs = wintypes.BOOL()
        require(
            kernel.IsProcessInJob(handle, observed.job.handle, ctypes.byref(belongs))
        )
        assert belongs.value
        assert kernel.WaitForSingleObject(handle, 0) == 258
        assert process.wait(timeout=5) == 0
        started = time.monotonic()
        return original(process, data, timeout)

    monkeypatch.setattr(backend, "communicate", communicate)
    try:
        with pytest.raises(DomainError) as error:
            result = environment.run(
                command_tree(tmp_path), cwd=str(tmp_path), timeout=1
            )
            pytest.fail(repr(result))
        assert error.value.code == "timeout"
        assert started is not None and time.monotonic() - started < 3
        assert observed.execution_stopped and observed.active_processes == 0
        assert observed.released and observed.io_done.is_set()
        assert observed.job.handle is None and observed.executor is None
        assert observed.process._handle.closed
        assert all(
            stream is None or stream.closed
            for stream in (
                observed.process.stdin,
                observed.process.stdout,
                observed.process.stderr,
            )
        )
        assert not backend._jobs and not backend._records and not backend._pending
        assert not backend._results
        assert kernel.WaitForSingleObject(handle, 5000) == 0
        assert creation_time() == identity
        time.sleep(2.2)
        assert not (tmp_path / "late").exists()
    finally:
        try:
            environment.close()
            assert not backend._windows
        finally:
            if handle is not None:
                require(kernel.CloseHandle(handle))


@pytest.mark.parametrize("enforce", [False, True])
def test_normal_exit_preserves_background_until_close(tmp_path, enforce):
    environment = LocalExecution([str(tmp_path)], enforce=enforce)
    try:
        result = environment.run(
            command_tree(tmp_path, detached=True), cwd=str(tmp_path)
        )
        assert result.exit_code == 0, result.stderr
        wait_for(tmp_path / "pid")
        pid = int((tmp_path / "pid").read_text())
        child = psutil.Process(pid)
        assert child.is_running()
        records = tuple(environment._backend._jobs.values())
        assert records
        environment.close()
        assert all(record.execution_stopped for record in records)
        assert all(record.active_processes == 0 for record in records)
        assert all(record.released for record in records)
        assert environment._backend._jobs == {}
        child.wait(timeout=5)
        time.sleep(3.1)
        assert not (tmp_path / "late").exists()
    finally:
        environment.close()


@pytest.mark.parametrize("enforce", [False, True])
def test_close_terminates_command_and_descendants(tmp_path, enforce):
    environment = LocalExecution([str(tmp_path)], enforce=enforce)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(environment.run, command_tree(tmp_path), cwd=str(tmp_path))
        try:
            wait_for(tmp_path / "pid")
            pid = int((tmp_path / "pid").read_text())
            started = time.monotonic()
            environment.close()
            future.result(timeout=5)
            assert time.monotonic() - started < 3
            assert not psutil.pid_exists(pid)
            assert not (tmp_path / "late").exists()
        finally:
            environment.close()


def test_interrupted_communicate_cleans_up(tmp_path, monkeypatch):
    environment = LocalExecution([str(tmp_path)], enforce=False)
    original = subprocess.Popen.communicate
    interrupted = False

    def communicate(process, *args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            wait_for(tmp_path / "pid")
            raise KeyboardInterrupt()
        return original(process, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "communicate", communicate)
    try:
        with pytest.raises(KeyboardInterrupt):
            environment.run(command_tree(tmp_path), cwd=str(tmp_path))
        assert not psutil.pid_exists(int((tmp_path / "pid").read_text()))
        assert environment._backend._jobs == {}
    finally:
        environment.close()


@pytest.mark.parametrize("joined", [False, True])
def test_close_during_helper_startup(tmp_path, monkeypatch, joined):
    marker = tmp_path / "helper-ready"
    executed = tmp_path / "executed"
    source = (
        "import sys,time; from pathlib import Path; "
        "from fora.adapters.windows import join_job; "
        + ("join_job(sys.argv[2]); " if joined else "")
        + f"Path({str(marker)!r}).write_text('ready'); time.sleep(30)"
    )
    monkeypatch.setattr(platforms, "entrypoint", lambda: [sys.executable, "-c", source])
    environment = LocalExecution(enforce=False)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(
            environment.run,
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(executed)!r}).touch()",
            ],
            cwd=str(tmp_path),
        )
        try:
            wait_for(marker)
            environment.close()
            future.result(timeout=5)
            assert not executed.exists()
            assert environment._backend._jobs == {}
        finally:
            environment.close()


def test_join_failure_never_runs_command(tmp_path):
    executed = tmp_path / "executed"
    result = subprocess.run(
        [
            *platforms.entrypoint(),
            "--windows-execution",
            "Local\\Fora-missing-job",
            "-",
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(executed)!r}).touch()",
        ],
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert b"WinError 2" in result.stderr
    assert not executed.exists()


def test_helper_spawn_failure_releases_job(tmp_path, monkeypatch):
    environment = LocalExecution(enforce=False)
    monkeypatch.setattr(sys, "_base_executable", str(tmp_path / "missing.exe"))
    try:
        with pytest.raises(DomainError) as error:
            environment.run(["unused"], cwd=str(tmp_path))
        assert error.value.code == "command_not_found"
        assert not environment._backend._jobs
        assert not environment._backend._pending
    finally:
        environment.close()


@pytest.mark.parametrize("value", [None, "relative.exe", 42])
def test_invalid_helper_entry_releases_job(tmp_path, monkeypatch, value):
    environment = LocalExecution(enforce=False)
    monkeypatch.setattr(sys, "_base_executable", value)
    try:
        with pytest.raises(ValueError, match="absolute path"):
            environment.run(["unused"], cwd=str(tmp_path))
        assert not environment._backend._jobs
        assert not environment._backend._pending
    finally:
        environment.close()


@pytest.mark.parametrize("enforce", [False, True])
def test_helper_interpreter_identity_and_environment(tmp_path, monkeypatch, enforce):
    import fora

    original_environment = dict(os.environ)
    source = (
        "import json,sys,fora; "
        "from fora.adapters.execution.platforms import dispatch_helper; "
        "print(json.dumps({'executable':sys.executable,"
        "'base_executable':sys._base_executable,'prefix':sys.prefix,"
        "'base_prefix':sys.base_prefix,'module':fora.__file__,"
        "'isolated':sys.flags.isolated}),flush=True); "
        "sys.exit(dispatch_helper(sys.argv[1:]))"
    )
    monkeypatch.setattr(
        platforms, "entrypoint", lambda: [sys.executable, "-I", "-c", source]
    )
    environment = LocalExecution([str(tmp_path)], enforce=enforce)
    try:
        result = environment.run(
            [
                sys._base_executable,
                "-c",
                "import os; print('__PYVENV_LAUNCHER__' in os.environ)",
            ],
            cwd=str(tmp_path),
        )
        assert result.exit_code == 0, result.stderr
        identity, inherited = result.stdout.splitlines()
        assert json.loads(identity) == {
            "executable": sys.executable,
            "base_executable": sys._base_executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "module": fora.__file__,
            "isolated": 1,
        }
        assert inherited == "False"
        assert dict(os.environ) == original_environment
    finally:
        environment.close()


def test_nested_job_inheritance(tmp_path):
    from fora.adapters.windows import WindowsJob

    outer = WindowsJob()
    source = (
        "from fora.adapters.execution.local import LocalExecution; "
        "import sys; "
        "environment=LocalExecution(enforce=False); "
        f"result=environment.run([sys.executable,'-c','print(42)'],cwd={str(tmp_path)!r}); "
        "environment.close(); assert result.exit_code==0,result.stderr; print(result.stdout)"
    )
    try:
        result = subprocess.run(
            [
                *platforms.entrypoint(),
                "--windows-execution",
                outer.name,
                "-",
                "--",
                sys.executable,
                "-c",
                source,
            ],
            check=False,
            capture_output=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
        assert b"42" in result.stdout
        assert outer.active_processes() == 0
    finally:
        outer.terminate()
        outer.close()


def test_closed_environment_rejects_command_creation(tmp_path):
    environment = LocalExecution(enforce=False)
    environment.close()
    with pytest.raises(DomainError, match="closed"):
        environment.run(
            [sys._base_executable, "-c", "raise AssertionError('executed')"],
            cwd=str(tmp_path),
        )
    assert not environment._backend._records
    assert not environment._backend._pending


def writer_source(directory: Path, name: str) -> str:
    return (
        "import time; from pathlib import Path\n"
        f"root=Path({str(directory)!r})\n"
        f"(root / '{name}-ready').touch()\n"
        "while True:\n"
        f" if (root / '{name}-request').exists():\n"
        f"  (root / '{name}-response').write_text('written')\n"
        " time.sleep(0.02)\n"
    )


def test_timeout_preserves_shared_roots_and_background_across_configuration(tmp_path):
    environment = LocalExecution([str(tmp_path)], enforce=True)
    backend = environment._backend
    source = writer_source(tmp_path, "background")
    parent = (
        "import subprocess,sys; "
        f"subprocess.Popen([sys.executable,'-c',{source!r}], "
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
    )
    with ThreadPoolExecutor() as pool:
        try:
            result = environment.run(
                [sys._base_executable, "-c", parent], cwd=str(tmp_path)
            )
            assert result.exit_code == 0, result.stderr
            wait_for(tmp_path / "background-ready")
            background = next(iter(backend._records))
            assert background.run_completed and not background.released
            first_result = pool.submit(
                environment.run,
                [sys._base_executable, "-c", writer_source(tmp_path, "first")],
                cwd=str(tmp_path),
                timeout=1,
            )
            second_result = pool.submit(
                environment.run,
                [sys._base_executable, "-c", writer_source(tmp_path, "second")],
                cwd=str(tmp_path),
            )
            wait_for(tmp_path / "first-ready")
            wait_for(tmp_path / "second-ready")
            with backend._lock:
                records = tuple(backend._records)
            first = next(r for r in records if "first-ready" in r.process.args[-1])
            second = next(r for r in records if "second-ready" in r.process.args[-1])
            access = backend._windows[(tmp_path,)]
            with pytest.raises(DomainError) as error:
                first_result.result(timeout=5)
            assert error.value.code == "timeout"
            assert first.execution_stopped and first.released
            candidate = LocalExecution(enforce=True)
            environment.apply_configuration(candidate)
            candidate.close()
            assert backend._windows[(tmp_path,)] is access
            for name in ("second", "background"):
                (tmp_path / f"{name}-request").touch()
                wait_for(tmp_path / f"{name}-response")
                assert (tmp_path / f"{name}-response").read_text() == "written"
            assert not second.execution_stopped
            environment.close()
            second_result.result(timeout=5)
            assert second.execution_stopped and background.execution_stopped
            assert not backend._windows
            assert not backend._records
        finally:
            environment.close()


def test_job_close_failure_preserves_stopped_result_and_retries(tmp_path, monkeypatch):
    environment = LocalExecution([str(tmp_path)], enforce=True)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(
            environment.run,
            [sys._base_executable, "-c", writer_source(tmp_path, "command")],
            cwd=str(tmp_path),
        )
        try:
            wait_for(tmp_path / "command-ready")
            record = next(iter(environment._backend._records))
            original = record.job.close
            calls = []

            def fail():
                calls.append("close")
                raise OSError(5, "job close denied")

            monkeypatch.setattr(record.job, "close", fail)
            with pytest.raises(platforms.ExecutionCleanupError) as failure:
                environment.close()
            assert failure.value.execution_stopped
            assert record.execution_stopped and not record.released
            assert record.attempt.error is not None
            assert record.job.handle
            assert record.process._handle.closed
            assert environment._backend._windows
            future.result(timeout=5)
            assert calls == ["close"]
            monkeypatch.setattr(record.job, "close", original)
            environment.close()
            assert record.released and record.attempt.error is None
            assert record.job.handle is None
            assert not environment._backend._windows
        finally:
            environment.close()


def test_close_deadline_preserves_unique_pipe_owner_until_completion(
    tmp_path, monkeypatch
):
    environment = LocalExecution(enforce=False)
    entered = threading.Event()
    release = threading.Event()
    original = subprocess.Popen.communicate
    readers = []

    def communicate(process, *args, **kwargs):
        readers.append(threading.get_ident())
        output = original(process, *args, **kwargs)
        entered.set()
        assert release.wait(10)
        return output

    monkeypatch.setattr(subprocess.Popen, "communicate", communicate)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(
            environment.run,
            [sys._base_executable, "-c", "print('done')"],
            cwd=str(tmp_path),
        )
        try:
            assert entered.wait(5)
            record = next(iter(environment._backend._records))
            deadline = time.monotonic() + 0.08
            with pytest.raises(platforms.ExecutionCleanupError) as failure:
                environment.close(deadline=deadline)
            assert not failure.value.execution_stopped
            assert record.stage == "pipe completion"
            attempt = record.attempt
            with pytest.raises(platforms.ExecutionCleanupError):
                environment.close(deadline=deadline)
            assert record.attempt is attempt
            assert not record.execution_stopped
            release.set()
            assert attempt.done.wait(5)
            assert attempt.error is None
            assert record.execution_stopped and record.released
            assert len(readers) == 1
            assert future.result(timeout=5).stdout.strip() == "done"
        finally:
            release.set()
            environment.close()


def test_access_failure_retries_only_remaining_resources(tmp_path, monkeypatch):
    environment = LocalExecution([str(tmp_path)], enforce=True)
    try:
        result = environment.run(
            [sys._base_executable, "-c", "print('ok')"], cwd=str(tmp_path)
        )
        assert result.exit_code == 0, result.stderr
        backend = environment._backend
        access = backend._windows[(tmp_path,)]
        original = access.close

        def fail():
            raise OSError(5, "access cleanup denied")

        monkeypatch.setattr(access, "close", fail)
        with pytest.raises(platforms.ExecutionCleanupError) as failure:
            environment.close()
        assert failure.value.execution_stopped
        assert not backend._records
        assert backend._windows[(tmp_path,)] is access
        monkeypatch.setattr(access, "close", original)
        environment.close()
        assert not backend._windows
    finally:
        environment.close()


def test_one_job_failure_does_not_prevent_other_job_cleanup(tmp_path, monkeypatch):
    environment = LocalExecution(enforce=False)
    with ThreadPoolExecutor() as pool:
        futures = [
            pool.submit(
                environment.run,
                [sys._base_executable, "-c", writer_source(tmp_path, name)],
                cwd=str(tmp_path),
            )
            for name in ("first", "second")
        ]
        try:
            wait_for(tmp_path / "first-ready")
            wait_for(tmp_path / "second-ready")
            with environment._backend._lock:
                records = tuple(environment._backend._records)
            first = next(r for r in records if "first-ready" in r.process.args[-1])
            second = next(r for r in records if "second-ready" in r.process.args[-1])
            job = first.job
            original = job.active_processes

            def fail():
                raise OSError(5, "query denied")

            monkeypatch.setattr(job, "active_processes", fail)
            with pytest.raises(platforms.ExecutionCleanupError) as failure:
                environment.close()
            assert not failure.value.execution_stopped
            assert not first.execution_stopped
            assert second.execution_stopped and second.released
            monkeypatch.setattr(job, "active_processes", original)
            environment.close()
            assert first.released
            for future in futures:
                future.result(timeout=5)
        finally:
            environment.close()


def test_creation_failure_during_close_uses_original_deadline(tmp_path, monkeypatch):
    environment = LocalExecution(enforce=False)
    entered = threading.Event()
    release = threading.Event()
    received = []
    backend = environment._backend
    cleanup = backend._cleanup_records

    def creating(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        raise OSError("creation failed")

    def record_deadline(records, deadline, **kwargs):
        received.append(deadline)
        return cleanup(records, deadline, **kwargs)

    monkeypatch.setattr(platforms.subprocess, "Popen", creating)
    monkeypatch.setattr(backend, "_cleanup_records", record_deadline)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(environment.run, ["unused"], cwd=str(tmp_path))
        try:
            assert entered.wait(5)
            record = next(iter(backend._records))
            deadline = time.monotonic() + 0.08
            with pytest.raises(platforms.ExecutionCleanupError):
                environment.close(deadline=deadline)
            assert record.cancel_deadline == deadline
            assert record.stage == "creation"
            release.set()
            with pytest.raises((OSError, ExceptionGroup)):
                future.result(timeout=5)
            assert record.attempt.done.wait(5)
            assert record.attempt.error is None
            assert received == [deadline, deadline]
            assert record.execution_stopped and record.released
            assert not backend._pending and not backend._records
        finally:
            release.set()
            environment.close()


def test_all_records_receive_cleanup_with_one_environment_deadline(
    tmp_path, monkeypatch
):
    environment = LocalExecution(enforce=False)
    entered = [threading.Event(), threading.Event()]
    release = threading.Event()
    original = subprocess.Popen.communicate

    def communicate(process, *args, **kwargs):
        output = original(process, *args, **kwargs)
        index = int(output[0].strip())
        entered[index].set()
        assert release.wait(10)
        return output

    monkeypatch.setattr(subprocess.Popen, "communicate", communicate)
    with ThreadPoolExecutor() as pool:
        futures = [
            pool.submit(
                environment.run,
                [sys._base_executable, "-c", f"print({index})"],
                cwd=str(tmp_path),
            )
            for index in range(2)
        ]
        try:
            assert all(event.wait(5) for event in entered)
            with environment._backend._lock:
                records = tuple(environment._backend._records)
            assert len(records) == 2
            started = time.monotonic()
            deadline = started + 0.08
            with pytest.raises(platforms.ExecutionCleanupError):
                environment.close(deadline=deadline)
            assert time.monotonic() - started < 0.5
            for record in records:
                assert record.cancel_deadline == deadline
                assert record.stage == "pipe completion"
                assert record.job_stop_requested
                assert not record.execution_stopped
            release.set()
            environment.close()
            assert all(record.released for record in records)
            for future in futures:
                future.result(timeout=5)
        finally:
            release.set()
            environment.close()


def test_scheduler_stop_closes_real_execution_manager(tmp_path):
    from unittest.mock import Mock

    from fora.adapters.execution.manager import ExecutionManager
    from fora.runtime.scheduler import Scheduler

    manager = ExecutionManager(
        settings={"write_directories": [str(tmp_path)]}, enforce=True
    )
    scheduler = Scheduler(Mock(execution=manager), Mock())
    with ThreadPoolExecutor() as pool:
        future = pool.submit(
            manager.snapshot().run,
            command_tree(tmp_path),
            cwd=str(tmp_path),
        )
        try:
            wait_for(tmp_path / "pid")
            record = next(iter(manager._environment._backend._records))
            scheduler.stop()
            future.result(timeout=5)
            assert record.execution_stopped and record.released
            assert not manager._environment._backend._windows
            time.sleep(3.1)
            assert not (tmp_path / "late").exists()
        finally:
            manager.close()


def test_timeout_and_close_share_one_cleanup_attempt(tmp_path, monkeypatch):
    environment = LocalExecution(enforce=False)
    entered = threading.Event()
    release = threading.Event()
    original = environment._backend.cleanup

    def cleanup(process, *, deadline):
        entered.set()
        assert release.wait(5)
        return original(process, deadline=deadline)

    monkeypatch.setattr(environment._backend, "cleanup", cleanup)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(
            environment.run,
            command_tree(tmp_path),
            cwd=str(tmp_path),
            timeout=1,
        )
        try:
            assert entered.wait(5)
            record = next(iter(environment._backend._records))
            environment.close()
            attempt = record.attempt
            release.set()
            with pytest.raises(DomainError) as error:
                future.result(timeout=5)
            assert error.value.code == "timeout"
            assert record.attempt is attempt
            assert record.released
        finally:
            release.set()
            environment.close()


def test_partial_access_creation_and_revoke_failure_keep_remaining_ownership(
    tmp_path, monkeypatch
):
    from fora.adapters.sandbox import windows

    environment = LocalExecution([str(tmp_path)], enforce=True)
    original = windows._change_ace
    changes = []

    def fail_second_grant(path, sid, mode):
        changes.append(mode)
        if changes.count(windows.SET_ACCESS) == 2:
            raise OSError(5, "grant denied")
        return original(path, sid, mode)

    monkeypatch.setattr(windows, "_change_ace", fail_second_grant)
    try:
        with pytest.raises(OSError, match="grant denied"):
            environment.run(["unused"], cwd=str(tmp_path))
        backend = environment._backend
        access = backend._windows[(tmp_path,)]
        assert len(access._grants) == 1
        assert not access._ready
        assert access._desktop and access._window_station

        def fail_revoke(path, sid, mode):
            assert mode == windows.REVOKE_ACCESS
            raise OSError(5, "revoke denied")

        monkeypatch.setattr(windows, "_change_ace", fail_revoke)
        with pytest.raises(platforms.ExecutionCleanupError) as error:
            environment.close()
        assert error.value.execution_stopped
        assert backend._windows[(tmp_path,)] is access
        assert len(access._grants) == 1
        assert access._desktop is None and access._window_station is None
        assert access._sid and access._logon_sid
        monkeypatch.setattr(windows, "_change_ace", original)
        environment.close()
        assert not access._grants
        assert not access._sid and not access._logon_sid
        assert not backend._windows
    finally:
        monkeypatch.setattr(windows, "_change_ace", original)
        environment.close()


def test_completed_records_release_native_helper_handles(tmp_path, monkeypatch):
    environment = LocalExecution(enforce=False)
    backend = environment._backend
    records = []
    finish = backend.finish

    def capture(process):
        records.append(backend._results[process])
        finish(process)

    monkeypatch.setattr(backend, "finish", capture)
    try:
        for _ in range(8):
            result = environment.run(
                [sys._base_executable, "-c", "print('ok')"], cwd=str(tmp_path)
            )
            assert result.exit_code == 0, result.stderr
        assert all(record.released for record in records)
        assert all(record.process._handle.closed for record in records)
        assert not backend._records
    finally:
        environment.close()


def test_invalid_command_has_no_execution_resources(tmp_path):
    environment = LocalExecution(enforce=False)
    try:
        with pytest.raises(DomainError) as error:
            environment.run([], cwd=str(tmp_path))
        assert error.value.code == "invalid_argv"
        assert not environment._backend._records
        assert not environment._backend._pending
    finally:
        environment.close()
