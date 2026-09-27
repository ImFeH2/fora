from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self, overload

from huddol.adapters.sandbox.commands import (
    linux_command,
    macos_command,
    windows_command,
)
from huddol.core.errors import DomainError

if TYPE_CHECKING:
    from huddol.adapters.sandbox.windows import WindowsWriteAccess
    from huddol.adapters.windows import WindowsJob


def entrypoint() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, "-I", "-m", "huddol"]


def dispatch_helper(argv: list[str]) -> int | None:
    if argv and argv[0] == "--read-file":
        from huddol.adapters.execution.reading import dispatch_read_file

        return dispatch_read_file(argv[1:])
    if argv and argv[0] == "--windows-execution" and os.name == "nt":
        from huddol.adapters.sandbox.windows import run_restricted_command
        from huddol.adapters.windows import join_job

        join_job(argv[1])
        if argv[3] != "--":
            raise ValueError("Execution command separator is required")
        return run_restricted_command(
            None if argv[2] == "-" else argv[2], argv[4:], os.getcwd()
        )
    if argv and argv[0] == "--windows-write-sandbox" and os.name == "nt":
        from huddol.adapters.sandbox.windows import run_restricted_command

        separator = argv.index("--")
        return run_restricted_command(argv[1], argv[separator + 1 :], os.getcwd())
    return None


class ExecutionBackend:
    manages_cleanup = False

    def __init__(self, enforce: bool) -> None:
        self._enforce = enforce
        self.name = sys.platform

    def wrap(
        self, argv: Sequence[str], cwd: Path, roots: tuple[Path, ...]
    ) -> list[str]:
        raise DomainError(
            "sandbox_unavailable",
            "Filesystem write protection is unavailable on this platform",
        )

    def spawn(
        self,
        argv: Sequence[str],
        cwd: Path,
        roots: tuple[Path, ...],
        *,
        piped_input: bool,
        input_data: bytes | None = None,
    ) -> subprocess.Popen[bytes]:
        command = (
            self.wrap(argv, cwd, tuple(root for root in roots if root.is_dir()))
            if self._enforce
            else list(argv)
        )
        return subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE if piped_input else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=os.name != "nt",
        )

    def terminate(self, process: subprocess.Popen[bytes]) -> None:
        if sys.platform != "win32":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()

    def communicate(
        self, process: subprocess.Popen[bytes], data: bytes | None, timeout: float
    ) -> tuple[bytes, bytes]:
        return process.communicate(input=data, timeout=timeout)

    def cleanup(self, process: subprocess.Popen[bytes], *, deadline: float) -> None:
        self.terminate(process)
        process.communicate(timeout=max(0, deadline - time.monotonic()))

    def finish(self, process: subprocess.Popen[bytes]) -> None:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def close(self, *, deadline: float | None = None) -> None:
        pass


class LinuxBackend(ExecutionBackend):
    def wrap(
        self, argv: Sequence[str], cwd: Path, roots: tuple[Path, ...]
    ) -> list[str]:
        return linux_command(argv, cwd, roots)


class MacOSBackend(ExecutionBackend):
    def wrap(
        self, argv: Sequence[str], cwd: Path, roots: tuple[Path, ...]
    ) -> list[str]:
        return macos_command(argv, roots)


@dataclass(eq=False)
class CleanupAttempt:
    done: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None


@dataclass(eq=False)
class ExecutionRecord:
    roots: tuple[Path, ...]
    cancel_deadline: float | None = None
    process: subprocess.Popen[bytes] | None = None
    job: WindowsJob | None = None
    created: threading.Event = field(default_factory=threading.Event)
    io_done: threading.Event = field(default_factory=threading.Event)
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    executor: ThreadPoolExecutor | None = None
    reader: Future[tuple[bytes, bytes]] | None = None
    execution_error: BaseException | None = None
    release_error: Exception | None = None
    attempt: CleanupAttempt | None = None
    pipe_retry: CleanupAttempt | None = None
    run_completed: bool = False
    helper_stop_requested: bool = False
    job_stop_requested: bool = False
    active_processes: int | None = None
    execution_stopped: bool = False
    released: bool = False
    stage: str = "creation"


class ExecutionCleanupError(ExceptionGroup):
    execution_stopped: bool

    def __new__(
        cls, message: str, errors: Sequence[Exception], execution_stopped: bool
    ) -> Self:
        result = super().__new__(cls, message, errors)
        result.execution_stopped = execution_stopped
        return result

    def __init__(
        self, message: str, errors: Sequence[Exception], execution_stopped: bool
    ) -> None:
        super().__init__(message, errors)

    @overload
    def derive[Error: Exception](
        self, errors: Sequence[Error], /
    ) -> ExceptionGroup[Error]: ...

    @overload
    def derive[Error: BaseException](
        self, errors: Sequence[Error], /
    ) -> BaseExceptionGroup[Error]: ...

    def derive(
        self, errors: Sequence[BaseException], /
    ) -> BaseExceptionGroup[BaseException]:
        exceptions = [error for error in errors if isinstance(error, Exception)]
        if len(exceptions) == len(errors):
            return ExecutionCleanupError(
                self.message, exceptions, self.execution_stopped
            )
        return BaseExceptionGroup(self.message, errors)


class WindowsBackend(ExecutionBackend):
    manages_cleanup = True

    def __init__(self, enforce: bool) -> None:
        super().__init__(enforce)
        self._windows: dict[tuple[Path, ...], WindowsWriteAccess] = {}
        self._jobs: dict[subprocess.Popen[bytes], ExecutionRecord] = {}
        self._results: dict[subprocess.Popen[bytes], ExecutionRecord] = {}
        self._pending: set[ExecutionRecord] = set()
        self._records: set[ExecutionRecord] = set()
        self._lock = threading.RLock()
        self._access_lock = threading.Lock()
        self._coordinator: threading.Thread | None = None
        self._closed = False

    @contextmanager
    def _state(self, deadline: float) -> Iterator[None]:
        if not self._lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise ExecutionCleanupError(
                "Windows execution state is busy",
                [TimeoutError("Cleanup waiting for execution state")],
                False,
            )
        try:
            yield
        finally:
            self._lock.release()

    @staticmethod
    def _stage(record: ExecutionRecord, stage: str) -> None:
        record.stage = stage

    def wrap(
        self, argv: Sequence[str], cwd: Path, roots: tuple[Path, ...]
    ) -> list[str]:
        return windows_command(self._access(roots).sid, argv, entrypoint())

    def _access(self, roots: tuple[Path, ...]) -> WindowsWriteAccess:
        from huddol.adapters.sandbox.windows import WindowsWriteAccess

        with self._access_lock:
            if roots not in self._windows:
                self._windows[roots] = WindowsWriteAccess(roots)
            access = self._windows[roots]
            access.initialize()
            return access

    def spawn(
        self,
        argv: Sequence[str],
        cwd: Path,
        roots: tuple[Path, ...],
        *,
        piped_input: bool,
        input_data: bytes | None = None,
    ) -> subprocess.Popen[bytes]:
        from huddol.adapters.windows import WindowsJob

        record = ExecutionRecord(tuple(root for root in roots if root.is_dir()))
        with self._lock:
            if self._closed:
                raise DomainError("execution_closed", "Command creation is closed")
            previous = tuple(self._jobs)
            self._records.add(record)
            self._pending.add(record)
            self._stage(record, "creation")
        try:
            for process in previous:
                self._reap(process)
            sid = self._access(record.roots).sid if self._enforce else "-"
            record.job = WindowsJob()
            executable = None
            environment = None
            if not getattr(sys, "frozen", False) and sys.prefix != sys.base_prefix:
                executable = getattr(sys, "_base_executable", None)
                if (
                    not isinstance(executable, str)
                    or not Path(executable).is_absolute()
                ):
                    raise ValueError("Python base executable must be an absolute path")
                environment = dict(os.environ)
                environment["__PYVENV_LAUNCHER__"] = sys.executable
            if record.cancel_requested.is_set():
                raise DomainError("execution_closed", "Command creation was cancelled")
            process = subprocess.Popen(
                [
                    *entrypoint(),
                    "--windows-execution",
                    record.job.name,
                    sid,
                    "--",
                    *argv,
                ],
                executable=executable,
                env=environment,
                cwd=cwd,
                stdin=subprocess.PIPE if piped_input else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            record.process = process
            with self._lock:
                self._jobs[process] = record
                self._results[process] = record
            record.executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"huddol-command-{process.pid}"
            )
            record.reader = record.executor.submit(process.communicate, input_data)
        except BaseException as error:
            record.execution_error = error
            if record.process is None:
                record.io_done.set()
            record.created.set()
            try:
                deadline = record.cancel_deadline
                if deadline is None:
                    deadline = time.monotonic() + 5
                self._cleanup_records((record,), deadline, retry=False)
            except (OSError, RuntimeError, ExceptionGroup) as cleanup_error:
                raise BaseExceptionGroup(
                    "Command creation and cleanup failed", [error, cleanup_error]
                ) from None
            raise
        finally:
            with self._lock:
                self._pending.discard(record)
                record.created.set()
                if not record.cancel_requested.is_set():
                    self._stage(record, "running")
        return process

    def communicate(
        self, process: subprocess.Popen[bytes], data: bytes | None, timeout: float
    ) -> tuple[bytes, bytes]:
        with self._lock:
            record = self._results[process]
            reader = record.reader
        assert reader is not None
        try:
            output = reader.result(timeout)
        except TimeoutError as error:
            if reader.done() and reader.exception() is error:
                raise
            raise subprocess.TimeoutExpired(process.args, timeout) from error
        with self._lock:
            record.io_done.set()
            if not record.cancel_requested.is_set():
                record.run_completed = True
                self._stage(record, "background handoff")
        return output

    @staticmethod
    def _pipes_complete(record: ExecutionRecord) -> bool:
        if record.io_done.is_set():
            return True
        process = record.process
        assert process is not None
        reader = record.reader
        if reader is None or (reader.done() and reader.exception() is not None):
            if reader is not None:
                record.execution_error = reader.exception()
            if record.pipe_retry is record.attempt:
                raise RuntimeError(
                    "Command pipe processing failed"
                ) from record.execution_error
            record.pipe_retry = record.attempt
            if record.executor is None:
                record.executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix=f"huddol-command-{process.pid}"
                )
            record.reader = record.executor.submit(process.communicate)
            return False
        if not reader.done():
            return False
        record.io_done.set()
        return True

    def cleanup(self, process: subprocess.Popen[bytes], *, deadline: float) -> None:
        self.terminate(process, deadline=deadline)

    def terminate(
        self, process: subprocess.Popen[bytes], *, deadline: float | None = None
    ) -> None:
        deadline = time.monotonic() + 5 if deadline is None else deadline
        with self._state(deadline):
            record = self._jobs.get(process)
            if record is None:
                return
            if record.run_completed and not record.cancel_requested.is_set():
                return
            record.cancel_requested.set()
            if record.cancel_deadline is None:
                record.cancel_deadline = deadline
            deadline = record.cancel_deadline
        self._cleanup_records((record,), deadline, retry=False)

    @staticmethod
    def _close_process(process: subprocess.Popen[bytes]) -> None:
        from huddol.adapters.windows import close_process_handle

        close_process_handle(process)

    def _release(self, record: ExecutionRecord) -> None:
        process = record.process
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
        if record.executor is not None:
            record.executor.shutdown(wait=False)
            record.executor = None
        if process is not None:
            self._stage(record, "helper handle release")
            self._close_process(process)
        self._stage(record, "job handle release")
        if record.job is not None:
            record.job.close()
        record.released = True
        record.release_error = None
        self._stage(record, "released")
        with self._lock:
            self._records.discard(record)
            self._pending.discard(record)
            if process is not None:
                self._jobs.pop(process, None)

    def _advance(self, record: ExecutionRecord) -> bool:
        if record.released:
            return True
        if not record.created.is_set():
            self._stage(record, "creation")
            return False
        if not record.execution_stopped:
            process = record.process
            if process is not None and process.poll() is None:
                self._stage(record, "helper termination")
                if not record.helper_stop_requested:
                    process.kill()
                    record.helper_stop_requested = True
                return False
            if record.job is not None:
                if not record.job_stop_requested:
                    self._stage(record, "job termination")
                    record.job.request_termination()
                    record.job_stop_requested = True
                record.active_processes = record.job.active_processes()
                if record.active_processes:
                    self._stage(record, "job completion")
                    return False
            self._stage(record, "pipe completion")
            if not self._pipes_complete(record):
                return False
            record.execution_stopped = True
        self._stage(record, "pipe release")
        self._release(record)
        return True

    def _cleanup_records(
        self,
        records: Sequence[ExecutionRecord],
        deadline: float,
        *,
        retry: bool = True,
    ) -> None:
        attempts: dict[ExecutionRecord, CleanupAttempt] = {}
        with self._state(deadline):
            for record in records:
                record.cancel_requested.set()
                if record.cancel_deadline is None:
                    record.cancel_deadline = deadline
                attempt = record.attempt
                if attempt is None or (retry and attempt.done.is_set()):
                    attempt = CleanupAttempt()
                    record.attempt = attempt
                    if record.released:
                        attempt.done.set()
                attempts[record] = attempt
            if self._coordinator is None and any(
                not attempt.done.is_set() for attempt in attempts.values()
            ):
                self._coordinator = threading.Thread(
                    target=self._drive_cleanup,
                    name="huddol-windows-cleanup",
                    daemon=True,
                )
                self._coordinator.start()
        errors: list[Exception] = []
        for record, attempt in attempts.items():
            if not attempt.done.wait(max(0, deadline - time.monotonic())):
                errors.append(TimeoutError(f"Cleanup in progress at {record.stage}"))
            elif attempt.error is not None:
                errors.append(attempt.error)
        if errors:
            raise ExecutionCleanupError(
                "Windows execution cleanup failed",
                errors,
                all(record.execution_stopped for record in records),
            )

    def _drive_cleanup(self) -> None:
        while True:
            with self._lock:
                pending = tuple(
                    (record, record.attempt)
                    for record in self._records
                    if record.attempt is not None and not record.attempt.done.is_set()
                )
                if not pending:
                    self._coordinator = None
                    return
            for record, attempt in pending:
                if not record.lock.acquire(blocking=False):
                    continue
                try:
                    if self._advance(record):
                        attempt.done.set()
                except (OSError, RuntimeError, ExceptionGroup) as error:
                    identity = (
                        record.process.pid if record.process is not None else "pending"
                    )
                    error.add_note(f"Execution {identity}: {record.stage}")
                    attempt.error = error
                    attempt.done.set()
                finally:
                    record.lock.release()
            time.sleep(0.01)

    def _reap(self, process: subprocess.Popen[bytes]) -> None:
        with self._lock:
            record = self._jobs.get(process)
        if record is None or not record.lock.acquire(blocking=False):
            return
        try:
            if record.attempt is not None or not record.run_completed:
                return
            if record.io_done.is_set() and record.job is not None:
                record.active_processes = record.job.active_processes()
                if record.active_processes == 0:
                    record.execution_stopped = True
                    self._release(record)
                else:
                    record.release_error = None
                    self._stage(record, "background handoff")
        except (OSError, RuntimeError, ExceptionGroup) as error:
            record.release_error = error
            raise ExecutionCleanupError(
                "Windows command finalization failed", [error], record.execution_stopped
            ) from None
        finally:
            record.lock.release()

    def finish(self, process: subprocess.Popen[bytes]) -> None:
        with self._lock:
            record = self._results.pop(process, None)
        if record is not None and record.io_done.is_set():
            self._reap(process)

    def close(self, *, deadline: float | None = None) -> None:
        deadline = time.monotonic() + 5 if deadline is None else deadline
        self._closed = True
        with self._state(deadline):
            records = tuple(self._records)
        errors: list[Exception] = []
        try:
            self._cleanup_records(records, deadline)
        except ExecutionCleanupError as error:
            errors.extend(error.exceptions)
        with self._lock:
            remaining_records = tuple(self._records)
        if not remaining_records:
            if self._access_lock.acquire(timeout=max(0, deadline - time.monotonic())):
                try:
                    for roots, access in tuple(self._windows.items()):
                        try:
                            access.close()
                            del self._windows[roots]
                        except (OSError, ExceptionGroup) as error:
                            errors.append(error)
                finally:
                    self._access_lock.release()
            else:
                errors.append(TimeoutError("Windows access cleanup is in progress"))
        if errors:
            raise ExecutionCleanupError(
                "Windows environment cleanup failed",
                errors,
                all(record.execution_stopped for record in remaining_records),
            )


def create_backend(enforce: bool) -> ExecutionBackend:
    if sys.platform.startswith("linux"):
        return LinuxBackend(enforce)
    if sys.platform == "darwin":
        return MacOSBackend(enforce)
    if os.name == "nt":
        return WindowsBackend(enforce)
    return ExecutionBackend(enforce)
