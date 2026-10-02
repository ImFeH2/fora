from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath

from fora.adapters.execution.editing import edit_file
from fora.adapters.execution.platforms import create_backend, entrypoint
from fora.adapters.sandbox.paths import normalize_directories, normalize_tolerantly
from fora.core.errors import DomainError
from fora.ports.execution import EditResult, RunResult

MAX_OUTPUT = 200_000
DEFAULT_TIMEOUT = 120
_READ_FILE_CODES = frozenset(
    {
        "not_found",
        "permission_denied",
        "not_regular_file",
        "file_too_large",
        "read_failed",
    }
)


def _read_file_protocol_error(
    exit_code: int, stdout: bytes, stderr: bytes
) -> DomainError:
    if stdout:
        return DomainError("execution_protocol", "read_file returned unexpected stdout")
    if exit_code != 3 or not stderr or len(stderr) > 8192:
        return DomainError("execution_protocol", "read_file returned an invalid error")
    try:
        payload = json.loads(stderr.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return DomainError("execution_protocol", "read_file returned invalid JSON")
    if not isinstance(payload, dict) or set(payload) != {"code", "message"}:
        return DomainError("execution_protocol", "read_file returned invalid fields")
    code = payload["code"]
    message = payload["message"]
    if (
        not isinstance(code, str)
        or code not in _READ_FILE_CODES
        or not isinstance(message, str)
        or len(message) > 512
    ):
        return DomainError(
            "execution_protocol", "read_file returned invalid diagnostics"
        )
    return DomainError(code, message)


class LocalExecution:
    def __init__(
        self,
        write_directories: Sequence[str] = (),
        *,
        enforce: bool = True,
        tolerant: bool = False,
    ) -> None:
        self._backend = create_backend(enforce)
        self._lock = threading.RLock()
        self._processes: set[subprocess.Popen[bytes]] = set()
        self._closed = False
        self.skipped: tuple[tuple[str, str], ...] = ()
        if tolerant:
            result = normalize_tolerantly(write_directories)
            self._roots = result.accepted
            self.skipped = result.skipped
        else:
            self._roots = normalize_directories(write_directories)

    @property
    def write_directories(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self._roots)

    def apply_configuration(self, candidate: LocalExecution) -> None:
        with self._lock:
            self._roots = candidate._roots
            self.skipped = candidate.skipped

    def resolve_path(self, value: str, *, base: str) -> str:
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            return value
        return str(Path(base) / value)

    def read_file(
        self, path: str, max_bytes: int, *, timeout: int | None = None
    ) -> bytes:
        if not isinstance(path, str) or "\0" in path or not Path(path).is_absolute():
            raise DomainError(
                "invalid_path", "path must be an absolute path without NUL characters"
            )
        if type(max_bytes) is not int or max_bytes <= 0:
            raise DomainError("invalid_limit", "max_bytes must be a positive integer")
        code, stdout, stderr = self._execute(
            [*entrypoint(), "--read-file", path, str(max_bytes)],
            cwd=str(Path(sys.executable).resolve().parent),
            timeout=timeout,
        )
        if code == 0:
            if stderr or len(stdout) > max_bytes + 1:
                raise DomainError(
                    "execution_protocol", "read_file returned invalid output"
                )
            return stdout
        raise _read_file_protocol_error(code, stdout, stderr)

    def close(self, *, deadline: float | None = None) -> None:
        deadline = time.monotonic() + 5 if deadline is None else deadline
        self._closed = True
        if self._backend.manages_cleanup:
            self._backend.close(deadline=deadline)
            return
        with self._lock:
            errors: list[Exception] = []
            for process in self._processes:
                try:
                    self._backend.terminate(process)
                except (OSError, subprocess.TimeoutExpired, ExceptionGroup) as error:
                    errors.append(error)
            try:
                self._backend.close(deadline=deadline)
            except (OSError, subprocess.TimeoutExpired, ExceptionGroup) as error:
                errors.append(error)
            if errors:
                raise ExceptionGroup("Execution cleanup failed", errors)

    def describe_environment(self, labeled: Sequence[tuple[str, str]] = ()) -> str:
        entries = [f"- {path} ({label})" for path, label in labeled]
        entries.extend(f"- {item}" for item in self.write_directories)
        listing = "\n".join(entries) or "- none"
        return f"Commands run on {self._backend.name}\nWritable directories:\n{listing}"

    def _resolve_cwd(self, cwd: str | None) -> Path:
        if cwd is None or not Path(cwd).is_absolute():
            raise DomainError("invalid_cwd", "cwd must be an absolute path")
        resolved = Path(cwd).resolve()
        if not resolved.is_dir():
            raise DomainError("invalid_cwd", f"{cwd} is not a directory")
        return resolved

    def _execute(
        self,
        argv: Sequence[str],
        cwd: str | None,
        timeout: int | None,
        data: bytes | None = None,
        write_directories: Sequence[str] | None = None,
    ) -> tuple[int, bytes, bytes]:
        if not argv or not all(
            isinstance(item, str) and "\0" not in item for item in argv
        ):
            raise DomainError(
                "invalid_argv",
                "argv must be a non-empty list of strings without NUL characters",
            )
        if timeout is not None and (type(timeout) is not int or timeout <= 0):
            raise DomainError("invalid_timeout", "timeout must be a positive integer")
        directory = self._resolve_cwd(cwd)
        try:
            with self._lock:
                if self._closed:
                    raise DomainError(
                        "execution_closed", "Execution environment is closed"
                    )
                roots = (
                    self._roots
                    if write_directories is None
                    else normalize_directories(write_directories)
                )
                process = self._backend.spawn(
                    argv,
                    directory,
                    roots,
                    piped_input=data is not None,
                    input_data=data,
                )
                self._processes.add(process)
        except FileNotFoundError as error:
            raise DomainError("command_not_found", str(error)) from error
        failure: BaseException | None = None
        try:
            stdout, stderr = self._backend.communicate(
                process, data, timeout or DEFAULT_TIMEOUT
            )
            return process.returncode, stdout, stderr
        except BaseException as error:
            failure = error
            try:
                self._backend.cleanup(process, deadline=time.monotonic() + 5)
            except (
                OSError,
                RuntimeError,
                subprocess.TimeoutExpired,
                ExceptionGroup,
            ) as cleanup_error:
                failure = BaseExceptionGroup(
                    "Command failed and cleanup failed", [error, cleanup_error]
                )
                raise failure from None
            if isinstance(error, subprocess.TimeoutExpired):
                raise DomainError(
                    "timeout", f"Command exceeded {timeout or DEFAULT_TIMEOUT} seconds"
                ) from error
            raise
        finally:
            with self._lock:
                self._processes.discard(process)
            try:
                self._backend.finish(process)
            except Exception as finish_error:
                if failure is not None:
                    raise BaseExceptionGroup(
                        "Command failed and finalization failed",
                        [failure, finish_error],
                    ) from None
                raise

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        timeout: int | None = None,
        write_directories: Sequence[str] | None = None,
    ) -> RunResult:
        code, output, errors = self._execute(
            argv, cwd, timeout, write_directories=write_directories
        )
        stdout, stderr = (
            output.decode("utf-8", "replace"),
            errors.decode("utf-8", "replace"),
        )
        return RunResult(
            code,
            stdout[:MAX_OUTPUT],
            stderr[:MAX_OUTPUT],
            len(stdout) > MAX_OUTPUT or len(stderr) > MAX_OUTPUT,
        )

    def edit(
        self,
        path: str,
        edits: Sequence[Mapping[str, object]],
        *,
        write_directories: Sequence[str] | None = None,
        create: bool = False,
    ) -> EditResult:
        return edit_file(
            path,
            edits,
            directories=list(
                self.write_directories
                if write_directories is None
                else write_directories
            ),
            create=create,
        )
