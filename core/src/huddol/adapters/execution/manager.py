from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from huddol.adapters.execution.local import LocalExecution
from huddol.core.errors import DomainError
from huddol.ports.execution import EditResult, ExecutionEnvironment, RunResult


def setting_directories(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise DomainError(
            "invalid_directory", "Writable directories must be a list of paths"
        )
    return list(value)


class BoundExecution:
    def __init__(self, lookup: Callable[[], ExecutionEnvironment]) -> None:
        self._lookup = lookup

    @property
    def skipped(self) -> tuple[tuple[str, str], ...]:
        return self._lookup().skipped

    @property
    def write_directories(self) -> tuple[str, ...]:
        return self._lookup().write_directories

    def describe_environment(self, labeled: Sequence[tuple[str, str]] = ()) -> str:
        return self._lookup().describe_environment(labeled)

    def resolve_path(self, value: str, *, base: str) -> str:
        return self._lookup().resolve_path(value, base=base)

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        timeout: int | None = None,
        write_directories: Sequence[str] | None = None,
    ) -> RunResult:
        return self._lookup().run(
            argv, cwd=cwd, timeout=timeout, write_directories=write_directories
        )

    def edit(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        replace_all: bool = False,
        write_directories: Sequence[str] | None = None,
        create: bool = False,
    ) -> EditResult:
        return self._lookup().edit(
            path,
            old_text,
            new_text,
            replace_all=replace_all,
            write_directories=write_directories,
            create=create,
        )


class ExecutionManager:
    def __init__(
        self,
        *,
        settings: dict[str, object] | None = None,
        enforce: bool = True,
        tolerant: bool = False,
    ) -> None:
        self._enforce = enforce
        self._lock = threading.RLock()
        self._closed = False
        self._environment: LocalExecution | None = None
        self._directories: list[str] = []
        self._error: str | None = None
        stored = settings or {}
        try:
            if set(stored) - {"write_directories"}:
                raise DomainError("invalid_setting", "Unknown execution setting")
            self._directories = setting_directories(stored.get("write_directories"))
            self._environment = LocalExecution(
                self._directories,
                enforce=self._enforce,
                tolerant=tolerant,
            )
        except DomainError as error:
            self._error = str(error)

    def snapshot(self) -> ExecutionEnvironment:
        def lookup() -> ExecutionEnvironment:
            with self._lock:
                if self._closed:
                    raise DomainError(
                        "execution_closed", "Execution environment is closed"
                    )
                if self._environment is None:
                    raise DomainError(
                        "execution_unavailable",
                        self._error or "Execution environment is unavailable",
                    )
                return self._environment

        return BoundExecution(lookup)

    def status(self) -> dict[str, Any]:
        with self._lock:
            skipped = self._environment.skipped if self._environment else ()
            return {
                "write_directories": list(self._directories),
                "unusable_write_directories": [
                    {"path": path, "reason": reason} for path, reason in skipped
                ],
                "error": self._error,
            }

    def configure(
        self, values: dict[str, Any], persist: Callable[[dict[str, object]], None]
    ) -> dict[str, Any]:
        if set(values) - {"write_directories"}:
            raise DomainError("invalid_setting", "Unknown execution setting")
        with self._lock:
            if self._closed:
                raise DomainError("execution_closed", "Execution environment is closed")
            directories = setting_directories(
                values.get("write_directories", self._directories)
            )
            candidate = LocalExecution(directories, enforce=self._enforce)
            configured: dict[str, object] = {
                "write_directories": list(candidate.write_directories)
            }
            try:
                persist(configured)
            except Exception:
                candidate.close()
                raise
            if self._environment is None:
                self._environment = candidate
            else:
                self._environment.apply_configuration(candidate)
                candidate.close()
            self._directories = list(candidate.write_directories)
            self._error = None
        return self.status()

    def close(self, *, deadline: float | None = None) -> None:
        deadline = time.monotonic() + 5 if deadline is None else deadline
        self._closed = True
        if not self._lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise TimeoutError("Execution manager is busy")
        try:
            environment = self._environment
        finally:
            self._lock.release()
        if environment is not None:
            environment.close(deadline=deadline)
