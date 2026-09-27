from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    stdout: str
    stderr: str
    truncated: bool


@dataclass(frozen=True)
class EditResult:
    path: str
    diff: str
    replacements: int


class ExecutionEnvironment(Protocol):
    @property
    def skipped(self) -> tuple[tuple[str, str], ...]: ...

    @property
    def write_directories(self) -> tuple[str, ...]: ...

    def describe_environment(self, labeled: Sequence[tuple[str, str]] = ()) -> str: ...

    def resolve_path(self, value: str, *, base: str) -> str: ...

    def read_file(
        self, path: str, max_bytes: int, *, timeout: int | None = None
    ) -> bytes: ...

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        timeout: int | None = None,
        write_directories: Sequence[str] | None = None,
    ) -> RunResult: ...

    def edit(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        replace_all: bool = False,
        write_directories: Sequence[str] | None = None,
        create: bool = False,
    ) -> EditResult: ...


class ExecutionControl(Protocol):
    def snapshot(self) -> ExecutionEnvironment: ...

    def status(self) -> dict[str, Any]: ...

    def configure(
        self, values: dict[str, Any], persist: Callable[[dict[str, object]], None]
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...
