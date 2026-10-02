from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol


@dataclass(frozen=True)
class TreeEntry:
    path: str
    kind: Literal["file", "directory"]
    size: int
    modified_at: str


class ConflictError(Exception):
    def __init__(self, path: str, expected: str | None, actual: str) -> None:
        super().__init__(f"{path} changed since it was read")
        self.path = path
        self.expected = expected
        self.actual = actual


class FileTree(Protocol):
    @property
    def root(self) -> Path: ...

    def list(self, path: str | None = None) -> tuple[TreeEntry, ...]: ...

    def read(self, path: str) -> tuple[str, str]: ...

    def write(
        self, path: str, content: str, *, expected_hash: str | None = None
    ) -> TreeEntry: ...

    def edit(
        self, path: str, old_text: str, new_text: str, *, replace_all: bool = False
    ) -> tuple[TreeEntry, str]: ...

    def mkdir(self, path: str) -> TreeEntry: ...

    def delete(self, path: str) -> None: ...

    def move(self, source: str, destination: str) -> TreeEntry: ...
