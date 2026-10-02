from __future__ import annotations

from fora.core.errors import DomainError
from fora.ports.files import FileTree, TreeEntry

INDEX = "MEMORY.md"


class Workspace:
    def __init__(self, tree: FileTree) -> None:
        self._tree = tree

    def _ensure_index(self) -> None:
        try:
            self._tree.read(INDEX)
        except DomainError as error:
            if error.code == "not_readable":
                return
            if error.code != "not_found":
                raise
            self._tree.write(INDEX, "")

    def list(self, path: str | None = None) -> tuple[TreeEntry, ...]:
        self._ensure_index()
        return self._tree.list(path)

    def read(self, path: str) -> tuple[str, str]:
        self._ensure_index()
        return self._tree.read(path)

    def write(
        self, path: str, content: str, *, expected_hash: str | None = None
    ) -> TreeEntry:
        self._ensure_index()
        return self._tree.write(path, content, expected_hash=expected_hash)

    def edit(
        self, path: str, old_text: str, new_text: str, *, replace_all: bool = False
    ) -> tuple[TreeEntry, str]:
        self._ensure_index()
        return self._tree.edit(path, old_text, new_text, replace_all=replace_all)

    def mkdir(self, path: str) -> TreeEntry:
        self._ensure_index()
        return self._tree.mkdir(path)

    def move(self, source: str, destination: str) -> TreeEntry:
        self._ensure_index()
        return self._tree.move(source, destination)

    def delete(self, path: str) -> None:
        self._ensure_index()
        self._tree.delete(path)

    def index(self, limit_bytes: int) -> str:
        self._ensure_index()
        content, _ = self._tree.read(INDEX)
        encoded = content.encode("utf-8")
        if len(encoded) <= limit_bytes:
            return content
        return encoded[:limit_bytes].decode("utf-8", errors="ignore") + (
            f"\n[MEMORY.md is longer than {limit_bytes} bytes and was cut here. "
            "Reorganize it: keep only what you must always remember and a map of "
            "your other workspace files.]"
        )
