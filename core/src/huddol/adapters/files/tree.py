from __future__ import annotations

import hashlib
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

from huddol.adapters.execution.editing import edit_file
from huddol.adapters.file_writes import directory_lock, replace_file
from huddol.core.errors import DomainError
from huddol.ports.files import ConflictError, TreeEntry


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


class DirectoryTree:
    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(
        self, path: str, *, allow_root: bool = False, follow_symlinks: bool = True
    ) -> Path:
        if not isinstance(path, str) or "\0" in path:
            raise DomainError("invalid_path", "Path must be a string without NUL")
        if path in (".", "./") or not path.replace("/", "").strip():
            if allow_root:
                return self._root
            raise DomainError(
                "invalid_path", "Path must name a file or folder inside the tree"
            )
        path = path.removeprefix("./")
        parts = path.replace("\\", "/").split("/")
        pure = PurePosixPath(path.replace("\\", "/"))
        if PureWindowsPath(path).drive or pure.is_absolute() or ".." in parts:
            raise DomainError("invalid_path", "Path must be relative without '..'")
        entry = self._root / pure
        target = entry.resolve()
        if target == self._root or self._root not in target.parents:
            raise DomainError("invalid_path", "Path must stay inside the tree")
        return target if follow_symlinks else entry.parent.resolve() / entry.name

    def _relative(self, target: Path) -> str:
        return "" if target == self._root else target.relative_to(self._root).as_posix()

    def _content(self, target: Path) -> str:
        try:
            return target.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise DomainError("not_readable", "File cannot be read as UTF-8") from error

    def _entry(self, target: Path) -> TreeEntry:
        try:
            stat = target.stat()
        except OSError:
            if not target.is_symlink():
                raise
            stat = target.lstat()
        directory = target.is_dir()
        modified = datetime.fromtimestamp(stat.st_mtime, UTC)
        return TreeEntry(
            path=self._relative(target),
            kind="directory" if directory else "file",
            size=0 if directory else stat.st_size,
            modified_at=modified.isoformat(timespec="seconds").replace("+00:00", "Z"),
        )

    def list(self, path: str | None = None) -> tuple[TreeEntry, ...]:
        base = self._root if path is None else self._resolve(path, allow_root=True)
        if not base.is_dir():
            return ()
        found = []
        for directory, directories, files in base.walk(follow_symlinks=False):
            for name in [*directories, *files]:
                found.append(self._entry(directory / name))
        return tuple(sorted(found, key=lambda entry: entry.path.split("/")))

    def read(self, path: str) -> tuple[str, str]:
        target = self._resolve(path)
        if not target.is_file():
            raise DomainError("not_found", f"{path} does not exist")
        content = self._content(target)
        return content, content_hash(content)

    def write(
        self, path: str, content: str, *, expected_hash: str | None = None
    ) -> TreeEntry:
        if not isinstance(content, str):
            raise DomainError("invalid_content", "Content must be a string")
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with directory_lock(target):
            if target.is_dir():
                raise DomainError("invalid_path", "Path is a directory")
            if target.exists():
                current = content_hash(self._content(target))
                if expected_hash is None:
                    raise DomainError(
                        "expected_hash_required",
                        "expected_hash is required when overwriting an existing document",
                    )
                if expected_hash != current:
                    raise ConflictError(path, expected_hash, current)
            with replace_file(target, content):
                return self._entry(target)

    def edit(
        self, path: str, old_text: str, new_text: str, *, replace_all: bool = False
    ) -> tuple[TreeEntry, str]:
        target = self._resolve(path)
        if not target.parent.is_dir():
            raise DomainError("not_found", f"{path} does not exist")
        with directory_lock(target):
            self.read(path)
            result = edit_file(
                str(target),
                [
                    {
                        "old_text": old_text,
                        "new_text": new_text,
                        "replace_all": replace_all,
                    }
                ],
                directories=[str(self._root)],
            )
            try:
                return self._entry(target), result.diff
            except Exception as error:
                raise DomainError(
                    "write_published", f"File already published at {target}: {error}"
                ) from error

    def mkdir(self, path: str) -> TreeEntry:
        target = self._resolve(path, allow_root=True)
        if target.exists() and not target.is_dir():
            raise DomainError("invalid_path", "Path is a file")
        target.mkdir(parents=True, exist_ok=True)
        return self._entry(target)

    def delete(self, path: str) -> None:
        target = self._resolve(path, follow_symlinks=False)
        if not target.exists() and not target.is_symlink():
            raise DomainError("not_found", f"{path} does not exist")
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()

    def move(self, source: str, destination: str) -> TreeEntry:
        origin = self._resolve(source, follow_symlinks=False)
        if not origin.exists() and not origin.is_symlink():
            raise DomainError("not_found", f"{source} does not exist")
        target = self._resolve(destination)
        if origin.is_dir() and (target == origin or origin in target.parents):
            raise DomainError("invalid_path", "Cannot move a directory into itself")
        if (
            target.exists()
            or (self._root / destination.replace("\\", "/")).is_symlink()
        ):
            raise DomainError("already_exists", f"{destination} already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(origin, target)
        return self._entry(target)
