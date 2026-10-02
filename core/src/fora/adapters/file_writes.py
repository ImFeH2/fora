from __future__ import annotations

import os
import tempfile
from builtins import BaseExceptionGroup
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, RLock
from weakref import WeakValueDictionary

from fora.core.errors import DomainError

_directory_locks: WeakValueDictionary[tuple[int, int], RLock] = WeakValueDictionary()
_registry_lock = Lock()


def directory_lock(target: Path) -> RLock:
    identity = target.parent.stat()
    key = (identity.st_dev, identity.st_ino)
    with _registry_lock:
        lock = _directory_locks.get(key)
        if lock is None:
            lock = RLock()
            _directory_locks[key] = lock
        return lock


@contextmanager
def replace_file(
    target: Path, content: str, *, mode: int | None = None, prefix: str = "tmp"
) -> Iterator[None]:
    temporary: Path | None = None
    published = False
    primary: BaseException | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=prefix, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content.encode("utf-8"))
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, target)
        published = True
        yield
    except BaseException as error:
        primary = error
        if published and isinstance(error, Exception):
            primary = DomainError(
                "write_published", f"File already published at {target}: {error}"
            )
            raise primary from error
        raise
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                if primary is not None:
                    raise BaseExceptionGroup(
                        f"File cleanup failed at {target}; published={published}: "
                        f"{primary}; {error}",
                        [primary, error],
                    ) from None
                if published:
                    raise DomainError(
                        "write_published",
                        f"File already published at {target}: {error}",
                    ) from error
                raise


@contextmanager
def create_file_exclusive(target: Path, content: str, *, prefix: str) -> Iterator[None]:
    temporary: Path | None = None
    published = False
    primary: BaseException | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=prefix, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content.encode("utf-8"))
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise DomainError(
                "already_exists", f"Path already exists: {target}"
            ) from error
        published = True
        yield
    except BaseException as error:
        primary = error
        if published and isinstance(error, Exception):
            primary = DomainError(
                "write_published", f"File already published at {target}: {error}"
            )
            raise primary from error
        raise
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                if primary is not None:
                    raise BaseExceptionGroup(
                        f"File cleanup failed at {target}; published={published}: "
                        f"{primary}; {error}",
                        [primary, error],
                    ) from None
                if published:
                    raise DomainError(
                        "write_published",
                        f"File already published at {target}: {error}",
                    ) from error
                raise
