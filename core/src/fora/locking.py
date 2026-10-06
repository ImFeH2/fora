from __future__ import annotations

from enum import IntEnum
from threading import RLock, local
from types import TracebackType
from typing import Self


class LockLevel(IntEnum):
    MEMBER = 1
    DATABASE = 2


_held = local()


class OrderedRLock:
    def __init__(self, level: LockLevel) -> None:
        self._level = level
        self._lock = RLock()

    def acquire(self) -> None:
        locks: list[OrderedRLock] = getattr(_held, "locks", [])
        if self not in locks and any(lock._level >= self._level for lock in locks):
            raise RuntimeError(
                f"Lock order violation: cannot acquire {self._level.name}"
            )
        self._lock.acquire()
        locks.append(self)
        _held.locks = locks

    def release(self) -> None:
        locks: list[OrderedRLock] = getattr(_held, "locks", [])
        if not locks or locks[-1] is not self:
            raise RuntimeError("Locks must be released in reverse acquisition order")
        self._lock.release()
        locks.pop()

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
