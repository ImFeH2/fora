from concurrent.futures import ThreadPoolExecutor

import pytest

from fora.locking import LockLevel, OrderedRLock


def test_ordered_locks_allow_nesting_and_reentry() -> None:
    member = OrderedRLock(LockLevel.MEMBER)
    database = OrderedRLock(LockLevel.DATABASE)
    with member, database, member, database:
        pass
    with database:
        pass


def test_reverse_lock_order_fails_before_waiting() -> None:
    member = OrderedRLock(LockLevel.MEMBER)
    database = OrderedRLock(LockLevel.DATABASE)

    def acquire_reverse() -> None:
        with (
            database,
            pytest.raises(RuntimeError, match="Lock order violation"),
            member,
        ):
            pytest.fail("Reverse lock order was accepted")

    with ThreadPoolExecutor(max_workers=1) as pool, member:
        pool.submit(acquire_reverse).result(timeout=5)


def test_distinct_locks_at_the_same_level_are_rejected() -> None:
    first = OrderedRLock(LockLevel.MEMBER)
    second = OrderedRLock(LockLevel.MEMBER)
    with first, pytest.raises(RuntimeError, match="Lock order violation"), second:
        pytest.fail("Unordered member locks were accepted")


def test_lock_order_is_restored_after_an_exception() -> None:
    member = OrderedRLock(LockLevel.MEMBER)
    database = OrderedRLock(LockLevel.DATABASE)
    with pytest.raises(ValueError), member, database:
        raise ValueError("Interrupted operation")
    with member, database:
        pass
    with pytest.raises(RuntimeError, match="reverse acquisition order"):
        member.release()
