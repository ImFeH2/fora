import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock

import pytest

from fora.adapters.sqlite.agent import SqliteAgentStore
from fora.adapters.sqlite.store import SqliteStore
from fora.core.errors import DomainError
from fora.locking import LockLevel, OrderedRLock


@pytest.mark.parametrize("use_reader", [False, True])
def test_member_and_database_lock_order_with_selected_connections(
    tmp_path: Path, use_reader: bool
):
    store = SqliteStore(tmp_path / "ordered.sqlite3")
    member = store.create_member("agent", "Before")
    member_lock = OrderedRLock(LockLevel.MEMBER)
    try:
        with member_lock:
            context = store._db.read() if use_reader else store._db
            with context:
                selected = store._db.current
                assert store._db.lock is selected._lock
                assert isinstance(store._db.lock, OrderedRLock)
                assert (selected is not store._db) is use_reader
                assert store.get_member(member.id).name == "Before"
                with store._db.read(), store._db.lock:
                    assert store._db.current is selected
        context = store._db.read() if use_reader else store._db
        with (
            context,
            pytest.raises(RuntimeError, match="Lock order violation"),
            member_lock,
        ):
            pytest.fail("Reverse database and member lock order was accepted")
        assert store._db.current is store._db
        assert store._db._readers.borrowed == 0
        with member_lock:
            store.rename_member(member.id, "After")
        assert store.get_member(member.id).name == "After"
    finally:
        store.close()


def test_read_connection_observes_committed_view_while_writer_runs(tmp_path: Path):
    store = SqliteStore(tmp_path / "reads.sqlite3")
    member = store.create_member("agent", "Before")
    entered = Event()
    release = Event()

    def write():
        with store._db:
            store.rename_member(member.id, "After")
            entered.set()
            assert release.wait(5)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            writer = executor.submit(write)
            try:
                assert entered.wait(5)
                reader = executor.submit(store.get_member, member.id)
                assert reader.result(timeout=1).name == "Before"
            finally:
                release.set()
                writer.result(timeout=5)
        assert store.get_member(member.id).name == "After"
        assert store._db._readers.borrowed == 0
    finally:
        release.set()
        store.close()


def test_nested_read_snapshot_and_writer_transaction_have_correct_views(tmp_path: Path):
    store = SqliteStore(tmp_path / "snapshot.sqlite3")
    member = store.create_member("agent", "Before")
    try:
        with store._db.read():
            connection = store._db.current
            assert store.get_member(member.id).name == "Before"
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(store.rename_member, member.id, "After").result(
                    timeout=5
                )
            with store._db.read():
                assert store._db.current is connection
                assert store.get_member(member.id).name == "Before"
        assert store.get_member(member.id).name == "After"
        with pytest.raises(ValueError), store._db:
            store.rename_member(member.id, "Uncommitted")
            with store._db.read():
                assert store._db.current is store._db
                assert store.get_member(member.id).name == "Uncommitted"
            raise ValueError("Rollback")
        assert store.get_member(member.id).name == "After"
        assert store._db.current is store._db
        assert store._db._readers.borrowed == 0
    finally:
        store.close()


def test_read_context_rejects_actual_sql_writes_and_releases_failure(tmp_path: Path):
    store = SqliteStore(tmp_path / "readonly.sqlite3")
    try:
        with (
            pytest.raises(sqlite3.OperationalError, match="readonly"),
            store._db.read(),
        ):
            store._db.execute(
                "INSERT INTO members VALUES (1, 'human', 'You', 'you', 0, 'idle')"
            )
        assert store._db.current is store._db
        assert store._db._readers.borrowed == 0
        assert store.list_members() == ()
        assert store.create_member("human", "You").id == 1
        with store._db.read():
            assert store._db.execute("PRAGMA query_only")[0][0] == 1
            assert store._db.execute("PRAGMA journal_mode")[0][0] == "wal"
    finally:
        store.close()


def test_read_pool_waits_and_close_waits_for_borrowed_connections(tmp_path: Path):
    store = SqliteStore(tmp_path / "pool.sqlite3")
    release = Event()
    all_entered = Event()
    waiting_started = Event()
    lock = Lock()
    entered = 0
    identities = set()

    def read():
        nonlocal entered
        with store._db.read():
            with lock:
                identities.add(id(store._db.current))
                entered += 1
                if entered == 8:
                    all_entered.set()
            assert release.wait(5)
            return store.list_members()

    def waiting():
        waiting_started.set()
        return store.list_members()

    try:
        with ThreadPoolExecutor(max_workers=10) as executor:
            readers = [executor.submit(read) for _ in range(8)]
            try:
                assert all_entered.wait(5)
                assert len(identities) == store._db._readers.borrowed == 8
                last = executor.submit(waiting)
                assert waiting_started.wait(5)
                with pytest.raises(TimeoutError):
                    last.result(timeout=0.1)
            finally:
                release.set()
                assert [reader.result(timeout=5) for reader in readers] == [()] * 8
            assert last.result(timeout=5) == ()
            assert len(store._db._readers.connections) == 8
        release.clear()
        held = Event()

        def holding():
            with store._db.read():
                held.set()
                assert release.wait(5)

        with ThreadPoolExecutor(max_workers=2) as executor:
            reader = executor.submit(holding)
            try:
                assert held.wait(5)
                closer = executor.submit(store.close)
                with pytest.raises(TimeoutError):
                    closer.result(timeout=0.1)
            finally:
                release.set()
                reader.result(timeout=5)
            closer.result(timeout=5)
        assert not store._db._readers.connections
        assert store._db._readers.borrowed == 0
    finally:
        release.set()
        store.close()


def test_history_prepares_immutable_unicode_content_before_write_transaction(
    tmp_path: Path,
):
    store = SqliteStore(tmp_path / "history.sqlite3")
    history = SqliteAgentStore(store._db)
    run = history.start_run(2)
    owners = []

    class Content(str):
        def encode(self, encoding="utf-8", errors="strict"):
            owners.append(store._db._owner)
            return super().encode(encoding, errors)

    try:
        content = Content('[{"content":"中文🙂"}]')
        handle = history.start_model_request(
            2,
            run.sequence,
            run.run_id,
            1,
            [content, content],
            Content("{}"),
            Content("{}"),
            Content("{}"),
            False,
        )
        history.finish_model_request(2, run.sequence, handle, [content])
        history.link_model_request(2, run.sequence, handle, [content])
        assert owners == [None] * 7
        summary = history.model_request_summary(2, run.sequence, handle.ordinal)
        assert (
            summary.input_count == 2
            and summary.response_count == summary.related_count == 1
        )
        assert summary.input_length == 2 * len(str(content).encode("utf-8"))
        assert len(store._db.execute("SELECT * FROM agent_history_message_blobs")) == 1
        wrong = type(handle)(handle.ordinal, "wrong-identity")
        with pytest.raises(DomainError, match="does not exist"):
            history.finish_model_request(
                2, run.sequence, wrong, ['[{"content":"Uncommitted"}]']
            )
        assert len(store._db.execute("SELECT * FROM agent_history_message_blobs")) == 1
    finally:
        store.close()
