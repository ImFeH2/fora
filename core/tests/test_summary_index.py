from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from fora.adapters.sqlite.agent import SqliteAgentStore
from fora.adapters.sqlite.store import SqliteStore

OLD_COLUMNS = (
    "agent_id",
    "sequence",
    "status",
    "started_at",
    "completed_at",
    "usage_json",
    "error",
    "reminded_json",
    "pending_revision",
)
SUMMARY_COLUMNS = (*OLD_COLUMNS, "run_id", "last_saved_at", "window_number")
LEGACY_SCHEMA = """
CREATE TABLE agent_runs (
    agent_id INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    messages_json TEXT NOT NULL DEFAULT '[]',
    usage_json TEXT,
    error TEXT,
    reminded_json TEXT NOT NULL DEFAULT '[]',
    pending_revision INTEGER NOT NULL DEFAULT -1,
    last_saved_at TEXT,
    window_number INTEGER,
    PRIMARY KEY (agent_id, sequence)
);
"""


def old_database(path: Path, legacy: bool) -> None:
    if legacy:
        with closing(sqlite3.connect(path)) as connection:
            connection.executescript(LEGACY_SCHEMA)
    store = SqliteStore(path)
    try:
        history = SqliteAgentStore(store._db)
        run = history.start_run(2, run_id="retained", reminded=[(1, 3)])
        history.finish_run(
            2,
            run.sequence,
            status="failed",
            messages_json='[{"text":"retained 中文"}]',
            usage_json='{"input_tokens":3}',
            error="retained error",
        )
        with store._db:
            store._db.execute("DROP INDEX agent_runs_summary")
            store._db.execute(
                f"CREATE INDEX agent_runs_summary ON agent_runs ({', '.join(OLD_COLUMNS)})"
            )
    finally:
        store.close()


@pytest.mark.parametrize("legacy", [False, True])
def test_summary_index_upgrades_old_definition_and_reuses_completed_upgrade(
    tmp_path: Path, legacy: bool
) -> None:
    path = tmp_path / "upgrade.sqlite3"
    old_database(path, legacy)
    with closing(sqlite3.connect(path)) as connection:
        original = connection.execute("SELECT * FROM agent_runs").fetchall()
    store = SqliteStore(path)
    try:
        assert (
            tuple(
                row["name"]
                for row in store._db.execute("PRAGMA index_info(agent_runs_summary)")
            )
            == SUMMARY_COLUMNS
        )
        history = SqliteAgentStore(store._db)
        assert history.run_messages(2, 1) == '[{"text":"retained 中文"}]'
        run = history.history_run(2, 1)
        assert run is not None
        assert run.run_id == "retained"
        assert run.last_saved_at is not None
        assert run.window_number == 1
        assert run.error == "retained error"
        version = store._db.execute("PRAGMA schema_version")[0][0]
    finally:
        store.close()
    for _ in range(2):
        store = SqliteStore(path)
        try:
            assert store._db.execute("PRAGMA schema_version")[0][0] == version
            assert [
                tuple(row) for row in store._db.execute("SELECT * FROM agent_runs")
            ] == original
        finally:
            store.close()


@pytest.mark.parametrize("boundary", ["create", "commit"])
def test_summary_index_upgrade_failure_rolls_back_and_closes_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    path = tmp_path / "rollback.sqlite3"
    old_database(path, True)
    connect = sqlite3.connect
    with closing(connect(path)) as connection:
        original = connection.execute("SELECT * FROM agent_runs").fetchall()
    opened = []
    failure = sqlite3.OperationalError(f"injected {boundary} failure")

    class FailingConnection(sqlite3.Connection):
        upgrading = False

        def execute(self, sql, parameters=()):
            result = super().execute(sql, parameters)
            if sql.startswith("CREATE INDEX agent_runs_summary"):
                self.upgrading = True
                if boundary == "create":
                    raise failure
            return result

        def commit(self):
            if self.upgrading:
                raise failure
            return super().commit()

    def failing_connect(*args, **kwargs):
        connection = connect(*args, **kwargs, factory=FailingConnection)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", failing_connect)
    with pytest.raises(sqlite3.OperationalError) as caught:
        SqliteStore(path)
    assert caught.value is failure
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
    with closing(connect(path)) as connection:
        assert connection.execute("SELECT * FROM agent_runs").fetchall() == original
        assert (
            tuple(
                row[2]
                for row in connection.execute("PRAGMA index_info(agent_runs_summary)")
            )
            == OLD_COLUMNS
        )
    monkeypatch.setattr(sqlite3, "connect", connect)
    SqliteStore(path).close()


def test_history_summary_index_requirement_propagates_missing_index(
    tmp_path: Path,
) -> None:
    store = SqliteStore(tmp_path / "missing-index.sqlite3")
    try:
        history = SqliteAgentStore(store._db)
        history.start_run(2)
        with store._db:
            store._db.execute("DROP INDEX agent_runs_summary")
        with pytest.raises(sqlite3.OperationalError, match="no such index"):
            history.history_run(2, 1)
        with pytest.raises(sqlite3.OperationalError, match="no such index"):
            history.history_runs(2)
    finally:
        store.close()


def test_history_summary_snapshot_during_continuous_progress_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "concurrent.sqlite3"
    writer = SqliteStore(path)
    reader = SqliteStore(path)
    try:
        history = SqliteAgentStore(writer._db)
        reading = SqliteAgentStore(reader._db)
        run = history.start_run(2)
        baseline = reading.history_run(2, run.sequence)
        assert baseline is not None

        def save() -> None:
            for index in range(20):
                monkeypatch.setattr(
                    history,
                    "_now",
                    lambda index=index: f"2026-10-06T13:01:{index:02d}Z",
                )
                history.save_progress(2, run.sequence, json.dumps([{"text": index}]))
            history.finish_run(
                2,
                run.sequence,
                status="completed",
                messages_json='[{"text":"final"}]',
                usage_json='{"input_tokens":20}',
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            with reader._db.transaction(immediate=False):
                assert reading.history_run(2, run.sequence) == baseline
                pool.submit(save).result(timeout=10)
                assert reading.history_run(2, run.sequence) == baseline
                assert reading.history_runs(2) == (baseline,)
            final = reading.history_run(2, run.sequence)
            assert final is not None
            assert final.status == "completed"
            assert final.last_saved_at == "2026-10-06T13:01:19Z"
            assert final.usage_json == '{"input_tokens":20}'
            assert reading.history_runs(2) == (final,)
            assert reading.run_messages(2, run.sequence) == '[{"text":"final"}]'
    finally:
        reader.close()
        writer.close()


def test_history_summary_tracks_saved_progress_rollback_and_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqliteStore(tmp_path / "progress.sqlite3")
    try:
        history = SqliteAgentStore(store._db)
        run = history.start_run(2)
        baseline = history.history_run(2, run.sequence)
        assert baseline is not None
        monkeypatch.setattr(history, "_now", lambda: "2026-10-06T13:00:00Z")
        history.save_progress(2, run.sequence, '[{"text":"saved"}]')
        saved = history.history_run(2, run.sequence)
        assert saved == replace(baseline, last_saved_at="2026-10-06T13:00:00Z")
        assert history.history_runs(2) == (saved,)
        with pytest.raises(ValueError, match="rollback"), history.transaction():
            monkeypatch.setattr(history, "_now", lambda: "2026-10-06T13:01:00Z")
            history.finish_run(
                2,
                run.sequence,
                status="completed",
                messages_json='[{"text":"changed"}]',
            )
            current = history.history_run(2, run.sequence)
            assert current is not None
            assert current.status == "completed"
            raise ValueError("rollback")
        assert history.history_run(2, run.sequence) == saved
        assert history.run_messages(2, run.sequence) == '[{"text":"saved"}]'
        with history.transaction():
            history._db.execute("DELETE FROM agent_runs WHERE agent_id = ?", (2,))
        assert history.history_run(2, run.sequence) is None
        assert history.history_runs(2) == ()
    finally:
        store.close()
