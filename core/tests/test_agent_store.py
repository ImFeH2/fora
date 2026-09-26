from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from huddol.adapters.model.config import ModelCatalog
from huddol.adapters.sqlite.agent import SqliteAgentStore
from huddol.adapters.sqlite.store import SqliteStore
from huddol.core.errors import DomainError
from huddol.ports.agent import RunSummary, WindowState
from huddol.services.history import History

AGENT = 13


@pytest.fixture
def agent_store(tmp_path: Path) -> SqliteAgentStore:
    base = SqliteStore(tmp_path / "huddol.sqlite3")
    yield SqliteAgentStore(base._db)
    base.close()


def test_history_reads_unicode_slices_without_loading_or_changing_the_run(
    agent_store: SqliteAgentStore,
) -> None:
    payload = '[{"text":"漢\\\\n' + "x" * 3_000_000 + '終"}]'
    run = agent_store.start_run(AGENT)
    agent_store.finish_run(
        AGENT, run.sequence, status="completed", messages_json=payload
    )
    history = History(agent_store, AGENT)
    before = agent_store.runs(AGENT)[0].messages_json
    chunks = []
    offset = 0
    while True:
        part = history.read(run.sequence, offset)
        assert part is not None
        assert len(part.messages) <= 2048
        chunks.append(part.messages)
        offset += len(part.messages)
        if offset == part.total_length:
            break
    assert "".join(chunks) == payload
    assert agent_store.runs(AGENT)[0].messages_json == before
    final = history.read(run.sequence, len(payload))
    assert final is not None
    assert final.messages == ""
    with pytest.raises(DomainError):
        history.read(run.sequence, len(payload) + 1)
    with pytest.raises(DomainError):
        history.read(run.sequence, True)
    with pytest.raises(DomainError):
        history.read(run.sequence, 2**63)


def test_history_reads_a_run_older_than_the_recent_run_window(
    agent_store: SqliteAgentStore,
) -> None:
    original = agent_store.start_run(AGENT)
    agent_store.finish_run(
        AGENT, original.sequence, status="completed", messages_json='[{"text":"old"}]'
    )
    for _ in range(1000):
        agent_store.start_run(AGENT)
    part = History(agent_store, AGENT).read(original.sequence)
    assert part is not None
    assert part.messages == '[{"text":"old"}]'


def test_runs_append_and_report_the_latest_history(
    agent_store: SqliteAgentStore,
) -> None:
    first = agent_store.start_run(AGENT)
    agent_store.finish_run(
        AGENT, first.sequence, status="completed", messages_json='[{"kind":"request"}]'
    )
    second = agent_store.start_run(AGENT)
    assert second.sequence == 2
    assert agent_store.latest_messages(AGENT) == '[{"kind":"request"}]'


def test_reminder_snapshot_survives_upgrade_restart_and_window_reset(tmp_path) -> None:
    path = tmp_path / "huddol.sqlite3"
    base = SqliteStore(path)
    store = SqliteAgentStore(base._db)
    legacy = store.start_run(AGENT)
    store.finish_run(AGENT, legacy.sequence, status="completed", messages_json="[]")
    base._db.execute("DROP INDEX agent_runs_summary")
    base._db.execute("ALTER TABLE agent_runs DROP COLUMN reminded_json")
    base._db.commit()
    base.close()

    base = SqliteStore(path)
    store = SqliteAgentStore(base._db)
    assert store.runs(AGENT)[0].sequence == legacy.sequence
    assert store.last_reminder(AGENT) == frozenset()
    store.start_run(AGENT, reminded=[(2, 1), (1, 1), (2, 1)])
    base.close()

    base = SqliteStore(path)
    try:
        store = SqliteAgentStore(base._db)
        assert store.mark_interrupted() == 1
        assert store.last_reminder(AGENT) == frozenset({(1, 1), (2, 1)})
        assert store.last_reminder(AGENT + 1) == frozenset()
        store.start_run(AGENT)
        store.reset_window(AGENT, "prepared")
        assert store.last_reminder(AGENT) == frozenset({(1, 1), (2, 1)})
        store.start_run(AGENT, reminded=[(2, 1)])
        assert store.last_reminder(AGENT) == frozenset({(2, 1)})
    finally:
        base.close()


def test_a_new_session_reminds_the_same_keys_again(tmp_path) -> None:
    path = tmp_path / "huddol.sqlite3"
    base = SqliteStore(path)
    store = SqliteAgentStore(base._db)
    run = store.start_run(AGENT, reminded=[(1, 1)])
    store.finish_run(AGENT, run.sequence, status="completed", messages_json="[]")
    assert store.last_reminder(AGENT) == frozenset({(1, 1)})
    assert store.mark_session_start() == 1
    assert store.last_reminder(AGENT) == frozenset()
    base.close()

    base = SqliteStore(path)
    store = SqliteAgentStore(base._db)
    try:
        assert store.last_reminder(AGENT) == frozenset()
        assert store.mark_session_start() == 1
        assert store.last_reminder(AGENT) == frozenset()
        assert store.last_reminder(AGENT + 1) == frozenset()
        store.start_run(AGENT, reminded=[(2, 3)])
        assert store.last_reminder(AGENT) == frozenset({(2, 3)})
        store.pause_for_safety(AGENT, "runtime_error")
        store.reset_safety(AGENT)
        assert store.last_reminder(AGENT) == frozenset()
    finally:
        base.close()


def test_safety_pause_and_resume_boundary_survive_reopening(tmp_path) -> None:
    path = tmp_path / "huddol.sqlite3"
    base = SqliteStore(path)
    store = SqliteAgentStore(base._db)
    run = store.start_run(AGENT, reminded=[(1, 1)])
    store.finish_run(
        AGENT,
        run.sequence,
        status="completed",
        messages_json="[]",
        usage_json='{"tool_calls":0}',
    )
    store.pause_for_safety(AGENT, "no_tool_calls")
    base.close()
    base = SqliteStore(path)
    store = SqliteAgentStore(base._db)
    assert store.pause_reason(AGENT) == "no_tool_calls"
    assert store.no_tool_streak(AGENT) == 1
    store.reset_window(AGENT, "prepared")
    assert store.pause_reason(AGENT) == "no_tool_calls"
    store.reset_safety(AGENT)
    base.close()
    base = SqliteStore(path)
    try:
        store = SqliteAgentStore(base._db)
        assert store.pause_reason(AGENT) is None
        assert store.no_tool_streak(AGENT) == 0
        assert store.last_reminder(AGENT) == frozenset()
        assert len(store.runs(AGENT)) == 1
    finally:
        base.close()


def test_windows_default_and_reset_without_any_runs(agent_store) -> None:
    assert agent_store.window(AGENT) == WindowState(1, 1, None, None)
    assert agent_store.latest_messages(AGENT) == "[]"
    state = agent_store.reset_window(AGENT, "prepared")
    assert state == WindowState(2, 1, state.reset_at, "prepared")
    assert state.reset_at is not None
    assert agent_store.window(AGENT) == state
    assert agent_store.window(AGENT + 1) == WindowState(1, 1, None, None)
    state = agent_store.reset_window(AGENT, "overflow")
    assert state == WindowState(3, 1, state.reset_at, "overflow")


def test_windows_scope_messages_but_keep_history_usage_and_effects(agent_store) -> None:
    first = agent_store.start_run(AGENT)
    agent_store.finish_run(
        AGENT,
        first.sequence,
        status="completed",
        messages_json='[{"text":"old"}]',
        usage_json='{"input_tokens":100,"output_tokens":20}',
    )
    agent_store.record_effect(AGENT, first.sequence, "send", "old message")
    agent_store.start_run(AGENT)
    for _ in range(5):
        agent_store.start_run(AGENT + 1)
    state = agent_store.reset_window(AGENT, "prepared")
    assert state.number == 2
    assert state.since_sequence == 3
    assert agent_store.latest_messages(AGENT) == "[]"
    assert len(agent_store.runs(AGENT)) == 2
    assert agent_store.search_runs(AGENT, "old")[0].sequence == first.sequence
    assert agent_store.usage_total(AGENT)["total_tokens"] == 120
    assert agent_store.effects(AGENT)[0].summary == "old message"
    run = agent_store.start_run(AGENT)
    assert run.sequence == state.since_sequence
    assert agent_store.latest_messages(AGENT) == "[]"
    agent_store.finish_run(
        AGENT, run.sequence, status="completed", messages_json='[{"text":"new"}]'
    )
    agent_store.start_run(AGENT)
    assert agent_store.latest_messages(AGENT) == '[{"text":"new"}]'
    state = agent_store.reset_window(AGENT, "overflow")
    assert state.number == 3 and state.since_sequence == 5
    assert agent_store.latest_messages(AGENT) == "[]"
    reopened = SqliteAgentStore(agent_store._db)
    assert reopened.window(AGENT) == state
    assert reopened.latest_messages(AGENT) == "[]"
    assert len(reopened.runs(AGENT)) == 4
    assert reopened.search_runs(AGENT, "old")[0].sequence == first.sequence


@pytest.mark.parametrize("finished", [False, True])
def test_save_progress_updates_only_the_target_runs_messages(
    agent_store, finished
) -> None:
    first = agent_store.start_run(AGENT)
    other = agent_store.start_run(AGENT + 1)
    if finished:
        agent_store.finish_run(
            AGENT,
            first.sequence,
            status="failed",
            messages_json="[]",
            usage_json='{"requests":1}',
            error="failure",
        )
    original = agent_store.runs(AGENT)[0]
    second = agent_store.start_run(AGENT)
    for messages in ('[{"text":"first"}]', '[{"text":"second"}]'):
        agent_store.save_progress(AGENT, first.sequence, messages)
        assert agent_store.runs(AGENT) == (
            second,
            replace(original, messages_json=messages),
        )
        assert agent_store.runs(AGENT + 1) == (other,)


def test_unfinished_runs_are_marked_interrupted(agent_store: SqliteAgentStore) -> None:
    prior = agent_store.start_run(AGENT)
    agent_store.finish_run(
        AGENT, prior.sequence, status="completed", messages_json='[{"text":"old"}]'
    )
    run = agent_store.start_run(AGENT)
    partial = '[{"text":"partial"}]'
    agent_store.save_progress(AGENT, run.sequence, partial)
    assert agent_store.mark_interrupted() == 1
    assert agent_store.runs(AGENT)[0].status == "interrupted"
    assert agent_store.runs(AGENT)[0].messages_json == partial
    agent_store.start_run(AGENT)
    assert agent_store.latest_messages(AGENT) == partial


def test_history_search_finds_runs_by_content(agent_store: SqliteAgentStore) -> None:
    run = agent_store.start_run(AGENT)
    agent_store.finish_run(
        AGENT, run.sequence, status="completed", messages_json='[{"text":"bubblewrap"}]'
    )
    history = History(agent_store, AGENT)
    assert len(history.search("bubblewrap")) == 1
    assert history.search("nothing") == ()


def test_model_catalog_migration_preserves_identity_across_restarts(tmp_path) -> None:
    path = tmp_path / "migration.sqlite3"
    base = SqliteStore(path)
    store = SqliteAgentStore(base._db)
    store.set_settings(
        "model",
        {
            "api_type": "google",
            "base_url": "https://example.invalid",
            "api_key": "test-only-key",
            "model": "gemini-2.5-pro",
        },
    )
    base.close()
    base = SqliteStore(path)
    store = SqliteAgentStore(base._db)
    converted = store.get_settings("model")
    catalog = ModelCatalog.restore(converted)
    assert catalog.version == 2
    assert catalog.resolve(2).model == "gemini-2.5-pro"
    assert catalog.resolve(3).api_key == "test-only-key"
    assert catalog.default_model_id == catalog.models[0].id
    base.close()
    base = SqliteStore(path)
    try:
        store = SqliteAgentStore(base._db)
        assert store.get_settings("model") == converted
        assert "test-only-key" not in str(ModelCatalog.restore(converted).redacted())
    finally:
        base.close()


def test_settings_updates_preserve_concurrent_changes(agent_store) -> None:
    def increment(_: int) -> None:
        def update(values):
            return {"count": (values or {}).get("count", 0) + 1}

        agent_store.update_settings("counter", update)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(increment, range(100)))
    assert agent_store.get_settings("counter") == {"count": 100}


def test_rejected_settings_update_preserves_stored_value(agent_store) -> None:
    agent_store.set_settings("counter", {"count": 1})

    def reject(values):
        values["count"] = 2
        raise ValueError("Rejected update")

    with pytest.raises(ValueError, match="Rejected update"):
        agent_store.update_settings("counter", reject)
    assert agent_store.get_settings("counter") == {"count": 1}


def test_settings_round_trip_without_a_directory_table(
    agent_store: SqliteAgentStore,
) -> None:
    agent_store.set_settings("model", {"model": "claude-opus-5", "compaction": 200})
    assert agent_store.get_settings("model") == {
        "model": "claude-opus-5",
        "compaction": 200,
    }
    assert agent_store.get_settings("missing") is None

    tables = {
        str(row["name"])
        for row in agent_store._db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert "write_directories" not in tables


@pytest.mark.parametrize("section", ["agent", "model", "execution", "observability"])
@pytest.mark.parametrize(
    "raw", ["null", "[]", "false", "0", '"private-value"', '{"private-value":']
)
def test_invalid_settings_preserve_stored_data_across_restarts(
    tmp_path, section, raw
) -> None:
    path = tmp_path / "huddol.sqlite3"
    base = SqliteStore(path)
    try:
        SqliteAgentStore(base._db)
        with base._db:
            base._db.execute(
                "INSERT INTO settings (section, values_json) VALUES (?, ?)"
                " ON CONFLICT (section) DO UPDATE SET values_json = excluded.values_json",
                (section, raw),
            )
    finally:
        base.close()
    base = SqliteStore(path)
    try:
        with pytest.raises(DomainError) as error:
            store = SqliteAgentStore(base._db)
            store.get_settings(section)
        assert error.value.code == "invalid_setting"
        assert section in str(error.value)
        assert "private-value" not in str(error.value)
        if raw.startswith("{"):
            assert isinstance(error.value.__cause__, json.JSONDecodeError)
        assert (
            base._db.execute(
                "SELECT values_json FROM settings WHERE section = ?", (section,)
            )[0]["values_json"]
            == raw
        )
    finally:
        base.close()


def test_effects_are_numbered_per_turn_and_read_back_in_order(
    agent_store: SqliteAgentStore,
) -> None:
    agent_store.start_run(AGENT)
    agent_store.record_effect(AGENT, 1, "run", "pytest exited 0")
    agent_store.record_effect(AGENT, 1, "send", "Discussion 2: message 9")
    agent_store.start_run(AGENT)
    agent_store.record_effect(AGENT, 2, "ack", "Discussion 2: 1 acknowledged")

    everything = agent_store.effects(AGENT)
    assert [(item.sequence, item.ordinal, item.tool) for item in everything] == [
        (2, 1, "ack"),
        (1, 1, "run"),
        (1, 2, "send"),
    ]

    only_first = agent_store.effects(AGENT, sequences=[1])
    assert [item.tool for item in only_first] == ["run", "send"]


def test_effects_are_scoped_to_one_agent(agent_store: SqliteAgentStore) -> None:
    agent_store.record_effect(AGENT, 1, "send", "mine")
    agent_store.record_effect(AGENT + 1, 1, "send", "theirs")
    assert [item.summary for item in agent_store.effects(AGENT)] == ["mine"]


def test_usage_totals_add_up_across_turns(agent_store: SqliteAgentStore) -> None:
    import json as _json

    for inbound, outbound in ((100, 20), (300, 50)):
        run = agent_store.start_run(AGENT)
        agent_store.finish_run(
            AGENT,
            run.sequence,
            status="completed",
            messages_json="[]",
            usage_json=_json.dumps(
                {
                    "input_tokens": inbound,
                    "output_tokens": outbound,
                    "cache_read_tokens": 5,
                    "requests": 1,
                }
            ),
        )

    total = agent_store.usage_total(AGENT)
    assert total["input_tokens"] == 400
    assert total["output_tokens"] == 70
    assert total["total_tokens"] == 470
    assert total["requests"] == 2


@pytest.mark.parametrize("size", [0, 4096, 1048576])
def test_metadata_queries_cover_history_and_preserve_records(agent_store, size) -> None:
    payload = json.dumps([{"text": "x" * size}])
    for status in ("completed", "failed", "interrupted"):
        run = agent_store.start_run(AGENT, reminded=[(1, 2)])
        agent_store.finish_run(
            AGENT,
            run.sequence,
            status=status,
            messages_json=payload,
            usage_json='{"input_tokens":100,"output_tokens":20,"tool_calls":0}',
            error="test" if status == "failed" else None,
        )
    agent_store.start_run(AGENT)
    agent_store.start_run(AGENT + 1)
    expected = tuple(
        RunSummary(
            run.sequence,
            run.status,
            run.started_at,
            run.completed_at,
            run.usage_json,
            run.error,
        )
        for run in agent_store.runs(AGENT)
    )
    connection = agent_store._db._connection
    statements = []

    def authorize(action, table, column, database, source):
        if (
            action == sqlite3.SQLITE_READ
            and table == "agent_runs"
            and column == "messages_json"
        ):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(authorize)
    connection.set_trace_callback(statements.append)
    try:
        assert agent_store.run_summaries(AGENT) == expected
        assert agent_store.run_summaries(AGENT, limit=1) == expected[:1]
        assert agent_store.run_summaries(AGENT, limit=0) == ()
        assert agent_store.run_summaries(999) == ()
        assert agent_store.usage_total(AGENT)["total_tokens"] == 360
        assert agent_store.no_tool_streak(AGENT) == 1
        assert agent_store.last_reminder(AGENT) == frozenset({(1, 2)})
        assert agent_store.repeated_turns(AGENT, [(1, 2)]) == 1
    finally:
        connection.set_trace_callback(None)
        connection.set_authorizer(None)
    for sql in statements:
        if not sql.startswith("SELECT") or "FROM agent_runs" not in sql:
            continue
        plan = connection.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
        assert any("COVERING INDEX agent_runs_summary" in row[3] for row in plan)
    assert agent_store.latest_messages(AGENT) == payload
    assert agent_store.runs(AGENT)[1].messages_json == payload


def test_summary_index_tracks_commits_and_rollbacks(agent_store) -> None:
    run = agent_store.start_run(AGENT)
    baseline = agent_store.run_summaries(AGENT)
    with pytest.raises(ValueError, match="rollback"), agent_store._db:
        agent_store.finish_run(
            AGENT,
            run.sequence,
            status="completed",
            messages_json='[{"text":"a"}]',
            usage_json='{"input_tokens":7}',
        )
        assert agent_store.usage_total(AGENT)["total_tokens"] == 7
        assert agent_store.run_summaries(AGENT)[0].status == "completed"
        raise ValueError("rollback")
    assert agent_store.run_summaries(AGENT) == baseline
    assert agent_store.usage_total(AGENT)["total_tokens"] == 0
    for tokens in (7, 11):
        agent_store.finish_run(
            AGENT,
            run.sequence,
            status="completed",
            messages_json='[{"text":"a"}]',
            usage_json=json.dumps({"input_tokens": tokens}),
        )
        assert agent_store.usage_total(AGENT)["total_tokens"] == tokens
    summary = agent_store.run_summaries(AGENT)
    agent_store.save_progress(AGENT, run.sequence, '[{"text":"b"}]')
    assert agent_store.run_summaries(AGENT) == summary
    assert agent_store.runs(AGENT)[0].messages_json == '[{"text":"b"}]'
    with agent_store._db:
        agent_store._db.execute("DELETE FROM agent_runs WHERE agent_id = ?", (AGENT,))
    assert agent_store.run_summaries(AGENT) == ()
    assert agent_store.usage_total(AGENT)["total_tokens"] == 0


def test_summary_index_preserves_invalid_usage_errors(agent_store) -> None:
    run = agent_store.start_run(AGENT)
    agent_store.finish_run(
        AGENT,
        run.sequence,
        status="failed",
        messages_json="[]",
        usage_json="{invalid",
    )
    assert agent_store.run_summaries(AGENT)[0].usage_json == "{invalid"
    with pytest.raises(sqlite3.OperationalError, match="malformed JSON"):
        agent_store.usage_total(AGENT)


@pytest.mark.parametrize("legacy", [False, True])
def test_summary_index_upgrade_preserves_history_and_is_idempotent(
    tmp_path, legacy
) -> None:
    path = tmp_path / "upgrade.sqlite3"
    base = SqliteStore(path)
    history = SqliteAgentStore(base._db)
    run = history.start_run(AGENT)
    history.finish_run(
        AGENT,
        run.sequence,
        status="failed",
        messages_json='[{"text":"retained"}]',
        usage_json='{"input_tokens":3}',
        error="failure",
    )
    original = history.runs(AGENT)
    base._db.execute("DROP INDEX agent_runs_summary")
    if legacy:
        base._db.execute("ALTER TABLE agent_runs DROP COLUMN reminded_json")
    base._db.commit()
    base.close()
    for _ in range(2):
        base = SqliteStore(path)
        try:
            history = SqliteAgentStore(base._db)
            assert history.runs(AGENT) == original
            assert history.usage_total(AGENT)["total_tokens"] == 3
            assert history.run_summaries(AGENT)[0].error == "failure"
            indexes = base._db.execute(
                "SELECT name FROM sqlite_schema WHERE name='agent_runs_summary'"
            )
            assert len(indexes) == 1
        finally:
            base.close()


def test_initialization_write_contention_releases_connection(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "busy.sqlite3"
    SqliteStore(path).close()
    connect = sqlite3.connect
    blocker = connect(path)
    blocker.execute("BEGIN IMMEDIATE")
    opened = []

    def short_timeout(*args, **kwargs):
        connection = connect(*args, **kwargs, timeout=0.01)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", short_timeout)
    try:
        with pytest.raises(sqlite3.OperationalError) as failure:
            SqliteStore(path)
        assert failure.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
        assert len(opened) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            opened[0].execute("SELECT 1")
    finally:
        blocker.rollback()
        blocker.close()
        monkeypatch.setattr(sqlite3, "connect", connect)
    SqliteStore(path).close()


def test_turns_without_usage_do_not_break_the_total(
    agent_store: SqliteAgentStore,
) -> None:
    run = agent_store.start_run(AGENT)
    agent_store.finish_run(AGENT, run.sequence, status="failed", messages_json="[]")
    assert agent_store.usage_total(AGENT)["total_tokens"] == 0
