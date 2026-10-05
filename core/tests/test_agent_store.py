from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from fora.adapters.model.config import ModelCatalog
from fora.adapters.sqlite.agent import SqliteAgentStore
from fora.adapters.sqlite.store import SqliteStore
from fora.core.errors import DomainError
from fora.ports.agent import RunSummary, WindowState
from fora.services.history import History

AGENT = 13


@pytest.fixture
def agent_store(tmp_path: Path) -> SqliteAgentStore:
    base = SqliteStore(tmp_path / "fora.sqlite3")
    yield SqliteAgentStore(base._db)
    base.close()


def test_history_reads_unicode_slices_without_loading_or_changing_the_run(
    agent_store: SqliteAgentStore,
) -> None:
    prefix = '[{"text":"'
    payload = prefix + "x" * (2047 - len(prefix)) + "\\n漢" + "x" * 8192 + '終"}]'
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
        assert part.offset == offset
        assert part.total_length == len(payload)
        assert part.messages == payload[offset : offset + 2048]
        assert len(part.messages) <= 2048
        chunks.append(part.messages)
        offset += len(part.messages)
        if offset == part.total_length:
            break
    assert "".join(chunks) == payload
    assert chunks[0].endswith("\\")
    assert chunks[1].startswith("n漢")
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


def test_history_reads_bounded_slices_from_a_large_run(
    agent_store: SqliteAgentStore,
) -> None:
    prefix = '[{"text":"'
    payload = prefix + "x" * (2047 - len(prefix)) + "\\n漢" + "x" * 3_000_000 + '終"}]'
    run = agent_store.start_run(AGENT)
    agent_store.finish_run(
        AGENT, run.sequence, status="completed", messages_json=payload
    )
    history = History(agent_store, AGENT)
    before = agent_store.runs(AGENT)[0].messages_json
    for offset in (0, 2048, len(payload) // 2, len(payload) - 2048, len(payload) - 1):
        part = history.read(run.sequence, offset)
        assert part is not None
        assert part.offset == offset
        assert part.total_length == len(payload)
        assert part.messages == payload[offset : offset + 2048]
        assert len(part.messages) <= 2048
    tail = history.read(run.sequence)
    assert tail is not None
    assert tail.offset == len(payload) - 2048
    assert tail.total_length == len(payload)
    assert tail.messages == payload[-2048:]
    final = history.read(run.sequence, len(payload))
    assert final is not None
    assert final.offset == len(payload)
    assert final.total_length == len(payload)
    assert final.messages == ""
    for offset in (len(payload) + 1, True, 2**63):
        with pytest.raises(DomainError):
            history.read(run.sequence, offset)
    assert agent_store.runs(AGENT)[0].messages_json == before == payload


def test_history_reads_a_run_older_than_the_recent_run_window(
    agent_store: SqliteAgentStore,
) -> None:
    original = agent_store.start_run(AGENT)
    agent_store.finish_run(
        AGENT, original.sequence, status="completed", messages_json='[{"text":"old"}]'
    )
    with agent_store._db:
        for _ in range(1000):
            agent_store.start_run(AGENT)
    recent = agent_store.runs(AGENT)
    assert len(recent) == 50
    assert [run.sequence for run in recent] == list(range(1001, 951, -1))
    assert original.sequence not in {run.sequence for run in recent}
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


def test_model_request_snapshots_keep_identity_payloads_and_progress_time(
    agent_store: SqliteAgentStore,
) -> None:
    run = agent_store.start_run(AGENT)
    handle = agent_store.start_model_request(
        AGENT,
        run.sequence,
        run.run_id,
        3,
        '[{"kind":"request"}]',
        '{"function_tools":[{"name":"tool"}]}',
        '{"temperature":0.2}',
        '{"model":"test-model"}',
        False,
    )
    summaries = agent_store.model_request_summaries(AGENT, run.sequence)
    assert summaries[0].ordinal == handle.ordinal == 1
    assert summaries[0].request_id == handle.request_id
    assert summaries[0].status == "pending"
    assert summaries[0].response_length == 0
    agent_store.finish_model_request(
        AGENT, run.sequence, handle, '[{"kind":"response"}]'
    )
    record = agent_store.model_request(AGENT, run.sequence, handle.ordinal)
    assert record is not None
    assert record.summary.status == "responded"
    assert record.summary.run_id == run.run_id
    assert record.summary.input_count == 1
    assert record.summary.response_count == 1
    assert agent_store.model_request_messages(
        AGENT, run.sequence, handle.ordinal, "input", 0, 10
    ).messages == ('[{"kind":"request"}]',)
    assert agent_store.model_request_messages(
        AGENT, run.sequence, handle.ordinal, "response", 0, 10
    ).messages == ('[{"kind":"response"}]',)
    history_run = agent_store.history_run(AGENT, run.sequence)
    assert history_run is not None
    assert history_run.window_number == run.window_number == 1
    assert history_run.request_count == 1
    assert history_run.last_saved_at is not None
    assert agent_store.history_runs(AGENT, limit=1)[0] == history_run


def test_window_events_and_run_window_numbers_survive_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "window-events.sqlite3"
    store = SqliteStore(path)
    agent_store = SqliteAgentStore(store._db)
    first = agent_store.start_run(AGENT)
    state = agent_store.reset_window(AGENT, "overflow")
    second = agent_store.start_run(AGENT)
    assert first.window_number == 1
    assert second.window_number == state.number == 2
    assert [event.number for event in agent_store.window_events(AGENT)] == [1, 2]
    assert agent_store.window_events(AGENT)[1].reason == "overflow"
    store.close()
    restored = SqliteStore(path)
    restored_agent_store = SqliteAgentStore(restored._db)
    assert restored_agent_store.runs(AGENT, limit=2)[0].window_number == 2
    assert (
        restored_agent_store.window_events(AGENT)[1].since_sequence
        == state.since_sequence
    )
    restored.close()


@pytest.mark.parametrize("legacy", [False, True])
def test_overflow_context_upgrade_restart_and_prepared_reset(tmp_path, legacy):
    path = tmp_path / "overflow.sqlite3"
    base = SqliteStore(path)
    store = SqliteAgentStore(base._db)
    first = store.start_run(AGENT, reminded=[(1, 1)])
    original = '[{"original":"中文🙂"}]'
    store.finish_run(
        AGENT,
        first.sequence,
        status="failed",
        messages_json=original,
        usage_json='{"last_input_tokens":230001}',
        error="original error",
    )
    context = "Failed Turn sequence: 1\nTool name: discussion\nArguments: 中文🙂\nFinal error: too long"
    overflow = store.reset_window(AGENT, "overflow", context)
    store.reset_window(AGENT + 1, "overflow", "another Agent")
    if legacy:
        for table in ("agent_windows", "agent_window_events"):
            base._db.execute(f"ALTER TABLE {table} DROP COLUMN overflow_context")
    first_record = store.runs(AGENT)[0]
    base.close()
    for _ in range(2):
        base = SqliteStore(path)
        store = SqliteAgentStore(base._db)
        state = store.window(AGENT)
        assert state.number == overflow.number
        assert state.since_sequence == overflow.since_sequence
        assert state.reset_at == overflow.reset_at
        assert state.overflow_context == (None if legacy else context)
        assert store.window_events(AGENT)[-1].overflow_context == state.overflow_context
        assert store.runs(AGENT)[0] == first_record
        assert store.latest_messages(AGENT) == "[]"
        for table in ("agent_windows", "agent_window_events"):
            assert [
                row["name"] for row in base._db.execute(f"PRAGMA table_info({table})")
            ].count("overflow_context") == 1
        base.close()
    base = SqliteStore(path)
    store = SqliteAgentStore(base._db)
    try:
        state = store.reset_window(AGENT, "overflow", context)
        assert store.window(AGENT + 1).overflow_context == (
            None if legacy else "another Agent"
        )
        store.reset_window(AGENT, "prepared", "ignored data")
        assert store.window(AGENT).overflow_context is None
        assert store.window_events(AGENT)[-1].overflow_context is None
        assert store.window_events(AGENT)[-2].overflow_context == context
        assert store.window_events(AGENT)[-2].reset_at == state.reset_at
        assert store.runs(AGENT)[0] == first_record
        assert store.last_reminder(AGENT) == frozenset({(1, 1)})
    finally:
        base.close()


def test_overflow_context_byte_budget_fails_before_window_changes(agent_store):
    accepted = "🙂" * 1024
    state = agent_store.reset_window(AGENT, "overflow", accepted)
    assert state.overflow_context == accepted
    events = agent_store.window_events(AGENT)
    with pytest.raises(ValueError, match="UTF-8 byte budget"):
        agent_store.reset_window(AGENT, "overflow", accepted + "x")
    assert agent_store.window(AGENT) == state
    assert agent_store.window_events(AGENT) == events


@pytest.mark.parametrize("table", ["agent_window_events", "agent_windows"])
def test_overflow_context_and_window_roll_back_together(agent_store, table):
    previous = agent_store.reset_window(AGENT, "overflow", "previous context")
    events = agent_store.window_events(AGENT)
    agent_store._db.execute(
        f"CREATE TEMP TRIGGER reject_context BEFORE INSERT ON {table} BEGIN SELECT RAISE(ABORT, 'context save failed'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="context save failed"):
        agent_store.reset_window(AGENT, "overflow", "new context")
    assert agent_store.window(AGENT) == previous
    assert agent_store.window_events(AGENT) == events


def test_model_request_message_blobs_are_deduplicated_across_requests(
    agent_store: SqliteAgentStore,
) -> None:
    run = agent_store.start_run(AGENT)
    payload = "x" * (5 * 1024 * 1024)
    first = agent_store.start_model_request(
        AGENT, run.sequence, run.run_id, 1, (payload,), "{}", "{}", "{}", False
    )
    second = agent_store.start_model_request(
        AGENT, run.sequence, run.run_id, 1, (payload,), "{}", "{}", "{}", False
    )
    blobs = agent_store._db.execute(
        "SELECT COUNT(*) AS count, SUM(byte_length) AS bytes"
        " FROM agent_history_message_blobs"
    )[0]
    refs = agent_store._db.execute(
        "SELECT COUNT(*) AS count FROM agent_model_request_messages"
    )[0]
    assert first.ordinal == 1
    assert second.ordinal == 2
    assert blobs["count"] == 1
    assert blobs["bytes"] == len(payload.encode())
    assert refs["count"] == 2


def test_history_pagination_and_model_fields_are_bounded(
    agent_store: SqliteAgentStore,
) -> None:
    run = agent_store.start_run(AGENT)
    field = '{"value":"' + "漢" * 12_000 + '"}'
    for _ in range(35):
        agent_store.start_model_request(
            AGENT,
            run.sequence,
            run.run_id,
            1,
            ("[]",),
            field,
            "{}",
            "{}",
            False,
        )
    summaries = agent_store.model_request_summaries(AGENT, run.sequence)
    following = agent_store.model_request_summaries(
        AGENT, run.sequence, after=summaries[-1].ordinal
    )
    assert len(summaries) == 30
    assert [item.ordinal for item in following] == list(range(31, 36))
    chunks = []
    offset = 0
    while True:
        page = agent_store.model_request_field(
            AGENT, run.sequence, 1, "parameters", offset, 16 * 1024
        )
        assert len(page.value.encode("utf-8")) <= 16 * 1024
        chunks.append(page.value)
        offset = page.offset + len(page.value.encode("utf-8"))
        if not page.has_more:
            break
    assert "".join(chunks) == field
    for _ in range(35):
        agent_store.reset_window(AGENT, "overflow")
    windows = agent_store.window_events(AGENT)
    following_windows = agent_store.window_events(AGENT, after=windows[-1].number)
    assert len(windows) == 30
    assert [event.number for event in following_windows] == list(range(31, 37))


def test_reminder_snapshot_survives_upgrade_restart_and_window_reset(tmp_path) -> None:
    path = tmp_path / "fora.sqlite3"
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
    path = tmp_path / "fora.sqlite3"
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
    path = tmp_path / "fora.sqlite3"
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
    path = tmp_path / "fora.sqlite3"
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


def _record_preparation_call(
    store,
    *,
    error="UnexpectedModelBehavior: empty",
    request_status="failed",
    run_status="failed",
    reminded=(),
    agent_id=AGENT,
):
    run = store.start_run(agent_id, reminded=reminded)
    handle = store.start_model_request(
        agent_id,
        run.sequence,
        run.run_id,
        store.window(agent_id).number,
        ('[{"input":"中文🙂"}]',),
        "{}",
        "{}",
        "{}",
        False,
    )
    if request_status in ("responded", "failed"):
        store.finish_model_request(
            agent_id, run.sequence, handle, ('[{"response":"中文🙂"}]',)
        )
    if request_status == "failed":
        store.fail_model_request(agent_id, run.sequence, handle, error)
    store.finish_run(
        agent_id,
        run.sequence,
        status=run_status,
        messages_json='[{"history":"中文🙂"}]',
        error=error if run_status == "failed" else None,
    )
    return run, handle


@pytest.mark.parametrize(
    "boundary",
    [
        "success",
        "ordinary",
        "pending",
        "interrupted",
        "reason",
        "type",
        "status_code",
        "provider_code",
        "no_call",
        "prepared",
        "overflow",
    ],
)
def test_preparation_failure_streak_reads_all_calls_and_window_boundaries(
    agent_store, boundary
):
    for _ in range(2):
        run, _ = _record_preparation_call(agent_store)
    assert agent_store.preparation_failure_streak(AGENT, run.sequence) == 2
    if boundary == "success":
        _record_preparation_call(
            agent_store, request_status="responded", run_status="completed"
        )
    elif boundary == "ordinary":
        _record_preparation_call(agent_store, reminded=[(1, 1)])
    elif boundary in ("pending", "interrupted"):
        _record_preparation_call(
            agent_store, request_status="pending", run_status=boundary
        )
    elif boundary in ("prepared", "overflow"):
        agent_store.reset_window(AGENT, boundary)
    elif boundary == "no_call":
        no_call = agent_store.start_run(AGENT)
        agent_store.finish_run(
            AGENT,
            no_call.sequence,
            status="failed",
            messages_json="[]",
            error="RuntimeError: startup",
        )
        assert agent_store.preparation_failure_streak(AGENT, no_call.sequence) == 0
    else:
        replacement = {
            "reason": "UnexpectedModelBehavior: filtered",
            "type": "ContentFilterError: empty",
            "status_code": "ModelHTTPError: status_code: 429, model_name: gpt-6.1, body: busy",
            "provider_code": "ModelHTTPError: {'code': 'rate_limit', 'message': 'busy'}",
        }[boundary]
        _record_preparation_call(agent_store, error=replacement)
    final, _ = _record_preparation_call(agent_store)
    assert agent_store.preparation_failure_streak(AGENT, final.sequence) == (
        3 if boundary == "no_call" else 1
    )
    assert agent_store.preparation_failure_streak(AGENT + 1, final.sequence) == 0
    assert agent_store.preparation_failure_streak(AGENT, run.sequence) == 0


@pytest.mark.parametrize(
    "error_type",
    ["UnexpectedModelBehavior", "ContentFilterError", "IncompleteToolCall"],
)
def test_legacy_response_failure_binds_only_last_request_and_preserves_records(
    agent_store, error_type
):
    error = f"{error_type}: empty after 4000 tokens"
    for _ in range(2):
        run, _ = _record_preparation_call(
            agent_store, error=error, request_status="responded"
        )
    originals = agent_store.runs(AGENT)
    summaries = [
        agent_store.model_request_summaries(AGENT, run.sequence) for run in originals
    ]
    assert agent_store.preparation_failure_streak(AGENT, run.sequence) == 2
    agent_store.mark_session_start()
    agent_store.reset_safety(AGENT)
    assert agent_store.preparation_failure_streak(AGENT, run.sequence) == 2
    assert agent_store.runs(AGENT) == originals
    assert [
        agent_store.model_request_summaries(AGENT, run.sequence) for run in originals
    ] == summaries
    successful = agent_store.start_model_request(
        AGENT,
        run.sequence,
        run.run_id,
        1,
        (),
        "{}",
        "{}",
        "{}",
        False,
    )
    agent_store.finish_model_request(
        AGENT, run.sequence, successful, ('[{"text":"ok"}]',)
    )
    assert agent_store.preparation_failure_streak(AGENT, run.sequence) == 1
    final, _ = _record_preparation_call(agent_store, error=error)
    assert agent_store.preparation_failure_streak(AGENT, final.sequence) == 2


@pytest.mark.parametrize(
    "error_type",
    [
        "RuntimeError",
        "HistoryPersistenceError",
        "HistoryValidationError",
        "UsageLimitExceeded",
        "RunCancelled",
        "CancelledError",
        "ToolFailed",
        "ToolRetryError",
        "ModelRetry",
        "UserError",
    ],
)
def test_non_model_legacy_errors_are_not_preparation_failures(agent_store, error_type):
    for _ in range(3):
        run, _ = _record_preparation_call(
            agent_store, error=f"{error_type}: failed", request_status="responded"
        )
    assert agent_store.preparation_failure_streak(AGENT, run.sequence) == 0


def test_preparation_failure_streak_is_bounded_metadata_and_repeatable(agent_store):
    for index in range(5):
        run, handle = _record_preparation_call(
            agent_store,
            error=f"ModelHTTPError: status_code: 502, model_name: gpt-6.1, body: request_id=req-{index}, busy after {index + 3}ms",
        )
    original = agent_store.model_request_summaries(AGENT, run.sequence)
    agent_store.fail_model_request(AGENT, run.sequence, handle, original[0].error)
    connection = agent_store._db._connection
    denied = {
        "messages_json",
        "response_json",
        "parameters_json",
        "settings_json",
        "model_json",
        "content_json",
    }

    def authorize(action, table, column, database, source):
        if action == sqlite3.SQLITE_READ and column in denied:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(authorize)
    try:
        for _ in range(3):
            assert agent_store.preparation_failure_streak(AGENT, run.sequence) == 3
    finally:
        connection.set_authorizer(None)
    assert len(agent_store.model_request_summaries(AGENT, run.sequence)) == 1
    assert (
        agent_store.model_request_summaries(AGENT, run.sequence)[0].response_count == 1
    )
    assert len(agent_store.window_events(AGENT)) == 1


def test_sdk_completed_request_retains_response_and_interrupts_legacy_error_reading(
    agent_store,
):
    for _ in range(2):
        _record_preparation_call(agent_store)
    run, handle = _record_preparation_call(agent_store, request_status="responded")
    original = agent_store.model_request_summary(AGENT, run.sequence, handle.ordinal)
    payloads = agent_store.model_request_messages(
        AGENT, run.sequence, handle.ordinal, "response", 0, 10
    )
    for _ in range(2):
        agent_store.complete_model_request(AGENT, run.sequence, handle)
        current = agent_store.model_request_summary(AGENT, run.sequence, handle.ordinal)
        assert current == replace(original, status="completed")
        assert agent_store.preparation_failure_streak(AGENT, run.sequence) == 0
        assert (
            agent_store.model_request_messages(
                AGENT, run.sequence, handle.ordinal, "response", 0, 10
            )
            == payloads
        )
    failed_run, failed = _record_preparation_call(agent_store)
    assert agent_store.preparation_failure_streak(AGENT, failed_run.sequence) == 1
    with pytest.raises(DomainError, match="successful response"):
        agent_store.complete_model_request(AGENT, failed_run.sequence, failed)


@pytest.mark.parametrize(
    "state", ["pending", "interrupted", "wrong_id", "missing_response"]
)
def test_sdk_completion_requires_successful_response_and_actual_identity(
    agent_store, state
):
    run = agent_store.start_run(AGENT)
    handle = agent_store.start_model_request(
        AGENT, run.sequence, run.run_id, 1, (), "{}", "{}", "{}", False
    )
    if state == "interrupted":
        agent_store.finish_run(
            AGENT, run.sequence, status="interrupted", messages_json="[]"
        )
    elif state in ("wrong_id", "missing_response"):
        agent_store.finish_model_request(
            AGENT,
            run.sequence,
            handle,
            ('[{"text":"done"}]',) if state == "wrong_id" else (),
        )
    original = agent_store.model_request_summary(AGENT, run.sequence, handle.ordinal)
    invalid = replace(handle, request_id="wrong-id") if state == "wrong_id" else handle
    with pytest.raises(DomainError, match="successful response"):
        agent_store.complete_model_request(AGENT, run.sequence, invalid)
    assert (
        agent_store.model_request_summary(AGENT, run.sequence, handle.ordinal)
        == original
    )


@pytest.mark.parametrize("failure", ["transaction", "completion"])
def test_sdk_completion_save_failure_preserves_response_status_and_fields(
    agent_store, failure
):
    run, handle = _record_preparation_call(agent_store, request_status="responded")
    original = agent_store.model_request_summary(AGENT, run.sequence, handle.ordinal)
    if failure == "completion":
        agent_store._db.execute(
            "CREATE TEMP TRIGGER reject_completed BEFORE UPDATE ON agent_model_requests "
            "WHEN NEW.status = 'completed' BEGIN SELECT RAISE(ABORT, 'completion failed'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="completion failed"):
            agent_store.complete_model_request(AGENT, run.sequence, handle)
    else:
        with (
            pytest.raises(ValueError, match="transaction failed"),
            agent_store.transaction(),
        ):
            agent_store.complete_model_request(AGENT, run.sequence, handle)
            raise ValueError("transaction failed")
    assert (
        agent_store.model_request_summary(AGENT, run.sequence, handle.ordinal)
        == original
    )


@pytest.mark.parametrize("status", ["running", "failed", "interrupted", "completed"])
@pytest.mark.parametrize("ordinary", [False, True])
def test_preparation_stage_uses_current_window_and_latest_run_metadata(
    agent_store, status, ordinary
):
    assert not agent_store.preparation_incomplete(AGENT)
    run = agent_store.start_run(AGENT, reminded=[(1, 1)] if ordinary else [])
    if status != "running":
        agent_store.finish_run(
            AGENT,
            run.sequence,
            status=status,
            messages_json="[]",
            usage_json='{"last_input_tokens":null}',
        )
    original = agent_store.runs(AGENT)
    expected = not ordinary and status != "completed"
    assert agent_store.preparation_incomplete(AGENT) is expected
    assert agent_store.preparation_incomplete(999) is False
    assert agent_store.preparation_incomplete(AGENT) is expected
    assert agent_store.runs(AGENT) == original
    agent_store.reset_window(AGENT, "overflow")
    assert not agent_store.preparation_incomplete(AGENT)
    assert agent_store.runs(AGENT) == original
    next_run = agent_store.start_run(AGENT, reminded=[(1, 2)])
    agent_store.finish_run(
        AGENT, next_run.sequence, status="failed", messages_json="[]"
    )
    assert not agent_store.preparation_incomplete(AGENT)
