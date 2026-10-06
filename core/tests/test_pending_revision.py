from __future__ import annotations

import sqlite3
import threading
from contextlib import closing

import pytest
from test_runtime import HELPER, HUMAN, MAIN, RecordingRunner, mention

pytest_plugins = ("test_runtime",)

from fora.adapters.sqlite.agent import SqliteAgentStore
from fora.adapters.sqlite.store import (
    PENDING_REVISION_SCHEMA,
    SCHEMA,
    LockedConnection,
    SqliteStore,
)
from fora.runtime.reminder import TurnOutcome
from fora.runtime.scheduler import Scheduler


def completed(request, tools):
    return TurnOutcome("[]", usage_json='{"tool_calls":1}')


def test_effective_ack_revoke_and_batch_ack(world):
    room = mention(world)
    assert world.history.pending_revision(MAIN) == 1
    world.store.ack(room, [1], MAIN)
    assert world.history.pending_revision(MAIN) == 2
    world.store.ack(room, [1], MAIN)
    assert world.history.pending_revision(MAIN) == 2
    world.store.revoke_ack(room, [1], MAIN)
    assert world.history.pending_revision(MAIN) == 3
    world.store.revoke_ack(room, [1], MAIN)
    world.store.append_message(room, HUMAN, "plain message")
    world.store.ack(room, [2], MAIN)
    world.store.revoke_ack(room, [2], MAIN)
    assert world.history.pending_revision(MAIN) == 3
    world.store.append_message(room, HUMAN, "@Main second")
    assert world.history.pending_revision(MAIN) == 4
    world.store.ack_pending(room, MAIN, 3)
    assert world.history.pending_revision(MAIN) == 6
    world.store.ack_pending(room, MAIN, 3)
    assert world.history.pending_revision(MAIN) == 6


def test_archive_and_inactive_ack(world):
    room = mention(world)
    world.store.set_archived(room, True)
    assert world.history.pending_revision(MAIN) == 2
    world.store.set_archived(room, True)
    world.store.ack(room, [1], MAIN)
    world.store.revoke_ack(room, [1], MAIN)
    assert world.history.pending_revision(MAIN) == 2
    world.store.set_archived(room, False)
    assert world.history.pending_revision(MAIN) == 3
    world.store.set_archived(room, False)
    assert world.history.pending_revision(MAIN) == 3


def test_member_replacement_preserves_retained_members(world):
    room = mention(world)
    for member_ids in ([MAIN, HUMAN], [HUMAN, MAIN, HELPER], [HUMAN, MAIN]):
        world.store.set_discussion_members(room, member_ids)
        assert world.history.pending_revision(MAIN) == 1
    world.store.set_discussion_members(room, [HUMAN])
    assert world.history.pending_revision(MAIN) == 2
    world.store.set_discussion_members(room, [HUMAN, MAIN])
    assert world.history.pending_revision(MAIN) == 3


def test_revision_rolls_back_with_pending(world):
    room = mention(world)
    pending = world.store.pending(MAIN)
    with pytest.raises(sqlite3.IntegrityError), world.history.transaction():
        world.store.ack(room, [1], MAIN)
        assert world.history.pending_revision(MAIN) == 2
        world.store._db.execute("INSERT INTO pending_revisions VALUES (?, 9)", (MAIN,))
    assert world.history.pending_revision(MAIN) == 1
    assert world.store.pending(MAIN) == pending


def test_distinct_discussion_identity_and_deleted_mention(world):
    first = mention(world)
    second = mention(world)
    assert world.history.pending_revision(MAIN) == 2
    world.store.ack(first, [1], MAIN)
    assert world.history.pending_revision(MAIN) == 3
    with world.store._db:
        world.store._db.execute(
            "DELETE FROM mentions WHERE discussion_id = ?", (second,)
        )
        assert world.history.pending_revision(MAIN) == 4
    assert world.history.pending_revision(MAIN) == 4
    assert world.store.pending(MAIN) == ()


@pytest.mark.parametrize("stage", range(10))
def test_migration_failure_rolls_back_and_reopens(tmp_path, monkeypatch, stage):
    path = tmp_path / "legacy.sqlite3"
    with closing(sqlite3.connect(path)) as legacy:
        legacy.executescript(SCHEMA)
        legacy.execute(
            "INSERT INTO members (id, type, name, name_key, state) VALUES (2, 'agent', 'Main', 'main', 'paused')"
        )
        legacy.execute(
            "INSERT INTO agent_runs (agent_id, sequence, run_id, status, started_at, messages_json) VALUES (2, 1, 'legacy', 'completed', '2026-01-01', '[]')"
        )
        legacy.commit()
    execute = LockedConnection.execute
    seen = []

    def fail_at_stage(connection, sql, parameters=()):
        if sql in PENDING_REVISION_SCHEMA or sql.startswith(
            (
                "ALTER TABLE agent_runs ADD COLUMN pending_revision",
                "CREATE INDEX agent_runs_summary",
            )
        ):
            seen.append(sql)
            if len(seen) == stage + 1:
                raise sqlite3.OperationalError("migration rejected")
        return execute(connection, sql, parameters)

    with monkeypatch.context() as patch:
        patch.setattr(LockedConnection, "execute", fail_at_stage)
        with pytest.raises(sqlite3.OperationalError, match="migration rejected"):
            SqliteStore(path)
    with closing(sqlite3.connect(path)) as check:
        assert "pending_revision" not in {
            row[1] for row in check.execute("PRAGMA table_info(agent_runs)")
        }
        assert (
            check.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'pending_revision%'"
                " OR name = 'agent_runs_summary'"
            ).fetchall()
            == []
        )
    with closing(SqliteStore(path)) as store:
        history = SqliteAgentStore(store._db)
        assert history.pending_revision(MAIN) == 0
        assert history.runs(MAIN)[0].pending_revision == -1
        assert history.runs(MAIN)[0].messages_json == "[]"
        assert store.get_member(MAIN).state == "paused"
        assert history.repeated_turns(MAIN, [(1, 1)]) == 0
        assert (
            len(
                store._db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'pending_revision_%'"
                )
            )
            == 7
        )


def test_three_completed_turns_block_until_human_resume(world):
    room = mention(world)
    scheduler = Scheduler(world, RecordingRunner(completed))
    for count in range(1, 4):
        assert scheduler.run_turn(MAIN).status == "completed"
        assert world.history.repeated_turns(MAIN, [(room, 1)]) == count
        assert scheduler.agent_status(MAIN)["state"] == (
            "blocked" if count == 3 else "idle"
        )
    assert world.history.pause_reason(MAIN) == "repeated_mentions"
    assert scheduler.run_turn(MAIN) is None
    world.store.append_message(room, HUMAN, "@Main new work")
    assert scheduler.run_turn(MAIN) is None
    world.store.ack_pending(room, MAIN, 2)
    assert scheduler.agent_status(MAIN)["state"] == "blocked"
    with pytest.raises(Exception, match="Human"):
        scheduler.resume(MAIN, is_agent=True)
    assert scheduler.resume(MAIN)["state"] == "idle"
    assert world.history.pause_reason(MAIN) is None
    world.store.revoke_ack(room, [1], MAIN)
    assert scheduler.run_turn(MAIN).status == "completed"
    assert world.history.repeated_turns(MAIN, [(room, 1)]) == 1


def test_repeated_mentions_has_priority_over_no_tool_calls(world):
    mention(world)
    scheduler = Scheduler(
        world,
        RecordingRunner(
            lambda request, tools: TurnOutcome("[]", usage_json='{"tool_calls":0}')
        ),
    )
    for _ in range(3):
        scheduler.run_turn(MAIN)
    assert world.history.pause_reason(MAIN) == "repeated_mentions"


def test_second_turn_count_survives_reopen(world):
    room = mention(world)
    scheduler = Scheduler(world, RecordingRunner(completed))
    scheduler.run_turn(MAIN)
    scheduler.run_turn(MAIN)
    path = world.store._path
    world.store.close()
    with closing(SqliteStore(path)) as reopened:
        world.store = reopened
        world.history = world.settings = SqliteAgentStore(reopened._db)
        restarted = Scheduler(world, RecordingRunner(completed))
        restarted.recover()
        assert world.history.repeated_turns(MAIN, [(room, 1)]) == 2
        assert restarted.run_turn(MAIN).status == "completed"
        assert restarted.agent_status(MAIN)["state"] == "blocked"


@pytest.mark.parametrize("change", ["reopen_ack", "archive", "membership"])
def test_reopened_pending_starts_new_cycle(world, change):
    room = mention(world)
    scheduler = Scheduler(world, RecordingRunner(completed))
    scheduler.run_turn(MAIN)
    scheduler.run_turn(MAIN)
    if change == "reopen_ack":
        world.store.ack(room, [1], MAIN)
        world.store.revoke_ack(room, [1], MAIN)
    elif change == "archive":
        world.store.set_archived(room, True)
        world.store.set_archived(room, False)
    else:
        world.store.set_discussion_members(room, [HUMAN])
        world.store.set_discussion_members(room, [HUMAN, MAIN])
    assert world.history.repeated_turns(MAIN, [(room, 1)]) == 0
    assert scheduler.run_turn(MAIN).status == "completed"
    assert scheduler.agent_status(MAIN)["state"] == "idle"
    assert world.history.repeated_turns(MAIN, [(room, 1)]) == 1


@pytest.mark.parametrize("change", ["ack", "mention"])
def test_pending_change_before_third_finish_starts_new_cycle(world, change):
    room = mention(world)
    entered = threading.Event()
    release = threading.Event()

    def respond(request, tools):
        if request.sequence == 3:
            entered.set()
            assert release.wait(10)
        return completed(request, tools)

    scheduler = Scheduler(world, RecordingRunner(respond))
    scheduler.run_turn(MAIN)
    scheduler.run_turn(MAIN)
    thread = threading.Thread(target=lambda: scheduler.run_turn(MAIN))
    try:
        thread.start()
        assert entered.wait(5)
        if change == "ack":
            world.store.ack(room, [1], MAIN)
        else:
            world.store.append_message(room, HUMAN, "@Main another")
        release.set()
        thread.join(5)
        assert not thread.is_alive()
        assert world.history.pause_reason(MAIN) is None
        assert scheduler.agent_status(MAIN)["state"] == "idle"
        assert world.history.repeated_turns(MAIN, [(room, 1)]) == 0
    finally:
        release.set()
        thread.join(5)
        scheduler.stop()


@pytest.mark.parametrize("resume", [False, True])
def test_pause_and_running_resume_keep_third_turn_count(world, resume):
    room = mention(world)
    entered = threading.Event()
    release = threading.Event()

    def respond(request, tools):
        if request.sequence == 3:
            entered.set()
            assert release.wait(10)
        return completed(request, tools)

    scheduler = Scheduler(world, RecordingRunner(respond))
    scheduler.run_turn(MAIN)
    scheduler.run_turn(MAIN)
    thread = threading.Thread(target=lambda: scheduler.run_turn(MAIN))
    try:
        thread.start()
        assert entered.wait(5)
        assert scheduler.pause(MAIN)["state"] == "running"
        if resume:
            assert scheduler.resume(MAIN)["state"] == "running"
        assert world.history.repeated_turns(MAIN, [(room, 1)]) == 2
        release.set()
        thread.join(5)
        assert not thread.is_alive()
        assert scheduler.agent_status(MAIN)["state"] == "blocked"
        assert world.history.lifecycle(MAIN).pause_requested is (not resume)
        assert world.history.repeated_turns(MAIN, [(room, 1)]) == 3
        assert MAIN not in scheduler._reserved
        scheduler.resume(MAIN)
        assert world.history.repeated_turns(MAIN, [(room, 1)]) == 0
    finally:
        release.set()
        thread.join(5)
        scheduler.stop()


def test_third_commit_precedes_pending_change_and_keeps_block(world, monkeypatch):
    room = mention(world)
    entered = threading.Event()
    release = threading.Event()
    changing = threading.Event()
    changed = threading.Event()
    save_reason = world.history.pause_for_safety

    def hold_reason(agent_id, reason):
        if reason == "repeated_mentions":
            entered.set()
            assert release.wait(10)
        save_reason(agent_id, reason)

    def change_pending():
        changing.set()
        world.store.append_message(room, HUMAN, "@Main new work")
        changed.set()

    scheduler = Scheduler(world, RecordingRunner(completed))
    scheduler.run_turn(MAIN)
    scheduler.run_turn(MAIN)
    monkeypatch.setattr(world.history, "pause_for_safety", hold_reason)
    turn = threading.Thread(target=lambda: scheduler.run_turn(MAIN))
    mutation = threading.Thread(target=change_pending)
    try:
        turn.start()
        assert entered.wait(5)
        mutation.start()
        assert changing.wait(5)
        assert not changed.is_set()
        release.set()
        turn.join(5)
        mutation.join(5)
        assert not turn.is_alive() and not mutation.is_alive()
        assert changed.is_set()
        assert scheduler.agent_status(MAIN)["state"] == "blocked"
        assert world.history.pause_reason(MAIN) == "repeated_mentions"
        assert len(world.store.pending(MAIN)) == 2
    finally:
        release.set()
        turn.join(5)
        if mutation.ident is not None:
            mutation.join(5)
        scheduler.stop()


def test_human_resume_failure_preserves_repeat_reason_and_boundary(world):
    room = mention(world)
    scheduler = Scheduler(world, RecordingRunner(completed))
    for _ in range(3):
        scheduler.run_turn(MAIN)
    world.store._db.execute(
        "CREATE TEMP TRIGGER reject_resume BEFORE UPDATE OF state ON members WHEN NEW.state = 'idle' BEGIN SELECT RAISE(ABORT, 'resume rejected'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="resume rejected"):
        scheduler.resume(MAIN)
    assert world.history.pause_reason(MAIN) == "repeated_mentions"
    assert world.history.repeated_turns(MAIN, [(room, 1)]) == 3
    assert scheduler.agent_status(MAIN)["state"] == "blocked"
    world.store._db.execute("DROP TRIGGER reject_resume")
    scheduler.resume(MAIN)
    assert world.history.repeated_turns(MAIN, [(room, 1)]) == 0


def test_preparation_preserves_completed_reminder_count(world):
    room = mention(world)
    world.settings.set_settings("agent", {"context_window_tokens": 100})

    def respond(request, tools):
        if request.sequence == 1:
            return TurnOutcome(
                "[]", usage_json='{"tool_calls":1,"last_input_tokens":100}'
            )
        return completed(request, tools)

    runner = RecordingRunner(respond)
    scheduler = Scheduler(world, runner)
    scheduler.run_turn(MAIN)
    assert world.history.repeated_turns(MAIN, [(room, 1)]) == 1
    scheduler.run_turn(MAIN)
    assert runner.requests[-1].reminder is None
    assert world.history.repeated_turns(MAIN, [(room, 1)]) == 1
    scheduler.run_turn(MAIN)
    assert world.history.repeated_turns(MAIN, [(room, 1)]) == 2
    scheduler.run_turn(MAIN)
    assert scheduler.agent_status(MAIN)["state"] == "blocked"
    assert world.history.repeated_turns(MAIN, [(room, 1)]) == 3


def test_failed_reminder_does_not_increment_completed_count(world):
    room = mention(world)

    def respond(request, tools):
        if request.sequence == 2:
            return TurnOutcome("[]", error="request failed")
        return completed(request, tools)

    scheduler = Scheduler(world, RecordingRunner(respond))
    scheduler.run_turn(MAIN)
    assert scheduler.run_turn(MAIN).status == "failed"
    assert world.history.repeated_turns(MAIN, [(room, 1)]) == 1
    assert scheduler.run_turn(MAIN) is None
    world.store.append_message(room, HUMAN, "@Main new request")
    assert scheduler.run_turn(MAIN).status == "completed"
    assert world.history.repeated_turns(MAIN, [(room, 1), (room, 2)]) == 1


@pytest.mark.parametrize("stage", ["reason", "lifecycle", "state"])
def test_third_finish_failure_keeps_counter_and_reports_error(world, stage):
    room = mention(world)
    scheduler = Scheduler(world, RecordingRunner(completed))
    scheduler.run_turn(MAIN)
    scheduler.run_turn(MAIN)
    statements = {
        "reason": "CREATE TEMP TRIGGER reject_repeat BEFORE INSERT ON agent_safety WHEN NEW.pause_reason = 'repeated_mentions' BEGIN SELECT RAISE(ABORT, 'repeat rejected'); END",
        "lifecycle": "CREATE TEMP TRIGGER reject_repeat BEFORE UPDATE ON agent_lifecycle WHEN NEW.error IS NULL AND EXISTS (SELECT 1 FROM agent_runs WHERE agent_id = NEW.agent_id AND sequence = 3 AND status = 'completed') BEGIN SELECT RAISE(ABORT, 'repeat rejected'); END",
        "state": "CREATE TEMP TRIGGER reject_repeat BEFORE UPDATE OF state ON members WHEN NEW.state = 'blocked' BEGIN SELECT RAISE(ABORT, 'repeat rejected'); END",
    }
    world.store._db.execute(statements[stage])
    result = scheduler.run_turn(MAIN)
    assert result.status == "failed"
    assert world.history.repeated_turns(MAIN, [(room, 1)]) == 2
    assert world.history.pause_reason(MAIN) is None
    assert scheduler.agent_status(MAIN)["state"] == "error"
    assert scheduler.run_turn(MAIN) is None
