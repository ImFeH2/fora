from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from huddol.adapters.execution.manager import ExecutionManager
from huddol.adapters.files.tree import DirectoryTree
from huddol.adapters.files.uploads import decode_image
from huddol.adapters.sqlite.agent import SqliteAgentStore
from huddol.adapters.sqlite.store import SqliteStore
from huddol.core.errors import DomainError
from huddol.ports.agent import WindowState
from huddol.runtime.reminder import (
    PREPARATION_PROMPT,
    TurnOutcome,
    TurnRequest,
    build_reminder,
    render_resident,
    reset_notice,
)
from huddol.runtime.scheduler import Scheduler
from huddol.tools import AgentTools, Dependencies

HUMAN = 1
MAIN = 2
HELPER = 3


class RecordingRunner:
    def __init__(self, behaviour=None) -> None:
        self.requests: list[TurnRequest] = []
        self._behaviour = behaviour

    def check_available(self, agent_id: int) -> None:
        pass

    def validate_history(self, raw: str) -> None:
        json.loads(raw)

    def run(self, request: TurnRequest, tools: AgentTools) -> TurnOutcome:
        self.requests.append(request)
        if self._behaviour is not None:
            return self._behaviour(request, tools)
        return TurnOutcome(messages_json=json.dumps([{"kind": "response"}]))


@pytest.fixture
def world(tmp_path: Path):
    store = SqliteStore(tmp_path / "huddol.sqlite3")
    agent_store = SqliteAgentStore(store._db)
    store.create_member("human", "You")
    store.create_member("agent", "Main")
    store.create_member("agent", "Helper")

    def agent_directory_for(member_id: int) -> Path:
        path = tmp_path / "agents" / str(member_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    deps = Dependencies(
        store=store,
        history=agent_store,
        settings=agent_store,
        agent_directory_for=agent_directory_for,
        decode_image=decode_image,
        execution=ExecutionManager(
            settings={"write_directories": [str(tmp_path)]},
            enforce=False,
        ),
        library_tree=DirectoryTree(tmp_path / "library"),
        workspace_tree_for=lambda member_id: DirectoryTree(
            tmp_path / "agents" / str(member_id) / "workspace"
        ),
    )
    yield deps
    store.close()


def mention(deps: Dependencies, body: str = "@Main please help") -> int:
    room = deps.store.create_discussion("work", [HUMAN, MAIN])
    deps.store.append_message(room.id, HUMAN, body)
    return room.id


def test_unavailable_execution_blocks_new_turns(world) -> None:
    room = mention(world)
    world.execution.close()
    world.execution = ExecutionManager(settings={"legacy": "invalid"})
    runner = RecordingRunner()
    scheduler = Scheduler(world, runner)
    try:
        assert scheduler.run_turn(MAIN) is None
        state = scheduler.agent_status(MAIN)
        assert state["state"] == "blocked"
        assert any(
            reason["code"] == "execution_unavailable" for reason in state["reasons"]
        )
        assert runner.requests == []
        assert world.history.new_mentions(MAIN, [(room, 1)]) == {(room, 1)}
    finally:
        scheduler.stop()


def test_reminder_never_contains_the_message_body(world) -> None:
    mention(world, "@Main the secret passphrase is hunter2")
    reminder = build_reminder(world.store, world.history, MAIN, "Main")
    assert reminder is not None
    rendered = reminder.render()
    assert "hunter2" not in rendered
    assert "secret passphrase" not in rendered
    assert "Discussion 1" in rendered
    assert "Message 1" in rendered
    assert "discussion action=read" in rendered


def test_reminder_names_the_topic_and_sender(world) -> None:
    mention(world)
    reminder = build_reminder(world.store, world.history, MAIN, "Main")
    assert reminder is not None
    assert reminder.items[0].topic == "work"
    assert reminder.items[0].sender_name == "You"


def test_model_discussion_list_defaults_to_twenty(world, monkeypatch) -> None:
    from pydantic_ai import models
    from pydantic_ai.messages import (
        ModelResponse,
        RetryPromptPart,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import FunctionModel

    from huddol.adapters.model.runner import PydanticModelRunner

    for day in range(1, 26):
        monkeypatch.setattr(
            "huddol.adapters.sqlite.store.now",
            lambda day=day: f"2026-01-{day:02}T00:00:00Z",
        )
        mention(world)
    actions = iter([{}, {"limit": 3}, {"limit": 30}, {"limit": 0}])
    returned = []
    retries = []

    def respond(messages, info):
        definition = next(
            tool for tool in info.function_tools if tool.name == "discussion"
        )
        assert "limit" in definition.parameters_json_schema["properties"]
        assert "defaults to 20" in definition.description
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart):
                returned.append(part.content)
            elif isinstance(part, RetryPromptPart):
                retries.append(part.content)
        action = next(actions, None)
        return (
            ModelResponse(parts=[TextPart("Done")])
            if action is None
            else ModelResponse(
                parts=[ToolCallPart("discussion", {"action": "list", **action})]
            )
        )

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    world.settings.set_settings(
        "model",
        {"base_url": "https://example.invalid", "api_key": "unused", "model": "local"},
    )
    runner = PydanticModelRunner(
        world.settings, build_model=lambda config: FunctionModel(respond)
    )
    record = Scheduler(world, runner).run_turn(MAIN)
    assert record is not None and record.status == "completed", record
    assert [[item["id"] for item in page] for page in returned] == [
        list(range(25, 5, -1)),
        [25, 24, 23],
        list(range(25, 0, -1)),
    ]
    assert len(retries) == 1 and "invalid_pagination" in str(retries[0])


def test_model_can_manage_discussions_and_page_messages(world, monkeypatch) -> None:
    from pydantic_ai import models
    from pydantic_ai.messages import (
        ModelResponse,
        RetryPromptPart,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import FunctionModel

    from huddol.adapters.model.runner import PydanticModelRunner

    room = mention(world, "@Main review this")
    world.store.append_message(room, HUMAN, "needle")
    world.store.append_message(room, HUMAN, "tail")
    actions = iter(
        [
            {"action": "add_members", "member_ids": [HELPER]},
            {"action": "read", "before": 3, "limit": 1},
            {"action": "read", "after": 0, "limit": 1},
            {"action": "read", "message_id": 2, "before": 3},
            {"action": "search", "query": "needle", "sender_id": HUMAN},
            {"action": "remove_members", "member_ids": [HELPER]},
            {"action": "archive"},
            {"action": "read", "message_id": 2, "limit": 1},
            {"action": "unarchive"},
        ]
    )
    returned = []
    retries = []

    def respond(messages, info):
        definition = next(
            tool for tool in info.function_tools if tool.name == "discussion"
        )
        properties = definition.parameters_json_schema["properties"]
        assert set(properties["action"]["enum"]) == {
            "create",
            "list",
            "read",
            "send",
            "ack",
            "revoke_ack",
            "search",
            "add_members",
            "remove_members",
            "archive",
            "unarchive",
        }
        assert {"before", "after", "limit", "sender_id"} <= properties.keys()
        assert (
            "Archive is reversible and is the only way to put a Discussion away."
            in definition.description
        )
        assert (
            "Each message lists its mentions; an @Name that is not listed there notified nobody."
            in definition.description
        )
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart):
                returned.append(part.content)
            elif isinstance(part, RetryPromptPart):
                retries.append(part.content)
        action = next(actions, None)
        if action is None:
            return ModelResponse(parts=[TextPart("Done")])
        return ModelResponse(
            parts=[ToolCallPart("discussion", {"discussion_id": room, **action})]
        )

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    world.settings.set_settings(
        "model",
        {"base_url": "https://example.invalid", "api_key": "unused", "model": "local"},
    )
    runner = PydanticModelRunner(
        world.settings, build_model=lambda config: FunctionModel(respond)
    )
    emitted = []
    record = Scheduler(
        world, runner, on_event=lambda name, payload: emitted.append(name)
    ).run_turn(MAIN)
    assert record is not None
    assert record.status == "completed", record.error
    assert [name for name in emitted if name.startswith("discussion.")] == [
        "discussion.updated",
        "discussion.updated",
        "discussion.updated",
        "discussion.updated",
    ]
    assert emitted[-1] == "turn.finished"
    assert len(retries) == 1 and "invalid_pagination" in str(retries[0])
    assert returned[0]["member_ids"] == [HUMAN, MAIN, HELPER]
    assert [item["id"] for item in returned[1]["messages"]] == [2]
    assert [item["id"] for item in returned[2]["messages"]] == [1]
    assert [item["id"] for item in returned[3]] == [2]
    assert returned[4]["member_ids"] == [HUMAN, MAIN]
    assert returned[5]["archived"] is True
    assert [item["id"] for item in returned[6]["messages"]] == [1, 2, 3]
    assert returned[7]["archived"] is False
    assert len(returned) == 8
    assert world.store.get_discussion(room) is not None
    assert world.history.latest_messages(MAIN) != "[]"


@pytest.mark.parametrize("actor_id", [HUMAN, MAIN])
def test_discussion_notifications_do_not_depend_on_the_caller_type(
    world, actor_id
) -> None:
    emitted = []
    scheduler = Scheduler(
        world, RecordingRunner(), on_event=lambda name, payload: emitted.append(name)
    )
    actor = scheduler.tools_for(actor_id)
    room = actor.create_discussion("Notifications", [HELPER])["id"]
    name = world.store.get_member(actor_id).name
    scheduler.tools_for(HELPER).send_message(room, f"@{name} please review")
    actor.read_discussion(room)
    actor.ack(room, [1])
    actor.revoke_ack(room, [1])
    assert emitted == [
        "discussion.created",
        "message.created",
        "mention.acked",
        "mention.revoked",
    ]


def test_lowering_the_window_budget_schedules_preparation(world, monkeypatch) -> None:
    from pydantic_ai import ModelMessagesTypeAdapter, models
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.usage import RequestUsage

    from huddol.adapters.model.runner import PydanticModelRunner

    mention(world)
    world.settings.set_settings(
        "model",
        {"base_url": "https://example.invalid", "api_key": "unused", "model": "local"},
    )
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    runner = PydanticModelRunner(
        world.settings,
        build_model=lambda config: FunctionModel(
            lambda messages, info: ModelResponse(
                parts=[TextPart("Done")], usage=RequestUsage(input_tokens=100)
            )
        ),
    )
    scheduler = Scheduler(world, runner)
    assert scheduler.run_turn(MAIN).status == "completed"
    assert not scheduler.preparation_due(MAIN)
    world.settings.set_settings("agent", {"context_window_tokens": 100})
    assert scheduler.preparation_due(MAIN)
    assert scheduler.run_turn(MAIN).status == "completed"
    runs = world.history.runs(MAIN)
    saved = ModelMessagesTypeAdapter.validate_json(runs[0].messages_json)
    assert saved[-2].parts[0].content == PREPARATION_PROMPT
    assert world.history.latest_messages(MAIN) == "[]"
    assert scheduler.runnable_agents() == (MAIN,)
    mention(world)
    assert scheduler.run_turn(MAIN).status == "completed"
    saved = ModelMessagesTypeAdapter.validate_json(world.history.latest_messages(MAIN))
    assert len(saved) == 3
    assert saved[0].parts[0].content == scheduler.resident_block(MAIN)
    assert saved[0].metadata["huddol"]["block"] == "resident"
    assert saved[0].parts[0].content.endswith(reset_notice(world.history.window(MAIN)))


@pytest.mark.parametrize("first_status", ["completed", "failed"])
@pytest.mark.parametrize(
    "preparation_status", ["completed", "failed", "exception", "overflow"]
)
def test_preparation_runs_without_pending_and_resets_after_completion(
    world, first_status, preparation_status
) -> None:
    room = mention(world)
    world.settings.set_settings("agent", {"context_window_tokens": 100})
    events = []

    def respond(request, tools):
        if request.reminder is not None:
            tools.read_discussion(room)
            tools.ack(room, [1])
            return TurnOutcome(
                messages_json='[{"text":"saved"}]',
                usage_json='{"input_tokens":100,"last_input_tokens":100}',
                input_tokens=100,
                error="failure" if first_status == "failed" else None,
            )
        assert request.prompt == PREPARATION_PROMPT
        assert request.ephemeral() == ""
        if preparation_status == "exception":
            raise RuntimeError("preparation failed")
        return TurnOutcome(
            messages_json='[{"text":"notes saved"}]',
            error="preparation failed" if preparation_status != "completed" else None,
            context_exceeded=preparation_status == "overflow",
        )

    runner = RecordingRunner(respond)
    scheduler = Scheduler(
        world, runner, on_event=lambda name, payload: events.append((name, payload))
    )
    assert scheduler.run_turn(MAIN).status == first_status
    assert world.store.pending(MAIN) == ()
    assert scheduler.preparation_due(MAIN)
    if first_status == "failed":
        assert scheduler.runnable_agents() == ()
        assert Scheduler(world, runner).runnable_agents() == ()
        assert scheduler.run_turn(MAIN) is None
        return
    assert scheduler.runnable_agents() == (MAIN,)
    restarted = Scheduler(world, runner)
    assert restarted.preparation_due(MAIN)
    assert restarted.runnable_agents() == (MAIN,)
    record = scheduler.run_turn(MAIN)
    assert record.status == (
        "completed" if preparation_status == "completed" else "failed"
    )
    assert runner.requests[-1].reminder is None
    assert runner.requests[-1].history_json == '[{"text":"saved"}]'
    state = world.history.window(MAIN)
    if preparation_status != "completed":
        assert state.number == 1 and state.since_sequence == 1
        assert not any(name == "window.reset" for name, _ in events)
        expected = "saved" if preparation_status == "exception" else "notes saved"
        assert json.loads(world.history.latest_messages(MAIN)) == [{"text": expected}]
        assert world.store.get_member(MAIN).state == "error"
        assert scheduler.runnable_agents() == ()
        return
    assert state.number == 2 and state.since_sequence == 3
    assert state.reason == "prepared" and state.reset_at is not None
    assert world.history.latest_messages(MAIN) == "[]"
    assert not scheduler.preparation_due(MAIN)
    assert not restarted.preparation_due(MAIN)
    assert scheduler.runnable_agents() == ()
    assert events == [
        (
            "turn.started",
            {"agent_id": MAIN, "sequence": 1, "items": 1, "kind": "reminder"},
        ),
        ("mention.acked", {"discussion_id": room, "acked": 1}),
        ("turn.finished", {"agent_id": MAIN, "sequence": 1, "status": first_status}),
        (
            "turn.started",
            {"agent_id": MAIN, "sequence": 2, "items": 0, "kind": "preparation"},
        ),
        *(
            [("organization.changed", {"id": MAIN, "state": "paused"})]
            if preparation_status == "exception"
            else []
        ),
        ("window.reset", {"agent_id": MAIN, "number": 2, "reason": "prepared"}),
        ("turn.finished", {"agent_id": MAIN, "sequence": 2, "status": record.status}),
    ]
    world.store.append_message(room, HUMAN, "@Main next task")
    if preparation_status == "exception":
        assert scheduler.run_turn(MAIN) is None
        scheduler.tools_for(HUMAN).resume_agent(MAIN)
    assert scheduler.run_turn(MAIN).status == first_status
    following = runner.requests[-1]
    assert following.agent_id == MAIN and following.agent_name == "Main"
    assert following.reminder is not None
    assert following.prompt == following.reminder.render()
    assert following.history_json == "[]"
    assert following.resident.endswith(reset_notice(state))


@pytest.mark.parametrize("fresh", [False, True])
def test_overflow_resets_and_automatically_continues_pending_mentions(
    world, fresh
) -> None:
    mention(world)
    if fresh:
        world.history.reset_window(MAIN, "prepared")
    events = []
    runner = RecordingRunner()
    scheduler = Scheduler(
        world, runner, on_event=lambda name, payload: events.append((name, payload))
    )
    assert scheduler.run_turn(MAIN).status == "completed"
    prior = world.history.latest_messages(MAIN)
    overflow_calls = 0

    def overflow_once(request, tools):
        nonlocal overflow_calls
        overflow_calls += 1
        if overflow_calls == 1:
            return TurnOutcome(
                messages_json=prior, error="too long", context_exceeded=True
            )
        assert request.reminder is not None
        for item in request.reminder.items:
            world.store.ack(item.discussion_id, [item.message_id], MAIN)
        return TurnOutcome(messages_json='[{"kind":"response"}]')

    runner._behaviour = overflow_once
    previous = world.history.window(MAIN)
    mention(world)
    scheduler._wake.clear()
    failed = scheduler.run_turn(MAIN)
    assert failed.status == "failed"
    state = world.history.window(MAIN)
    assert state.number == previous.number + 1
    assert state.reason == "overflow" and state.since_sequence == 3
    assert world.history.latest_messages(MAIN) == "[]"
    failed_run = world.history.runs(MAIN)[0]
    assert failed_run.status == "failed"
    assert failed_run.error == "too long"
    assert world.history.lifecycle(MAIN).error is None
    assert scheduler.agent_status(MAIN)["state"] == "idle"
    assert scheduler._wake.is_set()
    assert (
        "window.reset",
        {"agent_id": MAIN, "number": state.number, "reason": "overflow"},
    ) in events
    try:
        pending_before = scheduler.pending_keys(MAIN)
        assert pending_before
        assert scheduler.tick() == (MAIN,)
        scheduler._threads[MAIN].join(timeout=5)
        assert not scheduler._threads[MAIN].is_alive()
        following = runner.requests[-1]
        assert following.sequence == failed_run.sequence + 1
        assert following.history_json == "[]"
        assert following.reminder is not None
        assert following.reminder.items
        assert following.resident.endswith(reset_notice(state))
        assert world.history.window(MAIN) == state
        assert world.history.last_reminder(MAIN) == pending_before
        assert [run.status for run in world.history.runs(MAIN)[:2]] == [
            "completed",
            "failed",
        ]
        assert scheduler.tick() == ()
    finally:
        scheduler.stop()


def test_overflow_does_not_continue_when_no_mentions_remain(world) -> None:
    room = mention(world)
    previous = world.history.start_run(MAIN)
    world.history.finish_run(
        MAIN, previous.sequence, status="completed", messages_json="[]"
    )
    runner = RecordingRunner(
        lambda request, tools: TurnOutcome(
            messages_json="[]", error="too long", context_exceeded=True
        )
    )
    scheduler = Scheduler(world, runner)
    world.store.append_message(room, HUMAN, "@Main continue")
    assert scheduler.run_turn(MAIN).status == "failed"
    assert world.history.window(MAIN).reason == "overflow"
    for item in world.store.pending(MAIN):
        world.store.ack(item.discussion_id, [item.message_id], MAIN)
    assert scheduler.pending_keys(MAIN) == frozenset()
    assert scheduler.tick() == ()
    assert len(runner.requests) == 1
    assert [run.status for run in world.history.runs(MAIN)] == ["failed", "completed"]


def test_repeated_overflow_in_new_window_keeps_error_without_another_reset(
    world,
) -> None:
    previous = world.history.start_run(MAIN)
    world.history.finish_run(
        MAIN, previous.sequence, status="completed", messages_json="[]"
    )
    mention(world)
    runner = RecordingRunner(
        lambda request, tools: TurnOutcome(
            messages_json="[]", error="too long", context_exceeded=True
        )
    )
    scheduler = Scheduler(world, runner)
    assert scheduler.run_turn(MAIN).status == "failed"
    reset = world.history.window(MAIN)
    assert reset.reason == "overflow"
    assert scheduler.tick() == (MAIN,)
    scheduler._threads[MAIN].join(timeout=5)
    assert not scheduler._threads[MAIN].is_alive()
    assert scheduler.agent_status(MAIN)["state"] == "error"
    assert world.history.lifecycle(MAIN).error == "too long"
    assert world.history.window(MAIN) == reset
    assert [run.status for run in world.history.runs(MAIN)[:2]] == ["failed", "failed"]
    assert [run.error for run in world.history.runs(MAIN)[:2]] == [
        "too long",
        "too long",
    ]
    assert len(runner.requests) == 2
    assert scheduler.tick() == ()


def test_overflow_reset_failure_keeps_error_and_does_not_continue(world) -> None:
    previous = world.history.start_run(MAIN)
    world.history.finish_run(
        MAIN, previous.sequence, status="completed", messages_json="[]"
    )
    room = mention(world)
    world.store._db.execute(
        "CREATE TEMP TRIGGER reject_overflow_window BEFORE INSERT ON agent_windows "
        "WHEN NEW.reason = 'overflow' "
        "BEGIN SELECT RAISE(ABORT, 'overflow reset failed'); END"
    )
    runner = RecordingRunner(
        lambda request, tools: TurnOutcome(
            messages_json="[]", error="too long", context_exceeded=True
        )
    )
    scheduler = Scheduler(world, runner)
    record = scheduler.run_turn(MAIN)
    assert record is not None and record.status == "failed"
    assert "overflow reset failed" in (record.error or "")
    assert world.history.window(MAIN).number == 1
    assert world.history.lifecycle(MAIN).error == record.error
    assert scheduler.agent_status(MAIN)["state"] == "error"
    assert scheduler.tick() == ()
    assert len(runner.requests) == 1
    assert world.history.runs(MAIN)[0].error == record.error
    assert world.store.pending(MAIN)
    assert room


@pytest.mark.parametrize("gate", ["paused", "blocked", "concurrent"])
def test_overflow_continuation_respects_scheduler_gates(world, gate) -> None:
    room = mention(world)
    previous = world.history.start_run(MAIN)
    world.history.finish_run(
        MAIN, previous.sequence, status="completed", messages_json="[]"
    )
    calls = 0

    def respond(request, tools):
        nonlocal calls
        calls += 1
        if calls == 1:
            return TurnOutcome(
                messages_json="[]", error="too long", context_exceeded=True
            )
        assert request.reminder is not None
        for item in request.reminder.items:
            world.store.ack(item.discussion_id, [item.message_id], MAIN)
        return TurnOutcome(messages_json="[]")

    world.settings.set_settings("agent", {"max_concurrent_turns": 1})
    runner = RecordingRunner(respond)
    scheduler = Scheduler(world, runner)
    world.store.append_message(room, HUMAN, "@Main continue")
    assert scheduler.run_turn(MAIN).status == "failed"

    if gate == "paused":
        scheduler.pause(MAIN)
        assert scheduler.agent_status(MAIN)["state"] == "paused"
    elif gate == "blocked":
        world.history.pause_for_safety(MAIN, "repeated_mentions")
        assert scheduler.agent_status(MAIN)["state"] == "blocked"
    else:
        with scheduler._lock:
            scheduler._reserved.add(HELPER)
        assert scheduler.tick() == ()
        with scheduler._lock:
            scheduler._reserved.remove(HELPER)

    if gate != "concurrent":
        assert scheduler.tick() == ()
        assert calls == 1
        scheduler.resume(MAIN)
    assert scheduler.tick() == (MAIN,)
    scheduler._threads[MAIN].join(timeout=5)
    assert not scheduler._threads[MAIN].is_alive()
    assert calls == 2
    assert world.history.runs(MAIN)[1].status == "failed"
    assert scheduler.tick() == ()


@pytest.mark.parametrize("fresh", [False, True])
def test_first_turn_overflow_is_an_ordinary_failure(world, fresh) -> None:
    mention(world)
    if fresh:
        spend(world, MAIN, 1)
        world.history.reset_window(MAIN, "prepared")
    state = world.history.window(MAIN)
    events = []
    runner = RecordingRunner(
        lambda request, tools: TurnOutcome(
            messages_json='[{"text":"partial"}]',
            error="too long",
            context_exceeded=True,
        )
    )
    scheduler = Scheduler(
        world, runner, on_event=lambda name, payload: events.append(name)
    )
    assert scheduler.run_turn(MAIN).status == "failed"
    assert world.history.window(MAIN) == state
    assert world.history.last_reminder(MAIN) == scheduler.pending_keys(MAIN)
    assert scheduler.tick() == ()
    assert world.history.latest_messages(MAIN) == '[{"text":"partial"}]'
    assert "window.reset" not in events


@pytest.mark.parametrize("status", ["completed", "failed", "interrupted", "running"])
@pytest.mark.parametrize("tokens", [None, 99, 100, 101])
def test_preparation_uses_the_latest_finished_run_in_the_current_window(
    world, status, tokens
) -> None:
    world.settings.set_settings("agent", {"context_window_tokens": 100})
    scheduler = Scheduler(world, RecordingRunner())
    assert not scheduler.preparation_due(MAIN)
    for run_status, count in [("completed", 200), (status, tokens)]:
        run = world.history.start_run(MAIN)
        world.history.finish_run(
            MAIN,
            run.sequence,
            status=run_status,
            messages_json="[]",
            usage_json=json.dumps({"last_input_tokens": count}),
        )
    assert scheduler.preparation_due(MAIN) == (
        status in ("completed", "failed") and tokens is not None and tokens >= 100
    )
    world.history.reset_window(MAIN, "prepared")
    assert not scheduler.preparation_due(MAIN)


def test_token_limit_and_member_state_still_block_preparation(world) -> None:
    world.settings.set_settings("agent", {"context_window_tokens": 100})
    run = world.history.start_run(MAIN)
    world.history.finish_run(
        MAIN,
        run.sequence,
        status="completed",
        messages_json="[]",
        usage_json='{"input_tokens":100,"last_input_tokens":100}',
    )
    runner = RecordingRunner()
    scheduler = Scheduler(world, runner)
    assert scheduler.preparation_due(MAIN)
    world.settings.set_settings(
        "agent", {"context_window_tokens": 100, "token_limit": 100}
    )
    assert scheduler.runnable_agents() == ()
    assert scheduler.run_turn(MAIN) is None
    world.settings.set_settings(
        "agent", {"context_window_tokens": 100, "token_limit": 101}
    )
    world.store.set_agent_state(MAIN, "paused")
    assert scheduler.runnable_agents() == ()
    assert scheduler.run_turn(MAIN) is None
    world.store.set_agent_state(MAIN, "idle")
    assert scheduler.runnable_agents() == (MAIN,)
    assert scheduler.run_turn(MAIN).status == "completed"
    assert runner.requests[-1].prompt == PREPARATION_PROMPT


def test_preparation_does_not_record_pending_mentions_as_reminded(world) -> None:
    room = mention(world)
    run = world.history.start_run(MAIN)
    world.history.finish_run(
        MAIN,
        run.sequence,
        status="completed",
        messages_json="[]",
        usage_json='{"last_input_tokens":200000}',
    )
    runner = RecordingRunner()
    scheduler = Scheduler(world, runner)
    assert scheduler.run_turn(MAIN).status == "completed"
    assert runner.requests[-1].reminder is None
    assert world.history.previously_reminded(MAIN, [(room, 1)]) == frozenset()
    assert scheduler.run_turn(MAIN).status == "completed"
    assert not runner.requests[-1].reminder.items[0].previously_reminded


def test_reset_notices_use_the_exact_window_timestamp() -> None:
    assert reset_notice(WindowState(1, 1, None, None)) is None
    assert reset_notice(WindowState(2, 3, "STAMP", "prepared")) == (
        "Your context window was reset at STAMP after you saved your notes. "
        "Use the history tool at least once to review your progress and tool results "
        "from before the previous window ended. Then continue using your saved records "
        "and the latest discussions. Look up only the history you need; you do not "
        "need to read it all."
    )
    assert reset_notice(WindowState(2, 3, "STAMP", "overflow")) == (
        "Your context window was reset at STAMP in the middle of a Turn because it "
        "overflowed, so you could not save notes first. Use the history tool to "
        "inspect the most recent interrupted turn, the last completed operation, and "
        "any tool calls that did not return. Confirm where to continue from. Look up "
        "only the history you need; you do not need to read it all."
    )


def test_no_reminder_without_pending_mentions(world) -> None:
    assert build_reminder(world.store, world.history, MAIN, "Main") is None


@pytest.mark.parametrize("fail", [False, True])
def test_pause_and_resume_keep_pending_inside_one_session(world, fail) -> None:
    mention(world)
    runner = RecordingRunner(
        lambda request, tools: TurnOutcome(
            messages_json="[]", error="failure" if fail else None
        )
    )
    scheduler = Scheduler(world, runner)
    assert scheduler.run_turn(MAIN) is not None
    scheduler.pause(MAIN)
    assert scheduler.run_turn(MAIN) is None
    scheduler.resume(MAIN)
    assert scheduler.runnable_agents() == (MAIN,)
    assert scheduler.run_turn(MAIN) is not None
    assert len(runner.requests) == 2


@pytest.mark.parametrize("fail", [False, True])
def test_a_new_session_continues_unchanged_mentions_once(world, fail) -> None:
    mention(world)
    first = RecordingRunner(
        lambda request, tools: TurnOutcome(
            messages_json="[]", error="failure" if fail else None
        )
    )
    scheduler = Scheduler(world, first)
    assert scheduler.run_turn(MAIN) is not None
    assert scheduler.runnable_agents() == (() if fail else (MAIN,))

    world.history.mark_interrupted()
    assert world.history.mark_session_start() == 1

    second = RecordingRunner()
    restart = Scheduler(world, second)
    if fail:
        assert restart.runnable_agents() == ()
        assert restart.run_turn(MAIN) is None
        assert second.requests == []
        return
    assert restart.runnable_agents() == (MAIN,)
    record = restart.run_turn(MAIN)
    assert record is not None
    assert record.status == "completed"
    reminder = second.requests[0].reminder
    assert reminder is not None
    assert [(item.discussion_id, item.message_id) for item in reminder.items] == [
        (1, 1)
    ]
    assert reminder.items[0].previously_reminded
    assert "[previously reminded]" in second.requests[0].prompt
    assert restart.runnable_agents() == (MAIN,)
    assert restart.run_turn(MAIN).status == "completed"
    assert restart.run_turn(MAIN) is None
    assert world.history.pause_reason(MAIN) == "repeated_mentions"
    assert len(second.requests) == 2


def test_a_new_session_reminds_only_the_unhandled_mentions(world) -> None:
    runner = RecordingRunner(lambda request, tools: TurnOutcome(messages_json="[]"))
    scheduler = Scheduler(world, runner)
    mention(world)
    assert scheduler.run_turn(MAIN) is not None

    reviewed = world.store.create_discussion("review", [HUMAN, HELPER])
    world.store.append_message(reviewed.id, HUMAN, "@Helper please review")
    assert scheduler.run_turn(HELPER) is not None
    world.store.ack(reviewed.id, [1], HELPER)
    assert world.store.pending(HELPER) == ()

    assert world.history.mark_session_start() == 2
    restarted = Scheduler(world, RecordingRunner())
    assert restarted.runnable_agents() == (MAIN,)
    assert restarted.run_turn(HELPER) is None


def test_a_new_session_keeps_safety_pauses_and_token_limits(world) -> None:
    mention(world)
    runner = RecordingRunner(
        lambda request, tools: TurnOutcome(messages_json="[]", error="no model")
    )
    scheduler = Scheduler(world, runner)
    assert scheduler.run_turn(MAIN) is not None
    world.history.mark_session_start()

    world.history.pause_for_safety(MAIN, "runtime_error")
    world.store.set_agent_state(MAIN, "paused")
    restarted = Scheduler(world, runner)
    assert restarted.runnable_agents() == ()
    assert restarted.run_turn(MAIN) is None

    world.store.set_agent_state(MAIN, "idle")
    assert restarted.runnable_agents() == ()
    assert restarted.run_turn(MAIN) is None

    restarted.resume(MAIN)
    assert restarted.runnable_agents() == (MAIN,)

    world.settings.set_settings("agent", {"token_limit": 10})
    spend(world, MAIN, 100)
    assert restarted.over_token_limit(MAIN)
    assert restarted.runnable_agents() == ()


@pytest.mark.parametrize("change", ["partial_ack", "all_ack", "new_mention"])
@pytest.mark.parametrize("fail", [False, True])
def test_next_turn_compares_mentions_with_the_start_of_the_previous_turn(
    world, change, fail
) -> None:
    room = mention(world)
    world.store.append_message(room, HUMAN, "@Main another task")

    def respond(request, tools):
        if request.sequence == 1:
            tools.read_discussion(room)
            if change == "partial_ack":
                tools.ack(room, [1])
            elif change == "all_ack":
                tools.ack(room, [1, 2])
            else:
                world.store.append_message(room, HUMAN, "@Main arrived during the turn")
        return TurnOutcome(messages_json="[]", error="failure" if fail else None)

    scheduler = Scheduler(world, RecordingRunner(respond))
    assert scheduler.run_turn(MAIN) is not None
    restarted = Scheduler(world, RecordingRunner())
    eligible = change == "new_mention" or (change == "partial_ack" and not fail)
    assert restarted.runnable_agents() == ((MAIN,) if eligible else ())
    assert (restarted.run_turn(MAIN) is not None) == eligible
    assert restarted.runnable_agents() == ((MAIN,) if eligible else ())


def test_turn_records_history_and_returns_to_idle(world) -> None:
    mention(world)
    runner = RecordingRunner()
    scheduler = Scheduler(world, runner)
    record = scheduler.run_turn(MAIN)
    assert record is not None
    assert record.status == "completed"
    assert runner.requests[0].sequence == record.sequence
    assert world.store.get_member(MAIN).state == "idle"
    assert world.history.latest_messages(MAIN) == json.dumps([{"kind": "response"}])


@pytest.mark.parametrize("fail", [False, True])
def test_resident_is_persisted_and_stays_unchanged_until_a_reset(
    world, monkeypatch, fail
) -> None:
    from pydantic_ai import ModelMessagesTypeAdapter, models
    from pydantic_ai.messages import (
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )
    from pydantic_ai.models.function import FunctionModel

    from huddol.adapters.model.runner import PydanticModelRunner

    room = mention(world, "@Main keep the discussion text")
    received = []

    def respond(messages, info):
        received.append(ModelMessagesTypeAdapter.dump_json(messages).decode())
        if isinstance(messages[-1].parts[-1], ToolReturnPart):
            if fail:
                raise RuntimeError("local model failure")
            return ModelResponse(parts=[TextPart("Keep the model response")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "discussion",
                    {"action": "read", "discussion_id": room, "message_id": 1},
                    tool_call_id=f"read-{len(received)}",
                )
            ]
        )

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    world.settings.set_settings(
        "model",
        {"base_url": "https://example.invalid", "api_key": "unused", "model": "local"},
    )
    runner = PydanticModelRunner(
        world.settings, build_model=lambda config: FunctionModel(respond)
    )
    scheduler = Scheduler(world, runner)
    tools = scheduler.tools_for(MAIN)
    world.workspace_tree_for(MAIN).write("MEMORY.md", "WORKSPACE_STATE_OLD")

    for turn, state in enumerate(("OLD", "NEW"), start=1):
        if state == "NEW":
            tools.edit(
                "workspace/MEMORY.md",
                [
                    {
                        "old_text": "WORKSPACE_STATE_OLD",
                        "new_text": "WORKSPACE_STATE_NEW",
                    }
                ],
            )
            world.store.append_message(room, HUMAN, "@Main continue")
        reminder = build_reminder(world.store, world.history, MAIN, "Main")
        assert reminder is not None
        start = len(received)

        record = scheduler.run_turn(MAIN)

        assert record is not None
        assert record.status == ("failed" if fail else "completed")
        assert len(received) == start + 2
        for request in received[start:]:
            assert "WORKSPACE_STATE_OLD" in request
            assert "WORKSPACE_STATE_NEW" not in request
        saved = world.history.latest_messages(MAIN)
        assert "WORKSPACE_STATE_OLD" in saved
        messages = ModelMessagesTypeAdapter.validate_json(saved)
        prompts = [
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ]
        assert len(prompts) == turn + 1
        assert prompts[-1] == reminder.render()
        assert "@Main keep the discussion text" in saved
        assert (
            sum(
                isinstance(part, ToolCallPart)
                for message in messages
                for part in message.parts
            )
            == turn
        )
        assert (
            sum(
                isinstance(part, ToolReturnPart)
                for message in messages
                for part in message.parts
            )
            == turn
        )
        if not fail:
            assert "Keep the model response" in saved

    world.history.reset_window(MAIN, "prepared")
    assert scheduler.runnable_agents() == (() if fail else (MAIN,))
    world.store.append_message(room, HUMAN, "@Main continue after reset")
    record = scheduler.run_turn(MAIN)
    assert record is not None and record.status == ("failed" if fail else "completed")
    saved = world.history.latest_messages(MAIN)
    assert "WORKSPACE_STATE_NEW" in saved
    assert "WORKSPACE_STATE_OLD" not in saved
    assert saved.count('"block":"resident"') == 1


@pytest.mark.parametrize("reset", [False, True])
def test_history_is_preserved_or_reset_without_rewriting_old_prompts(
    world, monkeypatch, reset: bool
) -> None:
    from pydantic_ai import ModelMessagesTypeAdapter, models
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )
    from pydantic_ai.models.function import FunctionModel

    from huddol.adapters.model.runner import PydanticModelRunner

    mention(world)
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    world.settings.set_settings(
        "model",
        {
            "base_url": "https://example.invalid",
            "api_key": "unused",
            "model": "local",
        },
    )
    runner = PydanticModelRunner(
        world.settings,
        build_model=lambda config: FunctionModel(
            lambda messages, info: ModelResponse(parts=[TextPart("New response")])
        ),
    )
    scheduler = Scheduler(world, runner)
    reminder = build_reminder(world.store, world.history, MAIN, "Main")
    assert reminder is not None
    prompt = f"{reminder.render()}\n\n{scheduler.resident_block(MAIN)}"
    prior = []
    for index in range(6):
        prior.extend(
            [
                ModelRequest(parts=[UserPromptPart(prompt)], metadata={"index": index}),
                ModelResponse(parts=[TextPart(f"Saved response {index}")]),
            ]
        )
    original = ModelMessagesTypeAdapter.dump_json(prior).decode()
    old_run = world.history.start_run(MAIN)
    world.history.finish_run(
        MAIN, old_run.sequence, status="completed", messages_json=original
    )
    if reset:
        world.history.reset_window(MAIN, "prepared")

    record = scheduler.run_turn(MAIN)

    assert record is not None and record.status == "completed"
    saved = ModelMessagesTypeAdapter.validate_json(world.history.latest_messages(MAIN))
    if reset:
        assert len(saved) == 3
        assert saved[0].parts[0].content == scheduler.resident_block(MAIN)
    else:
        key = saved[0].metadata.pop("huddol")["cache_key"]
        assert len(key) == 32
        assert saved[: len(prior)] == prior
        assert saved[-2].metadata == {
            "huddol": {
                "block": "durable",
                "environment": scheduler.environment_facts(MAIN),
            }
        }
    assert saved[1 if reset else len(prior)].parts[0].content == reminder.render()
    assert saved[-1].parts[0].content == "New response"
    assert world.history.runs(MAIN)[1].messages_json == original


def test_resident_carries_workspace_and_environment(world, tmp_path: Path) -> None:
    mention(world)
    DirectoryTree(world.workspace_tree_for(MAIN).root).write(
        "MEMORY.md", "- prior knowledge"
    )
    runner = RecordingRunner()
    Scheduler(world, runner).run_turn(MAIN)
    context = runner.requests[0].resident
    assert context == (
        "Your MEMORY.md:\n- prior knowledge\n\n"
        f"Commands run on {sys.platform}\n"
        "Writable directories:\n"
        f"- {world.workspace_tree_for(MAIN).root} (your workspace, private)\n"
        f"- {world.library_tree.root} (Library, shared with the whole organization)\n"
        f"- {tmp_path}"
    )


@pytest.mark.parametrize("preparation", [False, True])
@pytest.mark.parametrize("error_kind", ["os", "domain"])
def test_resident_creation_failure_preserves_window_and_other_agents(
    world, monkeypatch, error_kind, preparation
) -> None:
    from huddol.core.errors import DomainError

    room = mention(world)
    world.settings.set_settings("agent", {"context_window_tokens": 100})

    def respond(request, tools):
        tools.read_discussion(room)
        tools.ack(room, [1])
        return TurnOutcome(
            messages_json='[{"text":"saved"}]',
            usage_json=json.dumps({"last_input_tokens": 100 if preparation else 1}),
        )

    runner = RecordingRunner(respond)
    scheduler = Scheduler(world, runner)
    assert scheduler.run_turn(MAIN).status == "completed"
    previous = world.history.latest_messages(MAIN)
    window = world.history.window(MAIN)
    tree = world.workspace_tree_for(MAIN)
    (tree.root / "MEMORY.md").unlink()

    def fail(*args, **kwargs):
        if error_kind == "os":
            raise OSError("Read-only filesystem")
        raise DomainError("invalid_path", "Cannot create index")

    original = world.workspace_tree_for
    monkeypatch.setattr(tree, "write", fail)
    monkeypatch.setattr(
        world,
        "workspace_tree_for",
        lambda agent_id: tree if agent_id == MAIN else original(agent_id),
    )
    if not preparation:
        mention(world)
    record = scheduler.run_turn(MAIN)
    assert record.status == "failed"
    assert len(runner.requests) == 1
    assert world.history.latest_messages(MAIN) == previous
    assert world.history.window(MAIN) == window
    assert scheduler.agent_status(MAIN)["state"] == "error"
    assert world.history.pause_reason(MAIN) is None
    assert not (tree.root / "MEMORY.md").exists()

    helper_room = world.store.create_discussion("helper", [HUMAN, HELPER])
    world.store.append_message(helper_room.id, HUMAN, "@Helper continue")
    helper_runner = RecordingRunner()
    assert Scheduler(world, helper_runner).run_turn(HELPER).status == "completed"
    assert len(helper_runner.requests) == 1


@pytest.mark.parametrize("environment", [None, "environment facts"])
@pytest.mark.parametrize("workspace", ["", "remember this"])
@pytest.mark.parametrize("reset", [None, "Window reset notice"])
def test_render_resident_orders_only_the_requested_blocks(
    workspace, environment, reset
) -> None:
    expected = (
        "Your MEMORY.md:\nremember this" if workspace else "Your MEMORY.md is empty."
    )
    if environment is not None:
        expected += "\n\nenvironment facts"
    if reset is not None:
        expected += "\n\n" + reset
    assert render_resident(workspace, environment, reset) == expected


def test_workspace_index_truncation_follows_the_current_parameter(world) -> None:
    world.workspace_tree_for(MAIN).write("MEMORY.md", "abc😀def")
    runner = RecordingRunner()
    scheduler = Scheduler(world, runner)
    for limit, prefix, truncated in [(5, "abc", True), (10, "abc😀def", False)]:
        mention(world)
        world.settings.set_settings("agent", {"memory_index_bytes": limit})
        assert scheduler.run_turn(MAIN).status == "completed"
        resident = runner.requests[-1].resident
        assert resident.startswith(f"Your MEMORY.md:\n{prefix}\n")
        assert ("was cut here" in resident) is truncated
        if truncated:
            assert f"longer than {limit} bytes" in resident
            assert "😀" not in resident


def test_resident_loading_failure_preserves_history_and_records_error(
    world, monkeypatch
) -> None:
    mention(world)
    scheduler = Scheduler(world, RecordingRunner())
    scheduler.run_turn(MAIN)
    previous = world.history.latest_messages(MAIN)

    def fail(agent_id):
        raise OSError("workspace is unreadable")

    monkeypatch.setattr(scheduler, "resident_block", fail)
    mention(world)
    record = scheduler.run_turn(MAIN)
    assert record is not None and record.status == "failed"
    assert "workspace is unreadable" in record.error
    assert world.history.latest_messages(MAIN) == previous
    assert scheduler.agent_status(MAIN)["state"] == "error"
    assert world.history.pause_reason(MAIN) is None


def test_second_model_call_failure_keeps_saved_response_and_completed_tool_return(
    world, monkeypatch
) -> None:
    from pydantic_ai import ModelMessagesTypeAdapter, models
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import FunctionModel

    from huddol.adapters.model.runner import PydanticModelRunner

    room = mention(world)
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    world.settings.set_settings(
        "model",
        {"base_url": "https://example.invalid", "api_key": "unused", "model": "local"},
    )
    persisted = []
    save = world.history.save_progress
    calls = 0

    def save_progress(agent_id, sequence, raw):
        persisted.append(raw)
        save(agent_id, sequence, raw)
        run = world.history.runs(agent_id)[0]
        assert run.sequence == sequence
        assert run.status == "running" and run.completed_at is None
        assert run.messages_json == raw

    def respond(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "discussion",
                        {"action": "read", "discussion_id": room},
                        tool_call_id="read",
                    )
                ]
            )
        assert len(persisted) == 2
        assert world.history.latest_messages(MAIN) == persisted[1]
        raise RuntimeError("second call failed")

    monkeypatch.setattr(world.history, "save_progress", save_progress)
    runner = PydanticModelRunner(
        world.settings, build_model=lambda config: FunctionModel(respond)
    )
    scheduler = Scheduler(world, runner)
    record = scheduler.run_turn(MAIN)
    assert record is not None and record.status == "failed"
    assert record.error == "RuntimeError: second call failed"
    assert calls == 2 and len(persisted) == 2
    saved = ModelMessagesTypeAdapter.validate_json(persisted[1])
    run = world.history.runs(MAIN)[0]
    captured = ModelMessagesTypeAdapter.validate_json(run.messages_json)
    assert captured[:-1] == saved[:-1]
    assert isinstance(saved[-1], ModelRequest)
    assert isinstance(saved[-1].parts[0], ToolReturnPart)
    assert saved[-1].parts[0].tool_call_id == "read"
    assert saved[-1].parts[0].content == (
        "The Turn ended without a confirmed result for this call. "
        "It may have executed; check the outcome before retrying."
    )
    assert isinstance(captured[-1], ModelRequest)
    assert len(captured[-1].parts) == 1
    returned = captured[-1].parts[0]
    assert isinstance(returned, ToolReturnPart)
    assert returned.tool_call_id == "read"
    assert returned.content["messages"][0]["body"] == "@Main please help"
    assert world.history.latest_messages(MAIN) == run.messages_json


@pytest.mark.parametrize("fail_at", [1, 2, 3])
@pytest.mark.parametrize("prior", [False, True])
def test_sqlite_progress_failure_preserves_history_and_stops_execution(
    world, monkeypatch, fail_at, prior
):
    from pydantic_ai import ModelMessagesTypeAdapter, models
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        UserPromptPart,
    )
    from pydantic_ai.models.function import FunctionModel

    from huddol.adapters.model.runner import PydanticModelRunner

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    world.settings.set_settings(
        "model",
        {"base_url": "https://example.invalid", "api_key": "unused", "model": "local"},
    )
    snapshot = "instructions\n "
    original = "[]"
    if prior:
        original = ModelMessagesTypeAdapter.dump_json(
            [
                ModelRequest(
                    parts=[UserPromptPart("resident")],
                    metadata={
                        "huddol": {"block": "resident", "agents_instructions": snapshot}
                    },
                )
            ]
        ).decode()
        previous = world.history.start_run(MAIN)
        world.history.finish_run(
            MAIN, previous.sequence, status="completed", messages_json=original
        )
    else:
        world.library_tree.write("AGENTS.md", snapshot)
    mention(world)
    before_window = world.history.window(MAIN)
    save = world.history.save_progress
    attempted = []
    persisted = []
    model_calls = []
    tool_calls = []
    list_members = AgentTools.list_members

    def track_tools(self):
        tool_calls.append(1)
        return list_members(self)

    def save_progress(agent_id, sequence, raw):
        attempted.append(raw)
        before = world.history.runs(MAIN)[0].messages_json
        if len(attempted) == fail_at:
            world.store._db.execute(
                "CREATE TEMP TRIGGER fail_progress AFTER UPDATE OF messages_json"
                " ON agent_runs WHEN NEW.status = 'running'"
                " BEGIN SELECT RAISE(ABORT, 'progress write failed'); END"
            )
            with pytest.raises(sqlite3.IntegrityError) as caught:
                save(agent_id, sequence, raw)
            assert world.history.runs(MAIN)[0].messages_json == before
            raise caught.value
        save(agent_id, sequence, raw)
        persisted.append(raw)

    def respond(messages, info):
        model_calls.append(1)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "organization",
                    {"action": "list_members"},
                    tool_call_id=f"{len(model_calls)}-{index}",
                )
                for index in range(3)
            ]
        )

    monkeypatch.setattr(world.history, "save_progress", save_progress)
    monkeypatch.setattr(AgentTools, "list_members", track_tools)
    scheduler = Scheduler(
        world,
        PydanticModelRunner(
            world.settings, build_model=lambda config: FunctionModel(respond)
        ),
    )
    record = scheduler.run_turn(MAIN)
    assert record.status == "failed"
    assert "HistoryPersistenceError" in record.error
    assert len(attempted) == fail_at
    assert len(model_calls) == fail_at - (not prior)
    assert len(tool_calls) == max(0, len(model_calls) - 1) * 3
    expected = persisted[-1] if persisted else original
    assert world.history.runs(MAIN)[0].messages_json == expected
    assert world.history.latest_messages(MAIN) == expected
    if expected != "[]":
        resident = ModelMessagesTypeAdapter.validate_json(expected)[0]
        assert resident.metadata["huddol"]["agents_instructions"] == snapshot
    if prior:
        assert world.history.runs(MAIN)[1].messages_json == original
    assert world.history.pause_reason(MAIN) is None
    assert scheduler.agent_status(MAIN)["state"] == "error"
    assert world.history.window(MAIN) == before_window
    assert scheduler.run_turn(MAIN) is None
    mention(world)
    assert MAIN in scheduler.runnable_agents()


@pytest.mark.parametrize(
    "failure_stage", [None, "snapshot", "response", "middle", "finish"]
)
def test_turn_model_snapshot_survives_history_saving_and_failure(
    world, monkeypatch, failure_stage
):
    from pydantic_ai import ModelMessagesTypeAdapter, models
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from huddol.adapters.model.config import ModelCatalog
    from huddol.adapters.model.prompt import SYSTEM_PROMPT
    from huddol.adapters.model.runner import PydanticModelRunner

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    first = ModelCatalog.model_validate(
        {
            "providers": [
                {
                    "id": "first-provider",
                    "name": "First",
                    "api_type": "openai-chat",
                    "base_url": "https://first.invalid",
                    "api_key": "unused-first",
                }
            ],
            "models": [
                {
                    "id": "first-model",
                    "provider_id": "first-provider",
                    "name": "First",
                    "model": "gpt-5",
                }
            ],
            "default_model_id": "first-model",
            "default_thinking": "high",
        }
    )
    second = ModelCatalog.model_validate(
        {
            "providers": [
                {
                    "id": "second-provider",
                    "name": "Second",
                    "api_type": "openai-responses",
                    "base_url": "https://second.invalid",
                    "api_key": "unused-second",
                }
            ],
            "models": [
                {
                    "id": "second-model",
                    "provider_id": "second-provider",
                    "name": "Second",
                    "model": "gpt-5.2",
                }
            ],
            "default_model_id": "second-model",
            "default_thinking": "low",
        }
    )
    world.settings.set_settings("model", first.model_dump())
    world.library_tree.write("AGENTS.md", "window instructions\n ")
    room = mention(world)
    built = []
    calls = []
    executions = []
    snapshots = []
    saved = []
    attempts = []
    save = world.history.save_progress
    list_members = AgentTools.list_members
    fail_at = {"snapshot": 1, "response": 2, "middle": 3}.get(failure_stage)
    window = world.history.window(MAIN)

    def persist(agent_id, sequence, raw):
        attempts.append(raw)
        if len(attempts) == 1:
            world.settings.set_settings("model", second.model_dump())
        if len(attempts) == fail_at:
            world.store._db.execute(
                "CREATE TEMP TRIGGER fail_snapshot AFTER UPDATE OF messages_json"
                " ON agent_runs WHEN NEW.status = 'running'"
                " BEGIN SELECT RAISE(ABORT, 'snapshot write failed'); END"
            )
        save(agent_id, sequence, raw)
        saved.append(raw)

    def track_tools(self):
        executions.append(1)
        return list_members(self)

    def build(config):
        built.append(config)
        requests = 0

        def respond(messages, info):
            nonlocal requests
            requests += 1
            calls.append(config)
            snapshots.append(info.model_request_parameters.instruction_parts[0].content)
            if config == first.resolve(MAIN) and requests < 3:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "organization",
                            {"action": "list_members"},
                            tool_call_id=f"{requests}-{index}",
                        )
                        for index in range(3)
                    ]
                )
            if config == first.resolve(MAIN) and failure_stage == "finish":
                world.store._db.execute(
                    "CREATE TEMP TRIGGER fail_snapshot AFTER UPDATE OF completed_at"
                    " ON agent_runs WHEN NEW.status = 'completed'"
                    " BEGIN SELECT RAISE(ABORT, 'final write failed'); END"
                )
            return ModelResponse(parts=[TextPart("done")])

        return FunctionModel(respond)

    monkeypatch.setattr(world.history, "save_progress", persist)
    monkeypatch.setattr(AgentTools, "list_members", track_tools)
    scheduler = Scheduler(world, PydanticModelRunner(world.settings, build_model=build))
    result = scheduler.run_turn(MAIN)
    expected_calls, expected_tools = {
        None: (3, 6),
        "snapshot": (0, 0),
        "response": (1, 0),
        "middle": (2, 3),
        "finish": (3, 6),
    }[failure_stage]
    assert result.status == ("completed" if failure_stage is None else "failed"), (
        result.error
    )
    assert calls == [first.resolve(MAIN)] * expected_calls
    assert len(executions) == expected_tools
    assert built == ([first.resolve(MAIN)] if expected_calls else [])
    requests = world.history.model_request_summaries(MAIN, result.sequence)
    assert len(requests) == expected_calls
    assert all(
        item.run_id == world.history.runs(MAIN, limit=1)[0].run_id for item in requests
    )
    assert world.history.window(MAIN) == window
    history = world.history.latest_messages(MAIN)
    assert history == (saved[-1] if saved else "[]")
    if saved:
        assert (
            ModelMessagesTypeAdapter.validate_json(history)[0].metadata["huddol"][
                "agents_instructions"
            ]
            == "window instructions\n "
        )
    if failure_stage is not None:
        assert world.history.pause_reason(MAIN) is None
        assert scheduler.agent_status(MAIN)["state"] == "error"
        assert scheduler.run_turn(MAIN) is None
        world.store._db.execute("DROP TRIGGER fail_snapshot")
        scheduler.tools_for(HUMAN).resume_agent(MAIN)
    world.store.append_message(room, HUMAN, "@Main continue")
    assert scheduler.run_turn(MAIN).status == "completed"
    assert built[-1] == second.resolve(MAIN)
    assert calls == [first.resolve(MAIN)] * expected_calls + [second.resolve(MAIN)]
    assert len(executions) == expected_tools
    assert snapshots == [SYSTEM_PROMPT + "\n\nwindow instructions\n "] * (
        expected_calls + 1
    )
    assert world.history.window(MAIN) == window


@pytest.mark.parametrize("invalid_json", [False, True])
def test_invalid_history_preserves_sqlite_records_and_window(
    world, monkeypatch, invalid_json
):
    from pydantic_ai import models

    from huddol.adapters.model.runner import PydanticModelRunner

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    world.settings.set_settings(
        "model",
        {"base_url": "https://example.invalid", "api_key": "unused", "model": "local"},
    )
    original = json.dumps(
        [
            {
                "kind": "request",
                "parts": [{"part_kind": "user-prompt", "content": "resident"}],
                "metadata": {
                    "huddol": {"block": "resident", "agents_instructions": "snapshot"}
                },
            },
            {"kind": "unknown"},
        ]
    )
    if invalid_json:
        original = original[:-1]
    previous = world.history.start_run(MAIN)
    world.history.finish_run(
        MAIN, previous.sequence, status="completed", messages_json=original
    )
    window = world.history.window(MAIN)
    built = []
    runner = PydanticModelRunner(
        world.settings, build_model=lambda config: built.append(config)
    )
    mention(world)
    scheduler = Scheduler(world, runner)
    assert scheduler.run_turn(MAIN) is None
    assert built == []
    assert world.history.latest_messages(MAIN) == original
    assert [run.messages_json for run in world.history.runs(MAIN)] == [original]
    assert world.history.window(MAIN) == window
    assert scheduler.agent_status(MAIN)["state"] == "blocked"
    assert world.history.pause_reason(MAIN) == "history_invalid"


@pytest.mark.parametrize("mode", ["normal", "overflow", "preparation"])
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("invalid_parameters", [False, True])
def test_final_sqlite_failure_protects_progress_and_window(
    world, mode, continuous, invalid_parameters, caplog
):
    original = '[{"text":"original"}]'
    progress = '[{"text":"saved progress"}]'
    previous = world.history.start_run(MAIN)
    world.history.finish_run(
        MAIN,
        previous.sequence,
        status="completed",
        messages_json=original,
        usage_json=json.dumps(
            {"last_input_tokens": 100 if mode == "preparation" else 1}
        ),
    )
    world.settings.set_settings("agent", {"context_window_tokens": 100})
    mention(world)
    window = world.history.window(MAIN)
    events = []
    calls = []

    def respond(request, tools):
        calls.append(1)
        request.persist(progress)
        if invalid_parameters:
            world.settings.set_settings("agent", {"no_tool_turns_before_pause": False})
        world.store._db.execute(
            "CREATE TEMP TRIGGER fail_finish AFTER UPDATE OF completed_at ON agent_runs"
            + (
                ""
                if continuous
                else " WHEN NEW.error IS NOT 'IntegrityError: final write failed'"
            )
            + " BEGIN SELECT RAISE(ABORT, 'final write failed'); END"
        )
        return TurnOutcome(
            messages_json='[{"text":"final"}]', context_exceeded=mode == "overflow"
        )

    scheduler = Scheduler(
        world, RecordingRunner(respond), on_event=lambda *event: events.append(event)
    )
    if continuous:
        with pytest.raises(
            sqlite3.IntegrityError, match="final write failed"
        ) as caught:
            scheduler.run_turn(MAIN)
        assert isinstance(caught.value.__context__, sqlite3.IntegrityError)
        assert world.history.runs(MAIN)[0].status == "running"
    else:
        record = scheduler.run_turn(MAIN)
        assert record.status == "failed"
        assert world.history.runs(MAIN)[0].status == "failed"
    assert "final write failed" in caplog.text
    assert calls == [1]
    assert world.history.runs(MAIN)[0].messages_json == progress
    assert world.history.runs(MAIN)[1].messages_json == original
    assert world.history.latest_messages(MAIN) == progress
    assert world.history.window(MAIN) == window
    assert all(name != "window.reset" for name, _ in events)
    assert scheduler.agent_status(MAIN)["state"] == (
        "blocked" if continuous or invalid_parameters else "error"
    )
    assert world.history.pause_reason(MAIN) is None
    assert scheduler.run_turn(MAIN) is None
    mention(world)
    assert (MAIN in scheduler.runnable_agents()) == (
        not continuous and not invalid_parameters
    )
    assert calls == [1]


@pytest.mark.parametrize("invalid_parameters", [False, True])
def test_progress_finish_and_safety_failures_remain_diagnostic_and_stop_agent(
    world, monkeypatch, caplog, invalid_parameters
):
    from pydantic_ai import models
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from huddol.adapters.model.runner import PydanticModelRunner
    from huddol.runtime.reminder import HistoryPersistenceError

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    world.settings.set_settings(
        "model",
        {"base_url": "https://example.invalid", "api_key": "unused", "model": "local"},
    )
    mention(world)
    calls = []
    tools = []

    def respond(messages, info):
        calls.append(1)
        if invalid_parameters:
            world.settings.set_settings("agent", {"no_tool_turns_before_pause": False})
        world.store._db.executescript(
            "CREATE TEMP TRIGGER fail_history AFTER UPDATE ON agent_runs"
            " BEGIN SELECT RAISE(ABORT, 'history unavailable'); END;"
            "CREATE TEMP TRIGGER fail_safety BEFORE INSERT ON agent_safety"
            " BEGIN SELECT RAISE(ABORT, 'safety unavailable'); END;"
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "organization", {"action": "list_members"}, tool_call_id=str(index)
                )
                for index in range(3)
            ]
        )

    monkeypatch.setattr(AgentTools, "list_members", lambda self: tools.append(1))
    scheduler = Scheduler(
        world,
        PydanticModelRunner(
            world.settings, build_model=lambda config: FunctionModel(respond)
        ),
    )
    with pytest.raises(sqlite3.IntegrityError, match="history unavailable") as caught:
        scheduler.run_turn(MAIN)
    finish_error = caught.value.__context__
    assert isinstance(finish_error, sqlite3.IntegrityError)
    progress_error = caught.value.__cause__
    assert isinstance(progress_error, HistoryPersistenceError)
    assert isinstance(progress_error.__cause__, sqlite3.IntegrityError)
    assert "history unavailable" in str(progress_error.__cause__)
    assert "history unavailable" in caplog.text
    assert calls == [1]
    assert tools == []
    assert world.history.runs(MAIN)[0].status == "running"
    assert world.store.get_member(MAIN).state == "running"
    assert world.history.window(MAIN).number == 1
    mention(world)
    assert scheduler.agent_status(MAIN)["state"] == "blocked"
    assert MAIN not in scheduler.runnable_agents()
    assert scheduler.run_turn(MAIN) is None
    assert calls == [1]
    assert tools == []


def test_scheduler_resumes_interrupted_progress_on_pending_mentions(world) -> None:
    room = mention(world)
    run = world.history.start_run(MAIN, reminded=[(room, 1)])
    partial = '[{"text":"partial"}]'
    world.history.save_progress(MAIN, run.sequence, partial)
    assert world.history.mark_interrupted() == 1
    runner = RecordingRunner()
    scheduler = Scheduler(world, runner)
    assert scheduler.runnable_agents() == (MAIN,)
    record = scheduler.run_turn(MAIN)
    assert record is not None and record.status == "completed"
    assert runner.requests[0].history_json == partial
    assert runner.requests[0].sequence == run.sequence + 1
    assert world.history.runs(MAIN)[1].messages_json == partial


def test_a_failing_turn_is_recorded_and_the_agent_recovers(world) -> None:
    mention(world)

    def explode(request, tools):
        return TurnOutcome(messages_json="[]", error="RuntimeError: model exploded")

    scheduler = Scheduler(world, RecordingRunner(explode))
    record = scheduler.run_turn(MAIN)
    assert record is not None
    assert record.status == "failed"
    assert "model exploded" in (record.error or "")
    assert scheduler.agent_status(MAIN)["state"] == "error"


def test_a_failed_turn_is_not_restarted_on_the_same_pending(world) -> None:
    room = mention(world)
    failing = RecordingRunner(
        lambda request, tools: TurnOutcome(messages_json="[]", error="no model")
    )
    scheduler = Scheduler(world, failing)
    try:
        assert scheduler.tick() == (MAIN,)
        scheduler._threads[MAIN].join(timeout=5)
        runs = world.history.runs(MAIN)
        assert [run.status for run in runs] == ["failed"]
        assert scheduler.agent_status(MAIN)["state"] == "error"
        assert world.store.pending(MAIN)

        assert scheduler.runnable_agents() == ()
        assert scheduler.tick() == ()
        assert len(world.history.runs(MAIN)) == 1

        world.store.append_message(room, HUMAN, "@Main are you there?")
        assert scheduler.runnable_agents() == (MAIN,)
        record = scheduler.run_turn(MAIN)
        assert record is not None and record.status == "failed"
        assert len(world.history.runs(MAIN)) == 2
        assert scheduler.runnable_agents() == ()

        scheduler._runner = RecordingRunner()
        assert scheduler.runnable_agents() == ()
        world.store.append_message(room, HUMAN, "@Main once more")
        assert scheduler.runnable_agents() == (MAIN,)
        record = scheduler.run_turn(MAIN)
        assert record is not None and record.status == "completed"
        assert scheduler.runnable_agents() == (MAIN,)
    finally:
        scheduler.stop()


def test_paused_agents_are_not_runnable(world) -> None:
    mention(world)
    world.store.set_agent_state(MAIN, "paused")
    scheduler = Scheduler(world, RecordingRunner())
    assert scheduler.runnable_agents() == ()
    assert scheduler.run_turn(MAIN) is None


def test_only_agents_with_pending_work_are_runnable(world) -> None:
    mention(world)
    scheduler = Scheduler(world, RecordingRunner())
    assert scheduler.runnable_agents() == (MAIN,)


def test_acking_inside_the_turn_stops_the_next_one(world) -> None:
    room = mention(world)

    def handle(request, tools):
        tools.read_discussion(room, message_id=1)
        tools.ack(room, [1])
        return TurnOutcome(messages_json="[]")

    scheduler = Scheduler(world, RecordingRunner(handle))
    scheduler.run_turn(MAIN)
    assert scheduler.runnable_agents() == ()


def test_an_agent_can_reopen_its_own_acknowledgement(world, monkeypatch) -> None:
    from pydantic_ai import models
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from huddol.adapters.model.runner import PydanticModelRunner

    room = mention(world)
    actions = iter(("read", "ack", "revoke_ack"))
    results = []

    def respond(messages, info):
        action = next(actions, None)
        if action is None:
            results.append(messages[-1].parts[0].content)
            return ModelResponse(parts=[TextPart("Reopened")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "discussion",
                    {"action": action, "discussion_id": room, "message_id": 1},
                    tool_call_id=action,
                )
            ]
        )

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    world.settings.set_settings(
        "model",
        {"base_url": "https://example.invalid", "api_key": "unused", "model": "local"},
    )
    runner = PydanticModelRunner(
        world.settings, build_model=lambda config: FunctionModel(respond)
    )

    record = Scheduler(world, runner).run_turn(MAIN)

    assert record is not None and record.status == "completed"
    assert results == [{"discussion_id": room, "revoked": 1}]
    assert world.store.acknowledged(room, MAIN) == ()
    assert [item.message_id for item in world.store.pending(MAIN)] == [1]


def test_an_agent_can_wake_another_by_mentioning_it(world) -> None:
    room = mention(world)
    world.store.set_discussion_members(room, [HUMAN, MAIN, HELPER])

    def handle(request, tools):
        tools.read_discussion(room, message_id=1)
        tools.send_message(room, "@Helper your turn")
        tools.ack(room, [1])
        return TurnOutcome(messages_json="[]")

    scheduler = Scheduler(world, RecordingRunner(handle))
    scheduler.run_turn(MAIN)
    assert scheduler.runnable_agents() == (HELPER,)


def test_events_are_emitted_around_a_turn(world) -> None:
    mention(world)
    seen: list[str] = []
    scheduler = Scheduler(
        world, RecordingRunner(), on_event=lambda name, payload: seen.append(name)
    )
    scheduler.run_turn(MAIN)
    assert seen == ["turn.started", "turn.finished"]


def test_previously_reminded_is_flagged_on_the_second_turn(world) -> None:
    mention(world)
    scheduler = Scheduler(world, RecordingRunner())
    first = build_reminder(world.store, world.history, MAIN, "Main")
    assert first is not None
    assert first.items[0].previously_reminded is False

    scheduler.run_turn(MAIN)
    second = build_reminder(world.store, world.history, MAIN, "Main")
    assert second is not None
    assert second.items[0].previously_reminded is True
    assert "still waiting" in second.render()


def test_prior_window_content_stays_retrievable_through_history(world) -> None:
    mention(world)
    from huddol.services.history import History

    early = json.dumps(
        [
            {
                "kind": "request",
                "parts": [{"part_kind": "user-prompt", "content": "bubblewrap notes"}],
            },
            {"kind": "response", "parts": [{"part_kind": "text", "content": "noted"}]},
        ]
    )
    first = world.history.start_run(MAIN)
    world.history.finish_run(
        MAIN, first.sequence, status="completed", messages_json=early
    )

    world.history.reset_window(MAIN, "prepared")
    later = json.dumps([{"kind": "response", "parts": []}])
    second = world.history.start_run(MAIN)
    world.history.finish_run(
        MAIN, second.sequence, status="completed", messages_json=later
    )

    assert "bubblewrap" not in world.history.latest_messages(MAIN)
    found = History(world.history, MAIN).search("bubblewrap")
    assert [item.sequence for item in found] == [first.sequence]


def test_a_long_back_and_forth_produces_a_nudge(world) -> None:
    room = world.store.create_discussion("ping pong", [HUMAN, MAIN, HELPER])
    for index in range(6):
        sender = MAIN if index % 2 == 0 else HELPER
        world.store.append_message(room.id, sender, f"turn {index}")
    world.store.append_message(room.id, HELPER, "@Main again")

    runner = RecordingRunner()
    Scheduler(world, runner).run_turn(MAIN)
    context = runner.requests[0].ephemeral()
    assert "exchanged" in context
    assert "Helper" in context
    assert "acknowledge instead of mentioning them again" in context


def test_nudge_threshold_is_read_again_for_each_model_call(world) -> None:
    room = world.store.create_discussion("exchange", [MAIN, HELPER])
    for sender, body in [(HELPER, "one"), (MAIN, "two"), (HELPER, "three @Main")]:
        world.store.append_message(room.id, sender, body)
    runner = RecordingRunner()
    scheduler = Scheduler(world, runner)
    world.settings.set_settings("agent", {"exchange_nudge_after": 3})
    assert scheduler.run_turn(MAIN).status == "completed"
    request = runner.requests[0]
    assert "exchanged 3 messages" in request.ephemeral()
    world.settings.set_settings("agent", {"exchange_nudge_after": 6})
    assert request.ephemeral() == ""


def test_no_nudge_when_a_third_member_is_involved(world) -> None:
    room = world.store.create_discussion("group", [HUMAN, MAIN, HELPER])
    for index in range(6):
        sender = [MAIN, HELPER, HUMAN][index % 3]
        world.store.append_message(room.id, sender, f"turn {index} @Main")

    runner = RecordingRunner()
    Scheduler(world, runner).run_turn(MAIN)
    assert runner.requests[0].ephemeral() == ""


def test_no_nudge_for_a_short_exchange(world) -> None:
    room = world.store.create_discussion("brief", [HUMAN, MAIN])
    world.store.append_message(room.id, MAIN, "one")
    world.store.append_message(room.id, HUMAN, "two @Main")

    runner = RecordingRunner()
    Scheduler(world, runner).run_turn(MAIN)
    assert runner.requests[0].ephemeral() == ""


def test_no_nudge_for_an_exchange_you_are_not_part_of(world) -> None:
    room = world.store.create_discussion("others", [HUMAN, MAIN, HELPER])
    for index in range(6):
        sender = HUMAN if index % 2 == 0 else HELPER
        world.store.append_message(room.id, sender, f"turn {index} @Main")

    runner = RecordingRunner()
    Scheduler(world, runner).run_turn(MAIN)
    assert runner.requests[0].ephemeral() == ""


def test_the_nudge_reaches_the_model_but_not_the_history(world) -> None:
    room = mention(world)
    world.store.set_discussion_members(room, [HUMAN, MAIN, HELPER])
    for index in range(6):
        sender = MAIN if index % 2 == 0 else HELPER
        world.store.append_message(room, sender, f"turn {index}")
    world.store.append_message(room, HELPER, "@Main once more")

    runner = RecordingRunner()
    Scheduler(world, runner).run_turn(MAIN)
    assert "exchanged" in runner.requests[0].ephemeral()
    world.store.append_message(room, HUMAN, "A third participant")
    assert runner.requests[0].ephemeral() == ""
    assert "exchanged" not in world.history.latest_messages(MAIN)


@pytest.mark.parametrize("fail", [False, True])
def test_concurrency_follows_the_current_parameter(world, fail) -> None:
    import threading

    entered = {agent_id: threading.Event() for agent_id in (MAIN, HELPER)}
    release = {agent_id: threading.Event() for agent_id in (MAIN, HELPER)}

    def slow(request, tools):
        entered[request.agent_id].set()
        assert release[request.agent_id].wait(timeout=5)
        if fail:
            return TurnOutcome(messages_json="[]", error="RuntimeError: model failed")
        return TurnOutcome(messages_json="[]")

    room = world.store.create_discussion("parallel", [HUMAN, MAIN, HELPER])
    world.store.append_message(room.id, HUMAN, "@Main @Helper go")
    world.settings.set_settings("agent", {"max_concurrent_turns": 1})
    scheduler = Scheduler(world, RecordingRunner(slow))
    try:
        assert scheduler.tick() == (MAIN,)
        assert entered[MAIN].wait(timeout=5)
        assert not entered[HELPER].is_set()
        assert scheduler.tick() == ()

        world.settings.set_settings("agent", {"max_concurrent_turns": 2})
        assert scheduler.tick() == (HELPER,)
        assert entered[HELPER].wait(timeout=5)
        assert scheduler.tick() == ()

        world.settings.set_settings("agent", {"max_concurrent_turns": 1})
        assert scheduler.tick() == ()
        assert all(thread.is_alive() for thread in scheduler._threads.values())
        release[MAIN].set()
        scheduler._threads[MAIN].join(timeout=5)
        assert not scheduler._threads[MAIN].is_alive()
        world.store.append_message(room.id, HUMAN, "@Main next")
        assert scheduler.runnable_agents() == (MAIN,)
        assert scheduler.tick() == ()

        release[HELPER].set()
        scheduler._threads[HELPER].join(timeout=5)
        assert not scheduler._threads[HELPER].is_alive()
        release[MAIN].clear()
        assert scheduler.tick() == (MAIN,)
    finally:
        for event in release.values():
            event.set()
        scheduler.stop()


def test_stop_is_safe_while_a_turn_is_being_started(world) -> None:
    import threading

    scheduler = Scheduler(world, RecordingRunner())
    never_started = threading.Thread(target=lambda: None)
    scheduler._threads[MAIN] = never_started
    scheduler.stop()


def test_stop_waits_for_a_running_turn(world) -> None:
    import threading

    release = threading.Event()

    def slow(request, tools):
        release.wait(timeout=5)
        return TurnOutcome(messages_json="[]")

    mention(world)
    scheduler = Scheduler(world, RecordingRunner(slow))
    assert scheduler.tick() == (MAIN,)
    release.set()
    scheduler.stop()
    assert world.store.get_member(MAIN).state == "idle"


def test_the_scheduler_loop_is_joined_on_stop(world) -> None:
    scheduler = Scheduler(world, RecordingRunner())
    scheduler.start(poll_seconds=0.01)
    assert scheduler._loop is not None
    loop = scheduler._loop
    scheduler.stop()
    assert loop.is_alive() is False


def test_starting_twice_keeps_a_single_loop(world) -> None:
    scheduler = Scheduler(world, RecordingRunner())
    scheduler.start(poll_seconds=0.01)
    first = scheduler._loop
    scheduler.start(poll_seconds=0.01)
    assert scheduler._loop is first
    scheduler.stop()


def spend(world, agent_id: int, tokens: int) -> None:
    run = world.history.start_run(agent_id)
    world.history.finish_run(
        agent_id,
        run.sequence,
        status="completed",
        messages_json="[]",
        usage_json=json.dumps({"input_tokens": tokens, "output_tokens": 0}),
    )


def test_an_agent_over_its_token_limit_is_not_scheduled(world) -> None:
    mention(world)
    world.settings.set_settings("agent", {"token_limit": 500})
    scheduler = Scheduler(world, RecordingRunner())
    assert scheduler.runnable_agents() == (MAIN,)

    spend(world, MAIN, 499)
    assert not scheduler.over_token_limit(MAIN)
    assert scheduler.runnable_agents() == (MAIN,)
    spend(world, MAIN, 1)
    assert scheduler.over_token_limit(MAIN)
    assert scheduler.runnable_agents() == ()
    assert scheduler.run_turn(MAIN) is None
    assert world.store.get_member(MAIN).state == "idle"
    assert scheduler.tick() == ()


def test_raising_the_limit_lets_the_agent_run_again(world) -> None:
    mention(world)
    world.settings.set_settings("agent", {"token_limit": 100})
    spend(world, MAIN, 400)
    scheduler = Scheduler(world, RecordingRunner())
    assert scheduler.runnable_agents() == ()

    world.settings.set_settings("agent", {"token_limit": 1000})
    assert scheduler.runnable_agents() == (MAIN,)


@pytest.mark.parametrize("values", [{}, {"token_limit": 0}])
def test_no_limit_configured_means_no_ceiling(world, values) -> None:
    mention(world)
    world.settings.set_settings("agent", values)
    spend(world, MAIN, 10_000_000)
    scheduler = Scheduler(world, RecordingRunner())
    assert scheduler.token_limit() == 0
    assert not scheduler.over_token_limit(MAIN)
    assert scheduler.runnable_agents() == (MAIN,)


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("method", ["start", "tick", "run_turn"])
def test_invalid_parameters_prevent_scheduling(world, pending, method) -> None:
    from huddol.core.errors import DomainError

    if pending:
        mention(world)
    world.settings.set_settings("agent", {"token_limit": "not a number"})
    runner = RecordingRunner()
    scheduler = Scheduler(world, runner)
    if method == "run_turn":
        assert scheduler.run_turn(MAIN) is None
        assert scheduler.agent_status(MAIN)["state"] == "blocked"
    else:
        with pytest.raises(DomainError, match="token_limit"):
            getattr(scheduler, method)()
    assert scheduler._loop is None
    assert scheduler._threads == {}
    assert runner.requests == []
    assert world.history.runs(MAIN) == ()
    assert world.store.get_member(MAIN).state == "idle"


def test_parameter_failure_after_a_turn_preserves_history_and_pauses(world) -> None:
    mention(world)
    messages = '[{"kind":"response","text":"saved progress"}]'
    usage = '{"tool_calls":1,"input_tokens":10,"output_tokens":2}'

    def respond(request, tools):
        world.settings.set_settings("agent", {"no_tool_turns_before_pause": False})
        return TurnOutcome(messages_json=messages, usage_json=usage)

    events = []
    scheduler = Scheduler(
        world,
        RecordingRunner(respond),
        on_event=lambda name, payload: events.append((name, payload)),
    )
    result = scheduler.run_turn(MAIN)
    assert result.status == "failed"
    assert "no_tool_turns_before_pause" in result.error
    stored = world.history.runs(MAIN)[0]
    assert stored.messages_json == messages
    assert stored.usage_json == usage
    assert world.history.pause_reason(MAIN) is None
    assert scheduler.agent_status(MAIN)["state"] == "blocked"
    assert world.history.window(MAIN).number == 1
    assert events[-1] == (
        "turn.finished",
        {"agent_id": MAIN, "sequence": 1, "status": "failed"},
    )


@pytest.mark.parametrize("finish_failure", [False, True])
@pytest.mark.parametrize("safety_failure", [False, True])
def test_invalid_parameters_during_finalization_preserve_saved_result(
    world, finish_failure, safety_failure, caplog
):
    mention(world)
    events = []
    messages = '[{"text":"completed response"}]'
    usage = '{"tool_calls":1,"input_tokens":10,"output_tokens":2}'

    def respond(request, tools):
        request.persist('[{"text":"earlier progress"}]')
        world.settings.set_settings("agent", {"no_tool_turns_before_pause": False})
        if finish_failure:
            world.store._db.execute(
                "CREATE TEMP TRIGGER fail_error_record AFTER UPDATE ON agent_runs"
                " WHEN NEW.status = 'failed'"
                " BEGIN SELECT RAISE(ABORT, 'error record unavailable'); END"
            )
        if safety_failure:
            world.store._db.execute(
                "CREATE TEMP TRIGGER fail_lifecycle BEFORE UPDATE ON agent_lifecycle"
                " BEGIN SELECT RAISE(ABORT, 'lifecycle unavailable'); END"
            )
        return TurnOutcome(messages_json=messages, usage_json=usage)

    runner = RecordingRunner(respond)
    scheduler = Scheduler(world, runner, on_event=lambda *event: events.append(event))
    if finish_failure or safety_failure:
        message = (
            "error record unavailable" if finish_failure else "lifecycle unavailable"
        )
        with pytest.raises(sqlite3.IntegrityError, match=message) as caught:
            scheduler.run_turn(MAIN)
        original = caught.value.__context__
        assert isinstance(original, DomainError)
        assert "no_tool_turns_before_pause" in str(original)
    else:
        result = scheduler.run_turn(MAIN)
        assert result.status == "failed"
    stored = world.history.runs(MAIN)[0]
    assert stored.messages_json == messages
    assert stored.usage_json == usage
    assert stored.status == (
        "completed" if finish_failure or safety_failure else "failed"
    )
    assert "no_tool_turns_before_pause" in caplog.text
    assert world.history.window(MAIN).number == 1
    assert not any(name == "window.reset" for name, _ in events)
    assert world.history.pause_reason(MAIN) is None
    assert scheduler.agent_status(MAIN)["state"] == "blocked"
    assert MAIN not in scheduler._reserved
    assert events[-1][0] == "turn.finished"
    assert events[-1][1]["status"] == "failed"
    assert scheduler.run_turn(MAIN) is None
    assert len(runner.requests) == 1


def test_scheduler_failure_logs_and_finishes_inflight_turns(
    world, tmp_path, monkeypatch
) -> None:
    import logging
    import threading

    from huddol.__main__ import configure_logging

    entered = threading.Event()
    release = threading.Event()
    failures = []
    monkeypatch.setattr(threading, "excepthook", failures.append)
    handlers = configure_logging(tmp_path)
    mention(world)

    def respond(request, tools):
        entered.set()
        assert release.wait(5)
        return TurnOutcome(
            messages_json='[{"kind":"response"}]', usage_json='{"tool_calls":1}'
        )

    runner = RecordingRunner(respond)
    scheduler = Scheduler(world, runner)
    try:
        scheduler.start(poll_seconds=0.01)
        assert entered.wait(5)
        loop = scheduler._loop
        world.settings.set_settings("agent", {"max_concurrent_turns": 0})
        scheduler.wake()
        loop.join(5)
        assert not loop.is_alive()
        assert len(failures) == 1
        assert "max_concurrent_turns" in str(failures[0].exc_value)
        log = (tmp_path / "logs" / "huddol.log").read_text()
        assert "Scheduler stopped after a runtime failure" in log
        assert "Agent parameter max_concurrent_turns" in log
        room = world.store.create_discussion("waiting", [HUMAN, HELPER])
        world.store.append_message(room.id, HUMAN, "@Helper please help")
        assert world.store.get_member(HELPER).state == "idle"
        assert world.history.runs(HELPER) == ()
        release.set()
        scheduler._threads[MAIN].join(5)
        assert not scheduler._threads[MAIN].is_alive()
        assert world.history.runs(MAIN)[0].status == "failed"
        assert world.history.pause_reason(MAIN) is None
        assert scheduler.agent_status(MAIN)["state"] == "blocked"
        world.settings.set_settings("agent", {"max_concurrent_turns": 1})
        scheduler.wake()
        scheduler.start()
        assert scheduler._loop is loop
        assert not loop.is_alive()
        assert len(runner.requests) == 1
    finally:
        release.set()
        scheduler.stop()
        for handler in handlers:
            logging.getLogger().removeHandler(handler)
            handler.close()


def test_legacy_limits_do_not_control_scheduling(world) -> None:
    mention(world)
    world.settings.set_settings("limits", {"agent_token_limit": 1})
    spend(world, MAIN, 10_000_000)
    scheduler = Scheduler(world, RecordingRunner())
    assert scheduler.token_limit() == 0
    assert scheduler.runnable_agents() == (MAIN,)


def test_the_limit_is_per_agent_not_shared(world) -> None:
    mention(world)
    world.settings.set_settings("agent", {"token_limit": 500})
    spend(world, HELPER, 900)
    scheduler = Scheduler(world, RecordingRunner())
    assert scheduler.over_token_limit(HELPER)
    assert scheduler.runnable_agents() == (MAIN,)


@pytest.mark.parametrize(
    "global_text,member_text",
    [
        (None, None),
        (" \t\n", ""),
        ("全局\n ", "\n 成员😀\t "),
        (None, "member"),
        ("global", None),
    ],
)
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_agents_instructions_are_read_only_at_window_creation(
    world, tmp_path, global_text, member_text, newline
):
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from huddol.adapters.model.prompt import SYSTEM_PROMPT
    from huddol.adapters.model.runner import PydanticModelRunner

    global_path = tmp_path / "library" / "AGENTS.md"
    member_path = world.workspace_tree_for(MAIN).root / "AGENTS.md"
    other_path = world.workspace_tree_for(HELPER).root / "AGENTS.md"
    other_path.write_bytes(b"must not leak")
    global_text = (
        global_text.replace("\n", newline) if global_text is not None else None
    )
    member_text = (
        member_text.replace("\n", newline) if member_text is not None else None
    )
    for path, text in ((global_path, global_text), (member_path, member_text)):
        if text is not None:
            path.write_bytes(text.encode("utf-8"))
    world.settings.set_settings(
        "model",
        {"base_url": "https://example.invalid", "api_key": "unused", "model": "local"},
    )
    seen = []

    def respond(messages, info):
        seen.append(info.model_request_parameters.instruction_parts[0].content)
        return ModelResponse(parts=[TextPart("done")])

    def scheduler():
        return Scheduler(
            world,
            PydanticModelRunner(
                world.settings, build_model=lambda config: FunctionModel(respond)
            ),
        )

    room = mention(world)
    assert scheduler().run_turn(MAIN).status == "completed"
    parts = [
        text for text in (global_text, member_text) if text is not None and text.strip()
    ]
    expected = SYSTEM_PROMPT + ("\n\n" + "\n\n".join(parts) if parts else "")
    assert seen == [expected]
    for path, text in ((global_path, global_text), (member_path, member_text)):
        assert path.exists() == (text is not None)
        if text is not None:
            assert path.read_bytes() == text.encode("utf-8")
        path.write_bytes(b"\xff")
    world.store.append_message(room, HUMAN, "@Main next turn")
    assert scheduler().run_turn(MAIN).status == "completed"
    assert seen == [expected, expected]
    global_path.write_bytes(b"new global")
    new_member_text = f"new member{newline} "
    member_path.write_bytes(new_member_text.encode("utf-8"))
    world.history.reset_window(MAIN, "prepared")
    world.store.append_message(room, HUMAN, "@Main new window")
    assert scheduler().run_turn(MAIN).status == "completed"
    assert seen[-1] == SYSTEM_PROMPT + "\n\nnew global\n\n" + new_member_text


@pytest.mark.parametrize("bad_global", [False, True])
def test_unreadable_agents_file_records_error_before_model(world, tmp_path, bad_global):
    target = (
        tmp_path / "library" if bad_global else world.workspace_tree_for(MAIN).root
    ) / "AGENTS.md"
    target.write_bytes(b"\xff")
    mention(world)
    runner = RecordingRunner()
    record = Scheduler(world, runner).run_turn(MAIN)
    assert record.status == "failed"
    assert str(target) in record.error
    assert "UTF-8" in record.error
    assert world.history.pause_reason(MAIN) is None
    assert world.history.lifecycle(MAIN).error == record.error
    assert runner.requests == []
    assert target.read_bytes() == b"\xff"


def test_process_interruption_before_response_recovers_saved_window_snapshot(
    world, tmp_path
):
    import subprocess
    import textwrap

    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from huddol.adapters.model.prompt import SYSTEM_PROMPT
    from huddol.adapters.model.runner import PydanticModelRunner

    world.settings.set_settings(
        "model",
        {"base_url": "https://example.invalid", "api_key": "unused", "model": "local"},
    )
    child = textwrap.dedent("""
        import os, sys
        from huddol.adapters.sqlite.store import SqliteStore
        from huddol.adapters.sqlite.agent import SqliteAgentStore
        from huddol.adapters.model.runner import PydanticModelRunner
        from huddol.runtime.reminder import TurnRequest
        from pydantic_ai.models.function import FunctionModel
        store = SqliteStore(sys.argv[1])
        history = SqliteAgentStore(store._db)
        run = history.start_run(2, reminded=[])
        request = TurnRequest(agent_id=2, sequence=run.sequence, agent_name="Main", prompt="work", reminder=None,
            history_json="[]", resident="resident", environment=lambda: None, ephemeral=lambda: "",
            persist=lambda raw: history.save_progress(2, run.sequence, raw), agents_instructions="snapshot\\n ")
        def interrupt(messages, info):
            os._exit(77)
        PydanticModelRunner(history, build_model=lambda config: FunctionModel(interrupt)).run(request, None)
    """)
    result = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path / "huddol.sqlite3")],
        check=False,
        timeout=30,
        capture_output=True,
    )
    assert result.returncode == 77, result.stderr.decode()
    world.history.mark_interrupted()
    (tmp_path / "library" / "AGENTS.md").write_bytes(b"\xff")
    seen = []

    def respond(messages, info):
        seen.append(info.model_request_parameters.instruction_parts[0].content)
        return ModelResponse(parts=[TextPart("recovered")])

    mention(world)
    runner = PydanticModelRunner(
        world.settings, build_model=lambda config: FunctionModel(respond)
    )
    record = Scheduler(world, runner).run_turn(MAIN)
    assert record.status == "completed"
    assert seen == [SYSTEM_PROMPT + "\n\nsnapshot\n "]


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_agents_file_changed_during_tool_call_waits_for_next_window(
    world, tmp_path, newline
):
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from huddol.adapters.model.prompt import SYSTEM_PROMPT
    from huddol.adapters.model.runner import PydanticModelRunner

    path = tmp_path / "library" / "AGENTS.md"
    content = f"before{newline} "
    path.write_bytes(content.encode("utf-8"))
    world.settings.set_settings(
        "model",
        {"base_url": "https://example.invalid", "api_key": "unused", "model": "local"},
    )
    room = mention(world)
    seen = []

    def respond(messages, info):
        seen.append(info.model_request_parameters.instruction_parts[0].content)
        if len(seen) == 1:
            path.write_bytes(b"\xff")
            return ModelResponse(
                parts=[ToolCallPart("organization", {"action": "list_members"})]
            )
        return ModelResponse(parts=[TextPart("done")])

    scheduler = Scheduler(
        world,
        PydanticModelRunner(
            world.settings, build_model=lambda config: FunctionModel(respond)
        ),
    )
    assert scheduler.run_turn(MAIN).status == "completed"
    world.store.append_message(room, HUMAN, "@Main again")
    assert scheduler.run_turn(MAIN).status == "completed"
    assert seen == [SYSTEM_PROMPT + "\n\n" + content] * 3
    world.history.reset_window(MAIN, "prepared")
    world.store.append_message(room, HUMAN, "@Main new window")
    record = scheduler.run_turn(MAIN)
    assert record.status == "failed" and str(path) in record.error
    assert len(seen) == 3
