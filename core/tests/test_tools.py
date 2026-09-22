from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from huddol.adapters.execution.manager import ExecutionManager
from huddol.adapters.files.tree import DirectoryTree
from huddol.adapters.model.runner import PydanticModelRunner
from huddol.adapters.sqlite.agent import SqliteAgentStore
from huddol.adapters.sqlite.store import SqliteStore
from huddol.core.errors import DomainError
from huddol.ports.execution import EditResult, RunResult
from huddol.runtime.reminder import Reminder, ReminderItem, TurnRequest
from huddol.tools import AgentTools, Dependencies, TurnBinding
from huddol.tools.authorize import Actor, Authorizer

HUMAN = 1
MAIN = 2
OTHER = 3


@pytest.fixture
def world(tmp_path: Path):
    store = SqliteStore(tmp_path / "huddol.sqlite3")
    agent_store = SqliteAgentStore(store._db)
    store.create_member("human", "You")
    store.create_member("agent", "Main")
    store.create_member("agent", "Other")

    def agent_directory_for(member_id: int) -> Path:
        path = tmp_path / "agents" / str(member_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    deps = Dependencies(
        store=store,
        history=agent_store,
        settings=agent_store,
        execution=ExecutionManager(enforce=False),
        agent_directory_for=agent_directory_for,
        library_tree=DirectoryTree(tmp_path / "library"),
        workspace_tree_for=lambda member_id: DirectoryTree(
            tmp_path / "agents" / str(member_id) / "workspace"
        ),
    )
    yield deps
    deps.execution.close()
    store.close()


def tools_for(deps: Dependencies, member_id: int, **kwargs) -> AgentTools:
    member = deps.store.get_member(member_id)
    assert member is not None
    return AgentTools(deps, Actor(member_id, member.is_agent), **kwargs)


def test_history_tool_returns_bounded_pages_and_hides_other_members_runs(world) -> None:
    payload = '[{"text":"' + "z" * 3_000_000 + '"}]'
    run = world.history.start_run(MAIN)
    world.history.finish_run(
        MAIN, run.sequence, status="completed", messages_json=payload
    )
    tools = tools_for(world, MAIN)
    tail = tools.read_history(run.sequence)
    assert len(tail["messages"]) == 2048
    assert tail["offset"] == len(payload) - 2048
    assert tail["total_length"] == len(payload)
    assert tail["next_offset"] is None
    first = tools.read_history(run.sequence, 0)
    assert len(first["messages"]) == 2048
    assert first["offset"] == 0
    assert first["next_offset"] == 2048
    second = tools.read_history(run.sequence, first["next_offset"])
    assert second["offset"] == 2048
    assert len(second["messages"]) == 2048
    assert len(json.dumps(first, ensure_ascii=False).encode()) < 32 * 1024
    with pytest.raises(DomainError) as error:
        tools_for(world, OTHER).read_history(run.sequence)
    assert error.value.code == "not_found"
    empty = world.history.start_run(MAIN)
    world.history.finish_run(MAIN, empty.sequence, status="completed", messages_json="")
    empty_result = tools.read_history(empty.sequence)
    assert empty_result["messages"] == ""
    assert empty_result["offset"] == 0
    assert empty_result["total_length"] == 0
    assert empty_result["next_offset"] is None


def test_new_window_runner_can_read_and_continue_large_history(world) -> None:
    payload = '[{"text":"' + "q" * 3_000_000 + '"}]'
    run = world.history.start_run(MAIN)
    world.history.finish_run(
        MAIN, run.sequence, status="completed", messages_json=payload
    )
    tools = tools_for(world, MAIN)
    returned = []

    class Settings:
        def get_settings(self, section):
            if section == "model":
                return {
                    "api_type": "openai-chat",
                    "base_url": "https://example.invalid/v1",
                    "api_key": "unused",
                    "model": "history-test",
                }
            return None

    def respond(messages, info):
        history_returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "history"
        ]
        if history_returns:
            result = history_returns[-1].content
            returned.append(result)
            assert len(result["messages"]) <= 2048
            assert len(json.dumps(result, ensure_ascii=False).encode()) < 32 * 1024
        calls = len(returned)
        if calls == 0:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "history",
                        {"action": "read", "sequence": run.sequence},
                        tool_call_id="tail",
                    )
                ]
            )
        if calls == 1:
            assert returned[0]["offset"] == len(payload) - 2048
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "history",
                        {"action": "read", "sequence": run.sequence, "offset": 0},
                        tool_call_id="first",
                    )
                ]
            )
        if calls == 2:
            assert returned[1]["offset"] == 0
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "history",
                        {"action": "read", "sequence": run.sequence, "offset": 2048},
                        tool_call_id="next",
                    )
                ]
            )
        assert returned[2]["offset"] == 2048
        return ModelResponse(parts=[TextPart("continued")])

    reminder = Reminder(MAIN, "Main", (ReminderItem(1, "test", 1, 1, "You", False),))
    persisted = []
    request = TurnRequest(
        agent_id=MAIN,
        sequence=run.sequence + 1,
        agent_name="Main",
        prompt=reminder.render(),
        reminder=reminder,
        history_json="[]",
        resident="",
        environment=lambda: "environment",
        ephemeral=lambda: "",
        persist=persisted.append,
    )
    runner = PydanticModelRunner(
        Settings(), build_model=lambda config: FunctionModel(respond)
    )
    outcome = runner.run(request, tools)
    assert outcome.error is None
    assert len(returned) == 3
    assert len(persisted) >= 1
    assert len(outcome.messages_json.encode()) < 32 * 1024
    assert payload not in outcome.messages_json
    assert (
        ModelMessagesTypeAdapter.validate_json(outcome.messages_json)[-1]
        .parts[0]
        .content
        == "continued"
    )
    continued = runner.run(
        replace(
            request,
            sequence=request.sequence + 1,
            history_json=outcome.messages_json,
        ),
        tools,
    )
    assert continued.error is None
    assert (
        ModelMessagesTypeAdapter.validate_json(continued.messages_json)[-1]
        .parts[0]
        .content
        == "continued"
    )


@pytest.mark.parametrize("member_id", [MAIN, OTHER])
def test_run_defaults_to_the_agents_directory_on_first_use(
    world, tmp_path: Path, member_id: int
) -> None:
    directory = tmp_path / "agents" / str(member_id)
    assert not directory.exists()
    tools = tools_for(world, member_id)
    for _ in range(2):
        result = tools.run([sys.executable, "-c", "import os; print(os.getcwd())"])
        assert result["exit_code"] == 0
        assert result["stdout"].strip() == str(directory)
    assert directory.is_dir()


def test_run_resolves_relative_cwd_under_the_agents_directory(world) -> None:
    directory = world.agent_directory_for(MAIN) / "nested"
    directory.mkdir()
    result = tools_for(world, MAIN).run(
        [sys.executable, "-c", "import os; print(os.getcwd())"], cwd="nested"
    )
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == str(directory)


def test_run_keeps_absolute_cwd(world, tmp_path: Path) -> None:
    result = tools_for(world, MAIN).run(
        [sys.executable, "-c", "import os; print(os.getcwd())"], cwd=str(tmp_path)
    )
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == str(tmp_path)
    assert (tmp_path / "agents" / str(MAIN) / "workspace").is_dir()


@pytest.mark.parametrize(
    "path", ["/home/agent/project", r"C:\project", r"\\server\share\project"]
)
def test_tools_forward_absolute_paths_in_both_platform_formats(
    world, tmp_path: Path, monkeypatch, path: str
) -> None:
    tools = tools_for(world, MAIN)
    calls = []

    def run(argv, *, cwd, timeout, write_directories):
        calls.append((argv, cwd, timeout))
        return RunResult(0, "", "", False)

    def edit(path, old_text, new_text, *, replace_all, write_directories, create):
        calls.append((path, old_text, new_text, replace_all, create))
        return EditResult(path, "", 0 if create else 1)

    environment = world.execution.snapshot()
    monkeypatch.setattr(environment, "run", run)
    monkeypatch.setattr(environment, "edit", edit)
    monkeypatch.setattr(world.execution, "snapshot", lambda: environment)
    tools.run(["pwd"], cwd=path, timeout=7)
    tools.edit(path, "before", "after", replace_all=True)
    tools.edit(path, "", "complete body", create=True)
    assert calls == [
        (["pwd"], path, 7),
        (path, "before", "after", True, False),
        (path, "", "complete body", False, True),
    ]
    assert (tmp_path / "agents" / str(MAIN) / "workspace").is_dir()


def test_relative_edit_resolves_under_the_agents_directory_and_is_not_writable(
    world, tmp_path: Path, monkeypatch
) -> None:
    directory = tmp_path / "agents" / str(MAIN)
    tools = tools_for(world, MAIN)
    paths = []
    environment = world.execution.snapshot()
    original = environment.edit

    def edit(path, *args, **kwargs):
        paths.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(environment, "edit", edit)
    monkeypatch.setattr(world.execution, "snapshot", lambda: environment)
    assert not directory.exists()
    with pytest.raises(DomainError) as error:
        tools.edit("file.txt", "before", "after")
    assert paths == [str(directory / "file.txt")]
    assert error.value.code == "not_writable"
    assert directory.is_dir()


def test_existing_tools_follow_directory_changes(
    world, tmp_path: Path, monkeypatch
) -> None:
    manager = world.execution
    target = tmp_path / "file.txt"
    target.write_text("before", encoding="utf-8")
    tools = tools_for(world, MAIN)
    command = [sys.executable, "-c", "print('native')"]
    assert tools.run(command)["stdout"].strip() == "native"
    with pytest.raises(DomainError, match="outside"):
        tools.edit(str(target), "before", "denied")
    manager.configure({"write_directories": [str(tmp_path)]}, lambda values: None)
    tools.edit(str(target), "before", "native")
    assert tools.run(command)["stdout"].strip() == "native"
    manager.configure({"write_directories": []}, lambda values: None)
    with pytest.raises(DomainError, match="outside"):
        tools.edit(str(target), "native", "denied")
    manager.configure({"write_directories": [str(tmp_path)]}, lambda values: None)
    tools.edit(str(target), "native", "after")
    assert target.read_text(encoding="utf-8") == "after"


def test_edit_keeps_absolute_paths(world, tmp_path: Path) -> None:
    directory = tmp_path / "writable"
    directory.mkdir()
    target = directory / "file.txt"
    target.write_text("before", encoding="utf-8")
    world.execution.configure(
        {"write_directories": [str(directory)]}, lambda values: None
    )
    result = tools_for(world, MAIN).edit(str(target), "before", "after")
    assert result["path"] == str(target)
    assert target.read_text(encoding="utf-8") == "after"
    assert (tmp_path / "agents" / str(MAIN) / "workspace").is_dir()


def test_edit_tool_creates_file_then_uses_existing_edit_behavior(
    world, tmp_path: Path
) -> None:
    directory = tmp_path / "writable"
    directory.mkdir()
    target = directory / "new.txt"
    world.execution.configure(
        {"write_directories": [str(directory)]}, lambda values: None
    )
    tools = tools_for(world, MAIN, turn=TurnBinding(MAIN, 1))
    created = tools.edit(str(target), "", "before", create=True)
    assert created["path"] == str(target)
    assert created["replacements"] == 0
    assert created["created"] is True
    assert "+before" in created["diff"]
    assert target.read_text(encoding="utf-8") == "before"
    empty = directory / "empty.txt"
    empty_created = tools.edit(str(empty), "", "", create=True)
    assert empty_created["created"] is True
    assert empty_created["diff"] == ""
    edited = tools.edit(str(target), "before", "after")
    assert edited["replacements"] == 1
    assert edited["created"] is False
    recorded = world.history.effects(MAIN, sequences=[1])
    assert ["created" in effect.summary for effect in recorded] == [True, True, False]
    assert "1 replaced" in recorded[-1].summary
    assert target.read_text(encoding="utf-8") == "after"


@pytest.mark.parametrize("actor_id", [HUMAN, MAIN])
def test_discussion_list_orders_last_message_and_limits_after_filtering(
    world, monkeypatch, actor_id
) -> None:
    human = tools_for(world, HUMAN)
    empty = [human.create_discussion("Empty", [MAIN])["id"] for _ in range(2)]
    rooms = []

    def send(room, stamp):
        monkeypatch.setattr("huddol.adapters.sqlite.store.now", lambda: stamp)
        world.store.append_message(room, HUMAN, "@You @Main needle")

    for day in range(1, 26):
        room = human.create_discussion(f"Room {day}", [MAIN])["id"]
        rooms.append(room)
        send(room, f"2026-01-{day:02}T01:00:00Z")
    send(rooms[0], "2026-01-29T00:00:00Z")
    send(rooms[0], "2026-01-01T20:00:00-05:00")
    archived = human.create_discussion("Archived", [MAIN])["id"]
    send(archived, "2026-01-30T00:00:00Z")
    human.archive_discussion(archived)
    hidden = world.store.create_discussion("Hidden", [OTHER])
    monkeypatch.setattr(
        "huddol.adapters.sqlite.store.now", lambda: "2026-01-31T00:00:00Z"
    )
    world.store.append_message(hidden.id, OTHER, "needle")
    actor = tools_for(world, actor_id)
    pending = world.store.pending(actor_id)
    watermarks = [world.store.watermark(room, actor_id) for room in rooms]
    expected = list(reversed(rooms[2:])) + rooms[:2] + empty
    assert [item["id"] for item in actor.list_discussions()] == expected
    assert [item["id"] for item in actor.list_discussions(limit=3)] == expected[:3]
    assert [item["id"] for item in actor.list_discussions(limit=100)] == expected
    assert [item["id"] for item in actor.list_discussions(limit=10**30)] == expected
    assert [item["id"] for item in actor.list_discussions(True, limit=1)] == [archived]
    assert len(actor.search_messages("needle")) == 28
    assert world.store.pending(actor_id) == pending
    assert [world.store.watermark(room, actor_id) for room in rooms] == watermarks


@pytest.mark.parametrize("limit", [0, -1, True, "2", 1.5])
def test_discussion_list_rejects_invalid_limits(world, limit) -> None:
    with pytest.raises(DomainError) as error:
        tools_for(world, MAIN).list_discussions(limit=limit)
    assert error.value.code == "invalid_pagination"


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"limit": 2}, [5, 6]),
        ({"before": 5, "limit": 2}, [3, 4]),
        ({"after": 2, "limit": 2}, [3, 4]),
        ({"after": 0, "limit": 2}, [1, 2]),
        ({"after": 2, "before": 5}, [3, 4]),
        ({"after": 1, "before": 6, "limit": 2}, [2, 3]),
        ({"before": 1}, []),
        ({"after": 6}, []),
        ({"message_id": 3, "limit": 1}, [1, 2, 3, 4, 5, 6]),
    ],
)
def test_reading_pages_keeps_order_and_exclusive_boundaries(
    world, params, expected
) -> None:
    human = tools_for(world, HUMAN)
    room = human.create_discussion("pages", [MAIN])
    for number in range(6):
        human.send_message(room["id"], f"Message {number}")
    page = tools_for(world, MAIN).read_discussion(room["id"], **params)
    assert [item["id"] for item in page["messages"]] == expected
    assert page["total_messages"] == 6
    assert world.store.watermark(room["id"], MAIN) == (max(expected) if expected else 0)


@pytest.mark.parametrize(
    "params",
    [
        {"before": -1},
        {"after": True},
        {"before": "2"},
        {"limit": 0},
        {"limit": -1},
        {"limit": 1.5},
        {"message_id": 1, "before": 2},
        {"message_id": 1, "after": 0},
    ],
)
def test_invalid_pagination_does_not_advance_read_state(world, params) -> None:
    room = tools_for(world, HUMAN).create_discussion("pages", [MAIN])
    tools_for(world, HUMAN).send_message(room["id"], "@Main read this")
    with pytest.raises(DomainError) as error:
        tools_for(world, MAIN).read_discussion(room["id"], **params)
    assert error.value.code == "invalid_pagination"
    assert world.store.watermark(room["id"], MAIN) == 0
    assert len(world.store.pending(MAIN)) == 1


@pytest.mark.parametrize("actor_id", [HUMAN, MAIN])
def test_discussion_management_preserves_history_and_pending(world, actor_id) -> None:
    human = tools_for(world, HUMAN)
    room = human.create_discussion("handoff", [MAIN, OTHER])["id"]
    untouched = human.create_discussion("separate", [MAIN])["id"]
    for body in ("@You @Main first", "@You @Main second"):
        tools_for(world, OTHER).send_message(room, body)
    actor = tools_for(world, actor_id)
    actor.read_discussion(room)
    actor.ack(room, [1])
    actor.remove_members(room, [actor_id])
    assert world.store.message_count(room) == 2
    assert world.store.acknowledged(room, actor_id) == (1,)
    assert world.store.pending(actor_id) == ()
    with pytest.raises(DomainError, match="do not belong"):
        actor.read_discussion(room)
    actor.add_members(room, [actor_id, actor_id])
    assert actor.read_discussion(room)["awaiting_ack"] == [2]
    actor.archive_discussion(room)
    assert world.store.pending(actor_id) == ()
    assert room not in [item["id"] for item in actor.list_discussions()]
    assert len(actor.search_messages("first", sender_id=OTHER, discussion_id=room)) == 1
    assert actor.search_messages("first", sender_id=HUMAN, discussion_id=room) == []
    assert actor.search_messages("first", discussion_id=untouched) == []
    actor.archive_discussion(room, False)
    restored = actor.read_discussion(room)
    assert (restored["awaiting_ack"], restored["archived"]) == ([2], False)
    assert not hasattr(actor, "delete_discussion")
    assert world.store.message_count(room) == 2
    assert world.store.acknowledged(room, actor_id) == (1,)
    assert world.store.get_discussion(untouched) is not None


@pytest.mark.parametrize(
    ("method", "params", "capability"),
    [
        ("add_members", {"member_ids": [OTHER]}, "discussion.add_members"),
        ("remove_members", {"member_ids": [MAIN]}, "discussion.remove_members"),
        ("set_discussion_members", {"member_ids": [HUMAN]}, "discussion.set_members"),
        ("archive_discussion", {}, "discussion.archive"),
        ("archive_discussion", {"archived": False}, "discussion.unarchive"),
    ],
)
def test_discussion_management_is_authorized_before_mutation(
    world, method, params, capability
) -> None:
    room = tools_for(world, HUMAN).create_discussion("guarded", [MAIN])["id"]
    original = world.store.get_discussion(room)
    seen = []

    def deny(actor, name, target):
        seen.append((actor.member_id, name, target))
        return "deny"

    emitted = []
    tools = tools_for(
        world,
        MAIN,
        authorizer=Authorizer(deny),
        on_change=lambda name, payload: emitted.append(name),
    )
    with pytest.raises(DomainError) as error:
        getattr(tools, method)(room, **params)
    assert error.value.code == "not_permitted"
    assert seen == [(MAIN, capability, room)]
    assert emitted == []
    assert world.store.get_discussion(room) == original


@pytest.mark.parametrize("method", ["add_members", "set_discussion_members"])
def test_invalid_members_are_rejected_without_partial_changes(world, method) -> None:
    room = tools_for(world, HUMAN).create_discussion("guarded", [MAIN])["id"]
    original = world.store.get_discussion(room)
    actor = tools_for(world, MAIN)
    with pytest.raises(DomainError, match="Unknown Members"):
        getattr(actor, method)(room, [OTHER, 999])
    assert world.store.get_discussion(room) == original
    world.store.delete_member(OTHER)
    with pytest.raises(DomainError, match="Unknown Members"):
        getattr(actor, method)(room, [OTHER])
    assert world.store.get_discussion(room) == original


def test_concurrent_member_changes_preserve_independent_updates(world) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    human = tools_for(world, HUMAN)
    room = human.create_discussion("concurrent", [MAIN])["id"]
    barrier = Barrier(2)

    def add():
        barrier.wait()
        tools_for(world, MAIN).add_members(room, [OTHER])

    def remove():
        barrier.wait()
        human.remove_members(room, [MAIN])

    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = [pool.submit(add), pool.submit(remove)]
        for worker in workers:
            worker.result(timeout=5)
    assert world.store.get_discussion(room).member_ids == frozenset([HUMAN, OTHER])


def test_creating_a_discussion_always_includes_the_creator(world) -> None:
    tools = tools_for(world, HUMAN)
    room = tools.create_discussion("plan the rewrite", [MAIN])
    assert room["member_ids"] == [HUMAN, MAIN]


def test_a_discussion_needs_at_least_one_other_member(world) -> None:
    with pytest.raises(DomainError) as error:
        tools_for(world, HUMAN).create_discussion("alone", [HUMAN])
    assert error.value.code == "needs_members"


def test_non_members_cannot_read_or_send(world) -> None:
    room = tools_for(world, HUMAN).create_discussion("private", [MAIN])
    outsider = tools_for(world, OTHER)
    with pytest.raises(DomainError) as error:
        outsider.read_discussion(room["id"])
    assert error.value.code == "not_a_member"
    with pytest.raises(DomainError):
        outsider.send_message(room["id"], "hello")


def test_reading_a_mention_returns_surrounding_context(world) -> None:
    human = tools_for(world, HUMAN)
    room = human.create_discussion("context", [MAIN])
    human.send_message(room["id"], "background one")
    human.send_message(room["id"], "background two")
    human.send_message(room["id"], "@Main please look")

    result = tools_for(world, MAIN).read_discussion(room["id"], message_id=3)
    assert [item["id"] for item in result["messages"]] == [1, 2, 3]
    assert result["total_messages"] == 3


@pytest.mark.parametrize(
    ("watermark", "expected"),
    [(0, [3, 4, 5]), (2, [1, 2, 3, 4, 5])],
)
def test_reading_a_mention_preserves_context_boundaries(
    world, watermark: int, expected: list[int]
) -> None:
    human = tools_for(world, HUMAN)
    room = human.create_discussion("context", [MAIN, OTHER])
    for body in (
        "Earlier background",
        "@Other handle another item",
        "Background for this item",
        "@Main please look",
        "More details",
    ):
        human.send_message(room["id"], body)
    tools_for(world, OTHER).send_message(room["id"], "A different sender")
    world.store.set_watermark(room["id"], MAIN, watermark)

    result = tools_for(world, MAIN).read_discussion(room["id"], message_id=4)

    assert result["read_through"] == watermark
    assert [item["id"] for item in result["messages"]] == expected


def test_reading_advances_the_watermark_and_clears_unread(world) -> None:
    human = tools_for(world, HUMAN)
    room = human.create_discussion("unread", [MAIN])
    human.send_message(room["id"], "@Main hello")

    agent = tools_for(world, MAIN)
    assert agent.list_discussions()[0]["unread"] == 1
    agent.read_discussion(room["id"], message_id=1)
    assert agent.list_discussions()[0]["unread"] == 0


def test_ack_requires_the_message_to_have_been_read(world) -> None:
    human = tools_for(world, HUMAN)
    room = human.create_discussion("ack", [MAIN])
    human.send_message(room["id"], "@Main do this")

    agent = tools_for(world, MAIN)
    with pytest.raises(DomainError) as error:
        agent.ack(room["id"], [1])
    assert error.value.code == "not_read"

    agent.read_discussion(room["id"], message_id=1)
    assert agent.ack(room["id"], [1])["acked"] == 1
    assert world.store.pending(MAIN) == ()


@pytest.mark.parametrize("member_id", [HUMAN, MAIN])
def test_members_only_revoke_their_own_acknowledgements(world, member_id: int) -> None:
    room = tools_for(world, HUMAN).create_discussion("review", [MAIN, OTHER])
    tools_for(world, OTHER).send_message(room["id"], "@You @Main please review")
    for actor in (HUMAN, MAIN):
        tools = tools_for(world, actor)
        tools.read_discussion(room["id"])
        tools.ack(room["id"], [1])
    tools = tools_for(world, member_id)
    assert tools.read_discussion(room["id"])["acknowledged"] == [1]

    assert tools.revoke_ack(room["id"], [1])["revoked"] == 1

    own = tools.read_discussion(room["id"])
    assert own["acknowledged"] == []
    assert own["awaiting_ack"] == [1]
    other = MAIN if member_id == HUMAN else HUMAN
    assert world.store.acknowledged(room["id"], other) == (1,)
    assert tools.revoke_ack(room["id"], [1])["revoked"] == 0
    world.store.set_discussion_members(room["id"], [other, OTHER])
    with pytest.raises(DomainError, match="do not belong"):
        tools.revoke_ack(room["id"], [1])


def test_sending_marks_your_own_message_as_read(world) -> None:
    human = tools_for(world, HUMAN)
    room = human.create_discussion("self", [MAIN])
    human.send_message(room["id"], "mine")
    assert human.list_discussions()[0]["unread"] == 0


def test_search_only_returns_discussions_you_belong_to(world) -> None:
    human = tools_for(world, HUMAN)
    mine = human.create_discussion("mine", [MAIN])
    human.send_message(mine["id"], "findme here")

    assert len(tools_for(world, MAIN).search_messages("findme")) == 1
    assert tools_for(world, OTHER).search_messages("findme") == []


def test_duplicate_agent_names_are_rejected(world) -> None:
    with pytest.raises(DomainError) as error:
        tools_for(world, HUMAN).create_agent("main")
    assert error.value.code == "duplicate_name"


def test_running_agents_cannot_be_deleted(world) -> None:
    world.store.set_agent_state(MAIN, "running")
    with pytest.raises(DomainError) as error:
        tools_for(world, HUMAN).delete_agent(MAIN)
    assert error.value.code == "agent_running"

    world.store.set_agent_state(MAIN, "paused")
    assert tools_for(world, HUMAN).delete_agent(MAIN)["deleted"] is True


def test_deleting_an_agent_preserves_all_its_data(world) -> None:
    human = tools_for(world, HUMAN)
    agent = tools_for(world, MAIN)
    rooms = [human.create_discussion(topic, [MAIN])["id"] for topic in ("A", "B")]
    agent.send_message(rooms[0], "@You keep this")
    human.send_message(rooms[0], "@Main review")
    agent.read_discussion(rooms[0])
    agent.ack(rooms[0], [2])
    world.workspace_tree_for(MAIN).write("notes.md", "Keep notes")
    world.history.start_run(MAIN)
    window = world.history.reset_window(MAIN, "prepared")
    runs = world.history.runs(MAIN)
    files = {
        path.relative_to(world.workspace_tree_for(MAIN).root): path.read_bytes()
        for path in world.workspace_tree_for(MAIN).root.rglob("*")
        if path.is_file()
    }
    messages = world.store.messages(rooms[0])

    assert human.delete_agent(MAIN) == {"id": MAIN, "deleted": True}

    assert world.history.runs(MAIN) == runs
    assert world.history.window(MAIN) == window
    assert {
        path.relative_to(world.workspace_tree_for(MAIN).root): path.read_bytes()
        for path in world.workspace_tree_for(MAIN).root.rglob("*")
        if path.is_file()
    } == files
    assert world.store.messages(rooms[0]) == messages
    assert world.store.acknowledged(rooms[0], MAIN) == (2,)
    assert all(
        MAIN not in world.store.get_discussion(room).member_ids for room in rooms
    )
    assert MAIN not in [item["id"] for item in human.list_members()]
    assert MAIN in [item["id"] for item in human.list_members(include_deleted=True)]
    assert world.store.get_member(MAIN).deleted


@pytest.mark.parametrize("params", [{}, {"limit": 1}, {"message_id": 1}])
def test_read_and_search_return_stored_mentions_not_current_names(
    world, params
) -> None:
    human = tools_for(world, HUMAN)
    room = human.create_discussion("Mentions", [MAIN])["id"]
    body = "Hi @Other and @mAiN and @Main"
    sent = human.send_message(room, body)
    assert sent["mentioned"] == [MAIN]
    expected = [{"member_id": MAIN, "position": 14, "length": 5}]
    human.add_members(room, [OTHER])
    human.rename_member(MAIN, "Renamed")
    human.remove_members(room, [MAIN])
    assert human.read_discussion(room, **params)["messages"][0]["mentions"] == expected
    assert human.search_messages("Hi")[0]["mentions"] == expected
    human.send_message(room, "@Renamed nobody")
    assert human.read_discussion(room, limit=1)["messages"][0]["mentions"] == []
    assert human.search_messages("nobody")[0]["mentions"] == []


@pytest.mark.parametrize("path", ["", ".", "./", "/", " / ", "///", " \t\n", "/ \t/\n"])
def test_tree_tools_accept_root_paths(world, path) -> None:
    tools = tools_for(world, HUMAN)
    tools.write_library("a/b.md", "shared")
    world.workspace_tree_for(MAIN).write("notes.md", "private")
    assert tools.list_library(path=path) == tools.list_library()
    assert tools.list_workspace(path=path, agent_id=MAIN) == tools.list_workspace(
        agent_id=MAIN
    )


def test_library_conflict_returns_the_current_content(world) -> None:
    author = tools_for(world, HUMAN)
    author.write_library("shared.md", "first")
    result = author.write_library("shared.md", "second", expected_hash="0" * 16)
    assert result["conflict"] is True
    assert result["current_content"] == "first"


def test_human_library_write_emits_one_update_and_conflicts_emit_none(world) -> None:
    emitted = []
    author = tools_for(
        world, HUMAN, on_change=lambda name, payload: emitted.append((name, payload))
    )
    written = author.write_library("shared.md", "first")
    assert emitted == [("library.updated", written)]
    assert emitted[0][1]["path"] == "shared.md"

    conflict = author.write_library("shared.md", "second", expected_hash="0" * 16)
    assert conflict["conflict"] is True
    assert len(emitted) == 1


def test_workspace_is_private_to_each_agent(world) -> None:
    world.workspace_tree_for(MAIN).write("notes.md", "mine")
    assert [item["path"] for item in tools_for(world, OTHER).list_workspace()] == [
        "MEMORY.md"
    ]
    assert len(tools_for(world, MAIN).list_workspace()) == 2


def test_library_is_shared_across_members(world) -> None:
    tools_for(world, HUMAN).write_library("shared.md", "for everyone")
    assert (
        tools_for(world, OTHER).read_library("shared.md")["content"] == "for everyone"
    )


def test_authorization_hook_can_deny_a_capability(world) -> None:
    def deny_sending(actor, capability, target):
        return "deny" if capability == "discussion.send" else "allow"

    tools = tools_for(world, HUMAN, authorizer=Authorizer(deny_sending))
    room = tools.create_discussion("guarded", [MAIN])
    with pytest.raises(DomainError) as error:
        tools.send_message(room["id"], "blocked")
    assert error.value.code == "not_permitted"


def test_read_reports_what_is_waiting_for_you(world) -> None:
    human = tools_for(world, HUMAN)
    room = human.create_discussion("waiting", [MAIN])
    human.send_message(room["id"], "background")
    world.store.append_message(room["id"], MAIN, "@You please confirm")

    result = human.read_discussion(room["id"])
    assert result["awaiting_ack"] == [2]

    human.ack(room["id"], [2])
    assert human.read_discussion(room["id"])["awaiting_ack"] == []


def test_read_reports_the_position_you_had_reached_before_reading(world) -> None:
    human = tools_for(world, HUMAN)
    room = human.create_discussion("watermark", [MAIN])
    world.store.append_message(room["id"], MAIN, "one")
    world.store.append_message(room["id"], MAIN, "two")

    first = human.read_discussion(room["id"])
    assert first["read_through"] == 0

    world.store.append_message(room["id"], MAIN, "three")
    second = human.read_discussion(room["id"])
    assert second["read_through"] == 2
    assert second["awaiting_ack"] == []


def test_awaiting_ack_only_covers_the_discussion_you_read(world) -> None:
    human = tools_for(world, HUMAN)
    first = human.create_discussion("first", [MAIN])
    second = human.create_discussion("second", [MAIN])
    world.store.append_message(first["id"], MAIN, "@You here")
    world.store.append_message(second["id"], MAIN, "@You and here")

    assert human.read_discussion(first["id"])["awaiting_ack"] == [1]
    assert human.read_discussion(second["id"])["awaiting_ack"] == [1]


def test_a_turn_records_what_it_produced(world) -> None:
    tools = tools_for(world, MAIN, turn=TurnBinding(MAIN, 1))
    discussion = tools.create_discussion("Work", [OTHER])["id"]
    sent = tools.send_message(discussion, "Starting now")["id"]
    tools.run([sys.executable, "-c", "print('hi')"])
    world.workspace_tree_for(MAIN).write("notes.md", "content")
    tools.edit("workspace/notes.md", "content", "updated")

    recorded = world.history.effects(MAIN, sequences=[1])
    assert [item.tool for item in recorded] == ["send", "run", "edit"]
    assert str(sent) in recorded[0].summary


def test_tools_outside_a_turn_record_nothing(world) -> None:
    tools = tools_for(world, MAIN)
    discussion = tools.create_discussion("Work", [OTHER])["id"]
    tools.send_message(discussion, "Starting now")
    assert world.history.effects(MAIN) == ()


def test_an_acknowledging_turn_is_not_counted_as_productive(world) -> None:
    from huddol.core.turn import is_productive

    author = tools_for(world, OTHER)
    discussion = author.create_discussion("Work", [MAIN])["id"]
    message = author.send_message(discussion, "@Main please look")["id"]

    tools = tools_for(world, MAIN, turn=TurnBinding(MAIN, 1))
    tools.read_discussion(discussion, message)
    tools.ack(discussion, [message])

    recorded = world.history.effects(MAIN, sequences=[1])
    assert [item.tool for item in recorded] == ["ack"]
    assert not is_productive([item.tool for item in recorded])


@pytest.mark.parametrize("exit_code", [0, 3])
def test_run_records_effects_and_emits_each_changed_library_file(
    world, exit_code
) -> None:
    emitted = []
    tools = tools_for(
        world,
        MAIN,
        turn=TurnBinding(MAIN, 1),
        on_change=lambda name, payload: emitted.append((name, payload)),
    )
    for name, content in (
        ("changed.txt", "before"),
        ("removed.txt", "gone"),
        ("unchanged.txt", "same"),
    ):
        world.library_tree.write(name, content)
    result = tools.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import sys; "
                "Path('added.txt').write_text('new'); "
                "Path('changed.txt').write_text('after'); "
                "Path('removed.txt').unlink(); "
                "Path('binary').write_bytes(bytes([255])); "
                f"sys.exit({exit_code})"
            ),
        ],
        cwd=str(world.library_tree.root),
    )
    assert result == {
        "exit_code": exit_code,
        "stdout": "",
        "stderr": "",
        "truncated": False,
    }
    assert emitted == [
        (
            "library.updated",
            {"path": "added.txt", "hash": tools.read_library("added.txt")["hash"]},
        ),
        ("library.updated", {"path": "binary", "hash": None}),
        (
            "library.updated",
            {"path": "changed.txt", "hash": tools.read_library("changed.txt")["hash"]},
        ),
        ("library.updated", {"path": "removed.txt", "deleted": True}),
    ]
    (effect,) = world.history.effects(MAIN, sequences=[1])
    assert effect.tool == "run" and f"exited {exit_code}" in effect.summary
    assert world.execution.snapshot().write_directories == ()


@pytest.mark.parametrize("member_id", [MAIN, OTHER])
def test_run_and_edit_add_existing_implicit_roots_to_current_configuration(
    world, tmp_path, monkeypatch, member_id
) -> None:
    tools = tools_for(world, member_id)
    workspace = tmp_path / "agents" / str(member_id) / "workspace"
    library = world.library_tree.root
    library.rmdir()
    assert not workspace.exists() and not library.exists()
    configured = tmp_path / "configured"
    configured.mkdir()
    environment = world.execution.snapshot()
    calls = []

    def run(argv, **params):
        assert workspace.is_dir() and library.is_dir()
        calls.append(("run", argv, params))
        return RunResult(0, "", "", False)

    def edit(path, old_text, new_text, **params):
        assert workspace.is_dir() and library.is_dir()
        calls.append(("edit", path, params))
        return EditResult(path, "", 1)

    monkeypatch.setattr(environment, "run", run)
    monkeypatch.setattr(environment, "edit", edit)
    monkeypatch.setattr(world.execution, "snapshot", lambda: environment)
    for roots in ([str(configured)], []):
        world.execution.configure({"write_directories": roots}, lambda values: None)
        tools.run(["command"], timeout=7)
        tools.edit("workspace/MEMORY.md", "old", "new", replace_all=True)
        expected = [*roots, str(workspace), str(library)]
        assert calls[-2:] == [
            (
                "run",
                ["command"],
                {
                    "cwd": str(workspace.parent),
                    "timeout": 7,
                    "write_directories": expected,
                },
            ),
            (
                "edit",
                str(workspace / "MEMORY.md"),
                {
                    "replace_all": True,
                    "write_directories": expected,
                    "create": False,
                },
            ),
        ]
        assert environment.write_directories == tuple(roots)


@pytest.mark.parametrize("operation", ["run", "edit"])
def test_execution_emits_partial_library_changes_on_error(
    world, monkeypatch, operation
) -> None:
    emitted = []
    tools = tools_for(
        world, MAIN, on_change=lambda name, payload: emitted.append((name, payload))
    )
    environment = world.execution.snapshot()

    def fail(*args, **params):
        world.library_tree.write("partial.txt", "partial")
        raise DomainError("timeout", "Command timed out")

    monkeypatch.setattr(environment, operation, fail)
    monkeypatch.setattr(world.execution, "snapshot", lambda: environment)
    with pytest.raises(DomainError, match="timed out"):
        if operation == "run":
            tools.run(["command"])
        else:
            tools.edit(str(world.library_tree.root / "partial.txt"), "old", "new")
    assert emitted == [
        (
            "library.updated",
            {"path": "partial.txt", "hash": tools.read_library("partial.txt")["hash"]},
        ),
    ]


def test_edit_emits_library_changes_but_workspace_run_and_edit_do_not(world) -> None:
    emitted = []
    tools = tools_for(
        world, MAIN, on_change=lambda name, payload: emitted.append((name, payload))
    )
    world.library_tree.write("note.txt", "before")
    tools.edit(str(world.library_tree.root / "note.txt"), "before", "after")
    assert emitted == [
        (
            "library.updated",
            {"path": "note.txt", "hash": tools.read_library("note.txt")["hash"]},
        )
    ]
    emitted.clear()
    result = tools.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('workspace/note.md').write_text('private')",
        ]
    )
    assert result["exit_code"] == 0
    tools.edit("workspace/note.md", "private", "updated")
    assert tools.read_workspace("note.md")["content"] == "updated"
    assert emitted == []


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux filesystem sandbox"
)
def test_agent_run_writes_only_configured_and_implicit_roots(world, tmp_path) -> None:
    configured = tmp_path / "configured"
    configured.mkdir()
    other_workspace = world.workspace_tree_for(OTHER).root
    world.execution.close()
    world.execution = ExecutionManager(
        settings={"write_directories": [str(configured)]}
    )
    library = world.library_tree.root
    result = tools_for(world, MAIN).run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import sys; "
                "[Path(path).write_text('allowed') for path in sys.argv[1:]]"
            ),
            "workspace/note.md",
            str(library / "shared.txt"),
            str(configured / "work.txt"),
        ]
    )
    assert result["exit_code"] == 0, result["stderr"]
    for root in (world.workspace_tree_for(MAIN).root, library, configured):
        assert any(path.read_text() == "allowed" for path in root.iterdir())
    for path in (
        world.agent_directory_for(MAIN) / "outside.txt",
        other_workspace / "private.md",
    ):
        denied = tools_for(world, MAIN).run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('denied')",
                str(path),
            ]
        )
        assert denied["exit_code"] != 0 and not path.exists()
    assert world.execution.snapshot().write_directories == (str(configured),)


def test_human_library_mutations_emit_path_deltas(world) -> None:
    emitted = []
    tools = tools_for(
        world,
        HUMAN,
        on_change=lambda name, payload: emitted.append((name, payload)),
    )
    assert tools.mkdir_library("notes") == {"path": "notes", "hash": None}
    written = tools.write_library("notes/file.txt", "before")
    result = tools.edit_library("notes/file.txt", "before", "after")
    assert "+after" in result["diff"]
    tools.move_library("notes", "renamed")
    tools.delete_library("renamed")
    assert emitted == [
        ("library.updated", {"path": "notes", "hash": None}),
        ("library.updated", {"path": "notes/file.txt", "hash": written["hash"]}),
        ("library.updated", {"path": "notes/file.txt", "hash": result["hash"]}),
        ("library.updated", {"path": "notes", "deleted": True}),
        ("library.updated", {"path": "renamed", "hash": None}),
        ("library.updated", {"path": "renamed", "deleted": True}),
    ]
    assert world.history.effects(MAIN) == ()


@pytest.mark.parametrize("content", [b"text", b"\xff"], ids=["text", "binary"])
def test_human_file_moves_preserve_update_hashes(world, content) -> None:
    emitted = []
    tools = tools_for(
        world,
        HUMAN,
        on_change=lambda name, payload: emitted.append((name, payload)),
    )
    (world.library_tree.root / ".source").write_bytes(content)
    moved = tools.move_library(".source", ".destination")
    digest = tools.read_library(".destination")["hash"] if content == b"text" else None
    assert moved == {"path": ".destination", "hash": digest}
    assert emitted == [
        ("library.updated", {"path": ".source", "deleted": True}),
        ("library.updated", moved),
    ]
    assert "hash" not in tools.list_library()[0]


def test_humans_can_read_other_agents_workspace_but_agents_cannot(world) -> None:
    tools = tools_for(world, MAIN)
    tree = world.workspace_tree_for(MAIN)
    tree.write("notes/note.md", "updated")
    human = tools_for(world, HUMAN)
    assert [entry["path"] for entry in human.list_workspace(agent_id=MAIN)] == [
        "MEMORY.md",
        "notes",
        "notes/note.md",
    ]
    assert human.read_workspace("notes/note.md", agent_id=MAIN)["content"] == "updated"
    assert tools.read_workspace("notes/note.md", agent_id=MAIN)["content"] == "updated"
    other = tools_for(world, OTHER)
    for operation in (
        lambda: other.list_workspace(agent_id=MAIN),
        lambda: other.read_workspace("notes/note.md", agent_id=MAIN),
    ):
        with pytest.raises(DomainError) as error:
            operation()
        assert (
            error.value.code == "not_allowed"
            and str(error.value) == "Workspace is private"
        )
    tree.delete("notes")
    assert [entry["path"] for entry in human.list_workspace(agent_id=MAIN)] == [
        "MEMORY.md"
    ]


@pytest.mark.parametrize(
    "method,args,capability",
    [
        ("edit_library", ("doc.txt", "old", "new"), "library.edit"),
        ("mkdir_library", ("folder",), "library.mkdir"),
        ("run", (["echo"],), "run"),
        ("edit", ("workspace/doc.md", "old", "new"), "edit"),
    ],
)
def test_tree_capabilities_are_checked_before_changes(
    world, method, args, capability
) -> None:
    calls = []

    def deny(actor, name, target):
        calls.append(name)
        return "deny"

    tools = tools_for(world, MAIN, authorizer=Authorizer(deny))
    with pytest.raises(DomainError) as error:
        getattr(tools, method)(*args)
    assert error.value.code == "not_permitted"
    assert calls == [capability]
    assert world.library_tree.list() == ()


@pytest.mark.parametrize("params", [{"entry": True}, {"after": 0}, {"before": 5}, {}])
def test_ui_pages_do_not_read_or_ack(world, params) -> None:
    human = tools_for(world, HUMAN)
    agent = tools_for(world, MAIN)
    room = human.create_discussion("page", [MAIN])["id"]
    for _ in range(8):
        agent.send_message(room, "@You page")
    page = human.discussion_page(room, limit=2, **params)
    assert len(page["messages"]) == 2
    assert len(page["awaiting_ack"]) == 2
    assert page["pending_count"] == 8
    assert page["latest_id"] == 8
    assert page["read_through"] == world.store.watermark(room, HUMAN) == 0
    assert "total_messages" not in page
    assert len(world.store.pending(HUMAN)) == 8
    if params.get("entry"):
        assert page["first_unread_id"] == 1
        assert page["previous_sender_id"] is None
        assert not page["has_before"]
    else:
        assert "metadata" not in page
    assert human.mark_read(room, 3)["read_through"] == 3
    assert human.mark_read(room, 1)["read_through"] == 3


def test_ui_entry_includes_own_unread_messages_and_send_does_not_advance(world) -> None:
    human = tools_for(world, HUMAN)
    agent = tools_for(world, MAIN)
    room = human.create_discussion("page", [MAIN])["id"]
    human.send_message(room, "own", mark_read=False)
    assert human.discussion_page(room, entry=True)["first_unread_id"] == 1
    agent.send_message(room, "@You new")
    human.send_message(room, "own newer", mark_read=False)
    page = human.discussion_page(room, entry=True, limit=1)
    assert page["first_unread_id"] == 1
    assert [m["id"] for m in page["messages"]] == [1]
    assert page["previous_sender_id"] is None
    assert not page["has_before"] and page["has_after"]
    assert world.store.watermark(room, HUMAN) == 0
    with pytest.raises(DomainError, match="before acknowledging"):
        human.ack(room, [2])
    assert human.mark_read(room, 2)["read_through"] == 2
    assert human.ack(room, [2])["acked"] == 1
    human.send_message(room, "old default")
    assert world.store.watermark(room, HUMAN) == 4


def test_ui_entry_pages_own_messages_after_the_read_boundary(world) -> None:
    human = tools_for(world, HUMAN)
    room = human.create_discussion("own pages", [MAIN])["id"]
    human.send_message(room, "read")
    for index in range(120):
        human.send_message(room, f"own {index}", mark_read=False)
    page = human.discussion_page(room, entry=True, limit=50)
    assert page["first_unread_id"] == 2
    assert page["read_through"] == world.store.watermark(room, HUMAN) == 1
    assert [message["id"] for message in page["messages"]] == list(range(1, 51))
    assert page["previous_sender_id"] is None
    assert not page["has_before"] and page["has_after"]
    assert page["pending_count"] == 0
    assert world.store.unread_counts(HUMAN).get(room, 0) == 0


def test_ui_entry_uses_latest_read_boundary_with_earlier_pending(world) -> None:
    human = tools_for(world, HUMAN)
    agent = tools_for(world, MAIN)
    room = human.create_discussion("pending", [MAIN])["id"]
    for _ in range(4):
        agent.send_message(room, "@You pending")
    human.mark_read(room, 3)
    human.mark_read(room, 1)
    page = human.discussion_page(room, entry=True, limit=2)
    assert page["read_through"] == 3
    assert page["first_unread_id"] == 4
    assert [message["id"] for message in page["messages"]] == [3, 4]
    assert page["awaiting_ack"] == [3, 4]
    assert page["pending_count"] == 4
    assert world.store.watermark(room, HUMAN) == 3
    human.mark_read(room, 4)
    human.send_message(room, "own next", mark_read=False)
    entered = human.discussion_page(room, entry=True, limit=2)
    assert entered["read_through"] == 4
    assert entered["first_unread_id"] == 5
    assert [message["id"] for message in entered["messages"]] == [4, 5]
    assert entered["awaiting_ack"] == [4]
    assert entered["pending_count"] == 4


def test_ui_entry_context_keeps_id_gaps_and_page_cursors_contiguous(world) -> None:
    human = tools_for(world, HUMAN)
    agent = tools_for(world, MAIN)
    room = human.create_discussion("entry gap", [MAIN])["id"]
    for index in range(7):
        agent.send_message(room, f"message {index}")
    with world.store._db:
        world.store._db.execute(
            "DELETE FROM messages WHERE discussion_id = ? AND id = 2", (room,)
        )
    human.mark_read(room, 1)
    one = human.discussion_page(room, entry=True, limit=1)
    assert one["first_unread_id"] == 3
    assert [message["id"] for message in one["messages"]] == [3]
    assert one["read_through"] == 1
    page = human.discussion_page(room, entry=True, limit=5)
    assert page["first_unread_id"] == 3
    assert [message["id"] for message in page["messages"]] == [1, 3, 4, 5, 6]
    assert page["previous_sender_id"] is None
    assert not page["has_before"] and page["has_after"]
    after = human.discussion_page(room, after=6, limit=3)
    assert [message["id"] for message in after["messages"]] == [7]
    before = human.discussion_page(room, before=7, limit=5)
    assert [message["id"] for message in before["messages"]] == [1, 3, 4, 5, 6]
    assert world.store.watermark(room, HUMAN) == 1


def test_ui_entry_returns_short_page_when_newer_messages_are_insufficient(
    world,
) -> None:
    human = tools_for(world, HUMAN)
    agent = tools_for(world, MAIN)
    room = human.create_discussion("short context", [MAIN])["id"]
    for _ in range(3):
        agent.send_message(room, "unread")
    human.mark_read(room, 2)
    page = human.discussion_page(room, entry=True, limit=5)
    assert page["first_unread_id"] == 3
    assert [message["id"] for message in page["messages"]] == [1, 2, 3]
    assert page["has_before"] is False
    assert page["has_after"] is False
    assert world.store.watermark(room, HUMAN) == 2


@pytest.mark.parametrize("count", [0, 4])
def test_ui_entry_without_unread_returns_latest_page(world, count) -> None:
    human = tools_for(world, HUMAN)
    room = human.create_discussion("read", [MAIN])["id"]
    for _ in range(count):
        human.send_message(room, "read")
    page = human.discussion_page(room, entry=True, limit=2)
    assert page["first_unread_id"] is None
    assert page["read_through"] == world.store.watermark(room, HUMAN) == count
    assert [message["id"] for message in page["messages"]] == ([3, 4] if count else [])
    assert page["has_before"] is bool(count)
    assert not page["has_after"]


def test_ui_bulk_handles_transaction_pending_up_to_chosen_boundary(world) -> None:
    human = tools_for(world, HUMAN)
    agent = tools_for(world, MAIN)
    room = human.create_discussion("bulk", [MAIN])["id"]
    for _ in range(4):
        agent.send_message(room, "@You pending")
    human.mark_read(room, 1)
    human.ack(room, [1])
    human.revoke_ack(room, [1])
    result = human.ack_pending(room, 3)
    assert result == {"acked": 3, "read_through": 3, "pending_count": 1}
    assert human.ack_pending(room, 3) == {
        "acked": 0,
        "read_through": 3,
        "pending_count": 1,
    }
    page = human.discussion_page(room, after=0, limit=2)
    assert page["acknowledged"] == [1, 2]
    assert page["awaiting_ack"] == []
    human.archive_discussion(room)
    assert human.ack_pending(room, 4) == {
        "acked": 0,
        "read_through": 3,
        "pending_count": 0,
    }
    human.archive_discussion(room, False)
    assert human.discussion_page(room)["pending_count"] == 1


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 101},
        {"limit": 0},
        {"limit": True},
        {"entry": True, "after": 0},
        {"after": -1},
        {"before": "2"},
        {"before": 2, "after": 3},
        {"metadata": 1},
    ],
)
def test_ui_page_rejects_invalid_bounds_without_reading(world, params) -> None:
    human = tools_for(world, HUMAN)
    room = human.create_discussion("bad bounds", [MAIN])["id"]
    with pytest.raises(DomainError):
        human.discussion_page(room, **params)
    assert world.store.watermark(room, HUMAN) == 0


def test_ui_operations_require_human_membership_and_valid_message(world) -> None:
    human = tools_for(world, HUMAN)
    agent = tools_for(world, MAIN)
    room = agent.create_discussion("private", [OTHER])["id"]
    agent.send_message(room, "body")
    for action in (
        lambda: human.discussion_page(room),
        lambda: human.mark_read(room, 1),
        lambda: human.ack_pending(room, 1),
    ):
        with pytest.raises(DomainError) as error:
            action()
        assert error.value.code == "not_a_member"
    for action in (
        lambda: agent.discussion_page(room),
        lambda: agent.mark_read(room, 1),
        lambda: agent.ack_pending(room, 1),
        lambda: agent.send_message(room, "body", mark_read=False),
    ):
        with pytest.raises(DomainError) as error:
            action()
        assert error.value.code == "not_permitted"
    agent.add_members(room, [HUMAN])
    for message in (True, -1, 0, 3):
        with pytest.raises(DomainError):
            human.mark_read(room, message)
        with pytest.raises(DomainError):
            human.ack_pending(room, message)
    assert world.store.watermark(room, HUMAN) == 0
    assert human.discussion_page(room)["messages"][0]["id"] == 1


def test_attachment_tools_check_membership_and_preserve_original(
    world, tmp_path
) -> None:
    import hashlib
    import uuid
    from io import BytesIO

    from PIL import Image

    from huddol.adapters.files.uploads import DirectoryUploads
    from huddol.services.uploads import Uploads

    world.uploads = Uploads(world.store, DirectoryUploads(tmp_path / "uploads"))
    human = tools_for(world, HUMAN)
    agent = tools_for(world, MAIN)
    room = human.create_discussion("Images", [MAIN])["id"]
    source = Path(__file__).parents[2] / "app/icons/icon.ico"
    with Image.open(source) as image:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
    data = buffer.getvalue()
    upload = human.create_upload(
        room, str(uuid.uuid4()), "picture.png", len(data), "image/png"
    )
    record, target = world.uploads.begin(upload["id"], HUMAN)
    with target:
        target.write(data)
    world.uploads.complete(record)
    world.uploads.end(record)
    with pytest.raises(DomainError, match="Attachment not found"):
        agent.view_attachment(room, 1, record.id)
    send_id = str(uuid.uuid4())
    sent = human.send_message(
        room, "@Main image", attachment_ids=[record.id], client_message_id=send_id
    )
    assert sent["mentioned"] == [MAIN]
    assert (
        human.send_message(
            room, "@Main image", attachment_ids=[record.id], client_message_id=send_id
        )["id"]
        == sent["id"]
    )
    assert len(world.store.pending(MAIN)) == 1
    description = agent.read_discussion(room)["messages"][0]["attachments"][0]
    assert "data" not in description
    assert Path(description["path"]).is_absolute()
    assert Path(description["path"]).read_bytes() == data
    assert "path" not in human.read_discussion(room)["messages"][0]["attachments"][0]
    assert agent.search_messages("image")[0]["attachments"][0] == description
    result = agent.view_attachment(room, sent["id"], record.id)
    assert result.image.data == data
    assert result.attachment.sha256 == hashlib.sha256(data).hexdigest()
    for actor, message in ((tools_for(world, OTHER), sent["id"]), (agent, 999)):
        with pytest.raises(DomainError):
            actor.view_attachment(room, message, record.id)
    denied = tools_for(
        world,
        MAIN,
        authorizer=Authorizer(
            lambda actor, capability, target: (
                "deny" if capability == "discussion.read" else "allow"
            )
        ),
    )
    with pytest.raises(DomainError, match="not permitted"):
        denied.view_attachment(room, sent["id"], record.id)
    human.remove_members(room, [MAIN])
    with pytest.raises(DomainError, match="do not belong"):
        agent.view_attachment(room, sent["id"], record.id)
    human.add_members(room, [MAIN])
    human.archive_discussion(room)
    assert agent.view_attachment(room, sent["id"], record.id).image.data == data
    Path(description["path"]).write_bytes(data[::-1])
    with pytest.raises(DomainError, match="content has changed"):
        agent.view_attachment(room, sent["id"], record.id)
    assert agent.read_discussion(room)["messages"][0]["attachments"][0] == description


@pytest.mark.parametrize("fail", [False, True])
def test_runner_persists_attachment_results_on_success_and_failure(
    world, tmp_path, monkeypatch, fail
) -> None:
    import uuid
    from dataclasses import replace
    from io import BytesIO

    from PIL import Image
    from pydantic_ai import BinaryContent, ModelMessagesTypeAdapter, models
    from pydantic_ai.messages import (
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import FunctionModel
    from test_runner import FakeSettings, model_values, request

    from huddol.adapters.files.uploads import DirectoryUploads
    from huddol.adapters.model.runner import PydanticModelRunner
    from huddol.services.uploads import Uploads

    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    world.uploads = Uploads(world.store, DirectoryUploads(tmp_path / "uploads"))
    human = tools_for(world, HUMAN)
    agent = tools_for(world, MAIN)
    room = human.create_discussion("Image history", [MAIN])["id"]
    with Image.open(Path(__file__).parents[2] / "app/icons/icon.ico") as source:
        buffer = BytesIO()
        source.convert("RGB").save(buffer, format="PNG")
    data = buffer.getvalue()
    upload = human.create_upload(
        room, str(uuid.uuid4()), "picture.png", len(data), "image/png"
    )
    record, target = world.uploads.begin(upload["id"], HUMAN)
    with target:
        target.write(data)
    world.uploads.complete(record)
    world.uploads.end(record)
    sent = human.send_message(room, "", attachment_ids=[record.id])
    settings = FakeSettings()
    settings.set_settings("model", model_values("local"))
    run = world.history.start_run(MAIN)
    snapshots = []

    def persist(raw):
        world.history.save_progress(MAIN, run.sequence, raw)
        snapshots.append(raw)

    def image_parts(messages):
        return [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "view_attachment"
        ]

    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "view_attachment",
                        {
                            "discussion_id": room,
                            "message_id": sent["id"],
                            "attachment_id": record.id,
                        },
                        "image-call",
                    )
                ]
            )
        parts = image_parts(messages)
        assert len(parts) == 1
        assert parts[0].tool_call_id == "image-call"
        description, image = parts[0].content
        assert description["attachment_id"] == record.id
        assert isinstance(image, BinaryContent) and image.data == data
        if calls == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "organization", {"action": "list_members"}, "members-call"
                    )
                ]
            )
        if fail:
            raise RuntimeError("Local test failure after image result")
        return ModelResponse(parts=[TextPart("Image received")])

    runner = PydanticModelRunner(
        settings, build_model=lambda config: FunctionModel(respond)
    )
    outcome = runner.run(
        replace(request(), sequence=run.sequence, persist=persist), agent
    )
    assert (outcome.error is not None) == fail
    assert any(
        image_parts(ModelMessagesTypeAdapter.validate_json(raw)) for raw in snapshots
    )
    world.history.finish_run(
        MAIN,
        run.sequence,
        status="failed" if fail else "completed",
        messages_json=outcome.messages_json,
        error=outcome.error,
    )
    restored = SqliteStore(tmp_path / "huddol.sqlite3")
    try:
        raw = SqliteAgentStore(restored._db).latest_messages(MAIN)
        parts = image_parts(ModelMessagesTypeAdapter.validate_json(raw))
        assert len(parts) == 1 and parts[0].tool_call_id == "image-call"
        description, image = parts[0].content
        assert description["attachment_id"] == record.id
        assert isinstance(image, BinaryContent) and image.data == data
    finally:
        restored.close()
