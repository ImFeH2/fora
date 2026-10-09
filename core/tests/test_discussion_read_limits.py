from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from test_protocol import call as protocol_call
from test_protocol import server as protocol_server
from test_tools import HUMAN, MAIN, OTHER, tools_for
from test_tools import world as tools_world

from fora.adapters.model.runner import PydanticModelRunner
from fora.core.errors import DomainError
from fora.runtime.reminder import Reminder, ReminderItem, TurnRequest
from fora.tools.discussion_read import Position, result_length

world = tools_world
server = protocol_server


def follow(actor, request: dict[str, Any]) -> dict[str, Any]:
    assert request["action"] == "read"
    return actor.read_discussion(
        **{key: value for key, value in request.items() if key != "action"}
    )


def bounded(result: dict[str, Any], limit: int, budget: int) -> None:
    assert result_length(result) <= budget
    assert len(ToolReturnPart("discussion", result).model_response_str()) <= budget
    if result["section"] == "messages":
        assert len(result["messages"]) <= limit
        ids = [item["id"] for item in result["messages"]]
        assert ids == sorted(set(ids))


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"limit": None, "max_chars": None},
        {"limit": 7},
        {"message_id": 150},
        {"message_id": 1},
        {"message_id": 301},
        {"message_id": 150, "limit": 1},
    ],
)
def test_defaults_and_anchor_are_bounded(world, params) -> None:
    room = tools_for(world, HUMAN).create_discussion("bounded", [MAIN])["id"]
    for number in range(301):
        world.store.append_message(room, HUMAN, f"message {number}")
    actor = tools_for(world, MAIN)
    result = actor.read_discussion(room, **params)
    bounded(result, params.get("limit") or 20, 16000)
    if params.get("message_id"):
        assert params["message_id"] in [item["id"] for item in result["messages"]]
    else:
        assert result["messages"][-1]["id"] == 301
    assert world.store.acknowledged(room, MAIN) == ()


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"after": 0},
        {"before": 302},
        {"before": 280, "after": 20},
        {"message_id": 150},
    ],
)
@pytest.mark.parametrize("limit,budget", [(7, 16000), (100, 4096)])
def test_all_continuation_branches_preserve_fixed_messages(
    world, params, limit, budget
) -> None:
    room = tools_for(world, HUMAN).create_discussion("pages", [MAIN])["id"]
    originals = {}
    for number in range(301):
        body = f"message {number}: " + '中😀\\\n"' * (2000 if number % 80 == 0 else 3)
        message, _ = world.store.append_message(room, HUMAN, body)
        originals[message.id] = body
    actor = tools_for(world, MAIN)
    first = actor.read_discussion(room, limit=limit, max_chars=budget, **params)
    world.store.append_message(room, HUMAN, "new message outside snapshot")
    queue = [first]
    fragments: dict[int, dict[int, str]] = {}
    visited = set()
    while queue:
        result = queue.pop(0)
        bounded(result, limit, budget)
        assert result["snapshot_max_id"] == 301
        for item in result["messages"]:
            offsets = fragments.setdefault(item["id"], {})
            assert item["body_offset"] not in offsets
            offsets[item["body_offset"]] = item["body"]
        for request in result["continuations"]:
            assert request["cursor"] not in visited
            visited.add(request["cursor"])
            queue.append(follow(actor, request))
    expected = {
        key: value
        for key, value in originals.items()
        if (params.get("after", 0) < key < params.get("before", 302))
    }
    assert set(fragments) == set(expected)
    for message_id, offsets in fragments.items():
        assert (
            "".join(value for _, value in sorted(offsets.items()))
            == expected[message_id]
        )


def test_long_unicode_body_and_metadata_continue_without_premature_read(world) -> None:
    room = tools_for(world, HUMAN).create_discussion("Unicode", [MAIN, OTHER])["id"]
    body = "@Main " + ('中😀e\u0301"\\\n\x00' * 13000)[:99994]
    message, _ = world.store.append_message(room, HUMAN, body)
    actor = tools_for(world, MAIN)
    result = actor.read_discussion(room, message_id=message.id, max_chars=4096)
    pieces = []
    while True:
        bounded(result, 20, 4096)
        item = result["messages"][0]
        assert item["body_offset"] == sum(map(len, pieces))
        pieces.append(item["body"])
        assert item["mentions_info"]["total"] == 1
        assert world.store.acknowledged(room, MAIN) == ()
        if item["body_complete"]:
            assert world.store.watermark(room, MAIN) == message.id
            break
        assert world.store.watermark(room, MAIN) == 0
        result = follow(actor, result["continuations"][-1])
    assert "".join(pieces) == body
    mentions = actor.read_discussion(room, section="mentions", message_id=message.id)
    assert mentions["mentions"] == [{"member_id": MAIN, "position": 0, "length": 5}]


@pytest.mark.parametrize("section", ["members", "awaiting_ack", "acknowledged"])
def test_metadata_and_ack_scope_are_bounded(world, section) -> None:
    ids = [HUMAN, MAIN]
    for number in range(120):
        ids.append(world.store.create_member("agent", f"Reader {number}").id)
    room = world.store.create_discussion("metadata", ids).id
    for number in range(301):
        message, _ = world.store.append_message(room, HUMAN, f"@Main message {number}")
        if number % 2:
            world.store.ack(room, [message.id], MAIN)
    actor = tools_for(world, MAIN)
    initial = actor.read_discussion(room, limit=1)
    assert initial["awaiting_ack"] == [301]
    assert initial["awaiting_ack_count"] == 151
    assert initial["acknowledged_count"] == 150
    watermark = world.store.watermark(room, MAIN)
    result = actor.read_discussion(room, section=section, limit=7, max_chars=2048)
    all_items = []
    while True:
        bounded(result, 7, 2048)
        all_items.extend(result[section])
        assert world.store.watermark(room, MAIN) == watermark
        if result["complete"]:
            break
        result = follow(actor, result["continuations"][0])
    assert len(all_items) == len(
        {item["id"] if isinstance(item, dict) else item for item in all_items}
    )
    assert (
        len(all_items)
        == {"members": 122, "awaiting_ack": 151, "acknowledged": 150}[section]
    )


@pytest.mark.parametrize(
    "params",
    [
        {"limit": True},
        {"limit": 101},
        {"limit": 0},
        {"max_chars": 2047},
        {"max_chars": 64001},
        {"max_chars": 4096.0},
        {"message_id": True},
        {"message_id": "1"},
        {"before": 1, "after": 2},
        {"section": "other"},
        {"section": "mentions"},
        {"cursor": "corrupt"},
        {"cursor": "x" * 2049},
    ],
)
def test_invalid_inputs_preserve_state(world, params) -> None:
    room = world.store.create_discussion("invalid", [HUMAN, MAIN]).id
    world.store.append_message(room, HUMAN, "@Main content")
    actor = tools_for(world, MAIN)
    with pytest.raises(DomainError):
        actor.read_discussion(room, **params)
    assert world.store.watermark(room, MAIN) == 0
    assert world.store.acknowledged(room, MAIN) == ()


def test_cursor_validation_permissions_and_budget_failures(world) -> None:
    room = world.store.create_discussion("cursor", [HUMAN, MAIN]).id
    world.store.append_message(room, HUMAN, "long body " * 5000)
    actor = tools_for(world, MAIN)
    result = actor.read_discussion(room, max_chars=4096)
    request = result["continuations"][0]
    position = Position.decode(request["cursor"], room, "messages")
    for bad in (
        replace(position, o=1000000),
        replace(position, m=1000),
        replace(position, d=room + 1),
        replace(position, v=2),
    ):
        with pytest.raises(DomainError):
            actor.read_discussion(room, cursor=bad.encode())
    with pytest.raises(DomainError):
        actor.read_discussion(room, cursor=request["cursor"], section="members")
    with pytest.raises(DomainError):
        actor.read_discussion(room, cursor=request["cursor"], message_id=1)
    with pytest.raises(DomainError, match="max_chars"):
        actor.read_discussion(
            room, _result_size=lambda result: result_length(result) + 100000
        )
    assert world.store.watermark(room, MAIN) == 0
    world.store.set_archived(room, True)
    archived = follow(actor, request)
    assert archived["archived"]
    world.store.change_discussion_members(room, [MAIN], remove=True)
    with pytest.raises(DomainError) as error:
        follow(actor, request)
    assert error.value.code == "not_a_member"
    assert world.store.watermark(room, MAIN) == 0


def test_permission_removed_after_result_construction_rejects_return(
    world, monkeypatch
) -> None:
    room = world.store.create_discussion("permission", [HUMAN, MAIN]).id
    world.store.append_message(room, HUMAN, "@Main content")
    finish = world.store.finish_read

    def revoke_then_finish(discussion_id, member_id, message_id):
        world.store.change_discussion_members(discussion_id, [member_id], remove=True)
        finish(discussion_id, member_id, message_id)

    monkeypatch.setattr(world.store, "finish_read", revoke_then_finish)
    with pytest.raises(DomainError):
        tools_for(world, MAIN).read_discussion(room)
    assert world.store.watermark(room, MAIN) == 0
    assert world.store.acknowledged(room, MAIN) == ()


def test_read_queries_never_load_all_messages_or_mentions(world, monkeypatch) -> None:
    room = world.store.create_discussion("bounded queries", [HUMAN, MAIN]).id
    for number in range(301):
        world.store.append_message(room, HUMAN, f"@Main record {number}")
    fetched = []
    original = world.store.read_message

    def capture(discussion_id, message_id):
        fetched.append(message_id)
        return original(discussion_id, message_id)

    def forbidden(*args, **kwargs):
        pytest.fail("unbounded legacy read used")

    monkeypatch.setattr(world.store, "read_message", capture)
    monkeypatch.setattr(world.store, "messages", forbidden)
    monkeypatch.setattr(world.store, "mentions_by_message", forbidden)
    monkeypatch.setattr(world.store, "pending", forbidden)
    monkeypatch.setattr(world.store, "list_members", forbidden)
    monkeypatch.setattr(world.store, "get_discussion", forbidden)
    monkeypatch.setattr(world.store, "acknowledged", forbidden)
    result = tools_for(world, MAIN).read_discussion(room, message_id=150, limit=7)
    bounded(result, 7, 16000)
    assert len(fetched) <= 7
    assert min(fetched) >= 147 and max(fetched) <= 153


def test_model_tool_schema_and_next_request_use_bounded_structured_returns(
    world,
) -> None:
    room = world.store.create_discussion("model budget", [HUMAN, MAIN]).id
    body = '中😀"\\\n\x00' * 900
    world.store.append_message(room, HUMAN, body)
    parts = []

    class Settings:
        def get_settings(self, section):
            if section == "model":
                return {
                    "api_type": "openai-chat",
                    "base_url": "https://example.invalid/v1",
                    "api_key": "unused",
                    "model": "test-model",
                }
            return None

    def respond(messages, info):
        returned = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "discussion"
        ]
        if not returned:
            tool = next(
                item for item in info.function_tools if item.name == "discussion"
            )
            assert "max_chars" in tool.parameters_json_schema["properties"]
            assert "section" in tool.parameters_json_schema["properties"]
            assert "cursor" in tool.parameters_json_schema["properties"]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "discussion",
                        {
                            "action": "read",
                            "discussion_id": room,
                            "message_id": 1,
                            "max_chars": 4096,
                        },
                        tool_call_id="initial",
                    )
                ]
            )
        result = returned[-1].content
        assert isinstance(result, dict)
        bounded(result, 20, 4096)
        assert returned[-1].model_response_object() == result
        item = result["messages"][0]
        parts.append(item["body"])
        if item["body_complete"]:
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "discussion",
                    result["continuations"][-1],
                    tool_call_id=f"part-{len(parts)}",
                )
            ]
        )

    reminder = Reminder(
        MAIN, "Main", (ReminderItem(room, "model budget", 1, HUMAN, "You", False),)
    )
    request = TurnRequest(
        agent_id=MAIN,
        sequence=1,
        agent_name="Main",
        prompt=reminder.render(),
        reminder=reminder,
        history_json="[]",
        resident="",
        environment=lambda: "environment",
        ephemeral=lambda: "",
        persist=lambda value: None,
    )
    runner = PydanticModelRunner(
        Settings(), build_model=lambda config: FunctionModel(respond)
    )
    outcome = runner.run(request, tools_for(world, MAIN))
    assert outcome.error is None
    assert "".join(parts) == body
    assert len(parts) > 1
    assert world.store.watermark(room, MAIN) == 1


def test_locked_provider_tool_result_mapping_preserves_budget_and_objects(
    world,
) -> None:
    import httpx
    import httpx2
    from anthropic import AsyncAnthropic
    from openai import AsyncOpenAI
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    room = world.store.create_discussion("SDK", [HUMAN, MAIN]).id
    world.store.append_message(room, HUMAN, '中😀"\\\n\x00' * 3000)
    result = tools_for(world, MAIN).read_discussion(room, max_chars=4096)
    part = ToolReturnPart("discussion", result, tool_call_id="call-sdk")
    expected = part.model_response_str()
    assert len(expected) <= 4096

    def forbidden(request):
        pytest.fail("Provider tests must not access the network")

    async def execute():
        async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as http:
            client = AsyncOpenAI(api_key="unused", http_client=http)
            provider = OpenAIProvider(openai_client=client)
            chat = OpenAIChatModel("gpt-4o", provider=provider)
            mapped = await chat._map_messages(
                [ModelRequest(parts=[part])], ModelRequestParameters()
            )
            assert mapped[0]["content"] == expected
            responses = OpenAIResponsesModel("gpt-4o", provider=provider)
            assert await responses._map_tool_return_output(part) == expected
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(forbidden)
        ) as http:
            anthropic = AnthropicModel(
                "claude-sonnet-4-5",
                provider=AnthropicProvider(
                    anthropic_client=AsyncAnthropic(api_key="unused", http_client=http)
                ),
            )
            _, mapped = await anthropic._map_message(
                [ModelRequest(parts=[part])], ModelRequestParameters(), {}
            )
            assert mapped[0]["content"][0]["content"][0]["text"] == expected
            google = GoogleModel(
                "gemini-2.5-flash",
                provider=GoogleProvider(api_key="unused", http_client=http),
            )
            mapped, extra = await google._map_tool_return(part)
            assert mapped["function_response"]["response"] == result
            assert extra == []

    asyncio.run(execute())


def test_jsonl_dispatcher_uses_budgets_and_strict_inputs(server) -> None:
    dispatcher, output, deps = server
    agent = deps.store.create_member("agent", "Reader")
    room = deps.store.create_discussion("protocol", [HUMAN, agent.id]).id
    deps.store.append_message(room, agent.id, '中文😀"\\\n' * 5000)
    for params in (
        {"message_id": "1"},
        {"message_id": True},
        {"max_chars": 4096.0},
        {"limit": True},
    ):
        response = protocol_call(
            dispatcher, output, "discussion.read", discussion_id=room, **params
        )
        assert "error" in response
        assert deps.store.watermark(room, HUMAN) == 0
    response = protocol_call(
        dispatcher,
        output,
        "discussion.read",
        discussion_id=room,
        message_id=1,
        max_chars=4096,
    )
    result = response["result"]
    bounded(result, 20, 4096)
    assert not result["messages"][0]["body_complete"]
    assert deps.store.watermark(room, HUMAN) == 0
    continuation = result["continuations"][0]
    response = protocol_call(
        dispatcher,
        output,
        "discussion.read",
        **{key: value for key, value in continuation.items() if key != "action"},
    )
    bounded(response["result"], 20, 4096)


def test_attachment_sections_preserve_ids_and_paths(world, tmp_path) -> None:
    from fora.adapters.files.uploads import DirectoryUploads
    from fora.services.uploads import Uploads

    world.uploads = Uploads(world.store, DirectoryUploads(tmp_path / "uploads"))
    human = tools_for(world, HUMAN)
    room = human.create_discussion("attachments", [MAIN])["id"]
    records = []
    for number in range(10):
        upload = human.create_upload(
            room, str(uuid.uuid4()), f"file-{number}.txt", 7, "text/plain"
        )
        record, target = world.uploads.begin(upload["id"], HUMAN)
        with target:
            target.write(b"content")
        world.uploads.complete(record)
        world.uploads.end(record)
        records.append(record.id)
    message = human.send_message(room, "", attachment_ids=records)["id"]
    actor = tools_for(world, MAIN)
    initial = actor.read_discussion(room, limit=1, max_chars=4096)
    item = initial["messages"][0]
    assert item["body"] == "" and item["body_complete"]
    assert item["attachments_info"]["total"] == 10
    assert not item["attachments_info"]["complete"]
    descriptions = list(item["attachments"])
    request = item["attachments_info"]["continuation"]
    while request:
        result = follow(actor, request)
        bounded(result, 1, 4096)
        descriptions.extend(result["attachments"])
        request = result["continuations"][0] if result["continuations"] else None
    assert [item["id"] for item in descriptions] == records
    for item in descriptions:
        assert item["path"] == world.uploads.files.path(item["id"])
    assert world.store.watermark(room, MAIN) == message
    assert world.store.acknowledged(room, MAIN) == ()


def test_partial_message_with_higher_existing_watermark_and_empty_pages(world) -> None:
    room = world.store.create_discussion("existing watermark", [HUMAN, MAIN]).id
    world.store.append_message(room, HUMAN, "long " * 19000)
    world.store.append_message(room, HUMAN, "latest")
    world.store.set_watermark(room, MAIN, 2)
    actor = tools_for(world, MAIN)
    result = actor.read_discussion(room, message_id=1, max_chars=4096)
    assert not result["messages"][0]["body_complete"]
    assert [item["id"] for item in result["messages"]] == [1]
    assert world.store.watermark(room, MAIN) == 2
    for params in ({"after": 999}, {"before": 0}):
        empty = actor.read_discussion(room, **params)
        assert empty["messages"] == []
        assert world.store.watermark(room, MAIN) == 2


def test_nondivisible_identity_and_metadata_item_report_budget(world) -> None:
    room = world.store.create_discussion("large identity", [HUMAN, MAIN]).id
    world.store.append_message(room, HUMAN, "content")
    with world.store._db as db:
        db.execute(
            "UPDATE messages SET sender_name = ? WHERE discussion_id = ?",
            ("sender" * 20000, room),
        )
    with pytest.raises(DomainError) as error:
        tools_for(world, MAIN).read_discussion(room)
    assert error.value.code == "read_budget"
    assert "sender" not in str(error.value)
    assert world.store.watermark(room, MAIN) == 0
    with world.store._db as db:
        db.execute("UPDATE members SET name = ? WHERE id = ?", ("member" * 20000, MAIN))
    members = tools_for(world, MAIN).read_discussion(
        room, section="members", max_chars=64000
    )
    assert members["members"] == [{"id": HUMAN, "name": "You"}]
    with pytest.raises(DomainError) as error:
        follow(tools_for(world, MAIN), members["continuations"][0])
    assert error.value.code == "read_budget"
    assert "membermember" not in str(error.value)


def test_context_uses_adjacent_message_order_with_sparse_ids(world) -> None:
    room = world.store.create_discussion("sparse", [HUMAN, MAIN]).id
    for number in range(9):
        world.store.append_message(room, HUMAN, f"record {number}")
    with world.store._db as db:
        db.execute("UPDATE messages SET id = id * 100 WHERE discussion_id = ?", (room,))
    result = tools_for(world, MAIN).read_discussion(room, message_id=500, limit=7)
    assert [item["id"] for item in result["messages"]] == [
        200,
        300,
        400,
        500,
        600,
        700,
        800,
    ]
    assert len(result["continuations"]) == 2


def test_historical_oversize_body_continues_and_read_watermark_is_monotonic(
    world,
) -> None:
    room = world.store.create_discussion("historical", [HUMAN, MAIN]).id
    world.store.append_message(room, HUMAN, "seed")
    world.store.append_message(room, HUMAN, "latest")
    body = "history\x00😀" * 15000
    with world.store._db as db:
        db.execute(
            "UPDATE messages SET body = ? WHERE discussion_id = ? AND id = 1",
            (body, room),
        )
    actor = tools_for(world, MAIN)
    result = actor.read_discussion(room, message_id=1, max_chars=64000)
    world.store.mark_read(room, MAIN, 2)
    pieces = []
    while True:
        bounded(result, 20, 64000)
        item = result["messages"][0]
        pieces.append(item["body"])
        assert world.store.watermark(room, MAIN) == 2
        if item["body_complete"]:
            break
        continuation = next(
            request
            for request in result["continuations"]
            if Position.decode(request["cursor"], room, "messages").m == 1
        )
        result = follow(actor, continuation)
    assert "".join(pieces) == body
