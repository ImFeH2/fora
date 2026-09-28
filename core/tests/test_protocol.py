from __future__ import annotations

import asyncio
import base64
import io
import json
import sqlite3
import threading
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self, cast

import pytest
from aiohttp import ClientSession, WSMsgType, WSServerHandshakeError
from pydantic_ai import BinaryContent, ModelMessagesTypeAdapter
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from websockets.sync.client import connect

from huddol.adapters.execution.manager import ExecutionManager
from huddol.adapters.files.tree import DirectoryTree
from huddol.adapters.files.uploads import DirectoryUploads, decode_image
from huddol.adapters.jsonl.api import HUMAN_ID, Api
from huddol.adapters.jsonl.protocol import Dispatcher, parse, wait_for_shutdown
from huddol.adapters.model.config import ModelCatalog
from huddol.adapters.model.runner import PydanticModelRunner
from huddol.adapters.sqlite.agent import SqliteAgentStore
from huddol.adapters.sqlite.store import SqliteStore
from huddol.adapters.voice.config import VoiceConfig
from huddol.adapters.voice.endpoint import VoiceEndpoint
from huddol.adapters.voice.remote import RemoteRecording
from huddol.adapters.websocket.server import WebServer
from huddol.core.errors import DomainError
from huddol.core.parameters import AgentParameters
from huddol.runtime.scheduler import Scheduler
from huddol.services.uploads import Uploads
from huddol.tools import Dependencies
from huddol.tools.authorize import Authorizer


class Capture:
    def __init__(self) -> None:
        self._frames: list[dict[str, Any]] = []

    def __call__(self, payload: dict[str, Any]) -> None:
        self._frames.append(json.loads(json.dumps(payload)))

    def frames(self) -> list[dict[str, Any]]:
        return list(self._frames)


class Probe:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []

    def list_models(
        self, values: dict[str, Any], stored: dict[str, Any] | None
    ) -> dict[str, Any]:
        self.calls.append(("list", values, stored))
        return {"models": ["b", "a"]}

    def test_model(
        self, values: dict[str, Any], stored: dict[str, Any] | None
    ) -> dict[str, Any]:
        self.calls.append(("test", values, stored))
        return {"ok": True, "latency_ms": 12, "reply": "OK"}


@pytest.fixture
def server(tmp_path: Path):
    store = SqliteStore(tmp_path / "huddol.sqlite3")
    agent_store = SqliteAgentStore(store._db)
    store.create_member("human", "You")

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
            settings={"directories": {"native": [str(tmp_path)]}},
            enforce=False,
        ),
        library_tree=DirectoryTree(tmp_path / "library"),
        workspace_tree_for=lambda member_id: DirectoryTree(
            tmp_path / "agents" / str(member_id) / "workspace"
        ),
    )
    output = Capture()
    dispatcher = Dispatcher()
    dispatcher.attach(output)
    scheduler = Scheduler(
        deps, PydanticModelRunner(agent_store), on_event=dispatcher.emit
    )
    probe = Probe()
    Api(
        scheduler,
        dispatcher,
        list_models=probe.list_models,
        test_model=probe.test_model,
    )
    deps.__dict__["probe"] = probe
    deps.__dict__["scheduler"] = scheduler
    yield dispatcher, output, deps
    store.close()


def call(dispatcher: Dispatcher, output: Capture, method: str, **params: Any) -> Any:
    before = len(output.frames())
    request = parse(json.dumps({"id": 99, "method": method, "params": params}))
    assert request is not None
    dispatcher.handle(request, output)
    frames = output.frames()[before:]
    responses = [item for item in frames if item.get("type") == "response"]
    assert responses, f"no response for {method}"
    return responses[-1]


@pytest.mark.parametrize(
    "reuse_id,cancel", [(False, False), (True, False), (False, True)]
)
def test_send_receipt_before_disconnected_worker_commits(
    server, monkeypatch, reuse_id: bool, cancel: bool
) -> None:
    dispatcher, _, deps = server
    store = deps.store
    room = store.create_discussion("Send recovery", [HUMAN_ID])
    original = store.submit_message
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    first_id = str(uuid.uuid4())
    second_id = first_id if reuse_id else str(uuid.uuid4())
    first = True

    def paused(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            entered.set()
            assert release.wait(10)
            try:
                return original(*args, **kwargs)
            finally:
                completed.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "submit_message", paused)
    token = str(uuid.uuid4())
    web = WebServer(dispatcher, token, None)
    web.start()
    url = f"ws://127.0.0.1:{web.port}/ws?token={token}"

    def request(connection, request_id, method, **params):
        connection.send(
            json.dumps({"id": request_id, "method": method, "params": params})
        )
        while True:
            frame = json.loads(connection.recv(timeout=5))
            if frame.get("type") == "response" and frame.get("id") == request_id:
                return frame

    try:
        with connect(url, close_timeout=0.1) as old:
            old.send(
                json.dumps(
                    {
                        "id": 1,
                        "method": "discussion.send",
                        "params": {
                            "discussion_id": room.id,
                            "body": "Pending message",
                            "client_message_id": first_id,
                        },
                    }
                )
            )
            assert entered.wait(5)
        with connect(url) as current:
            status = request(
                current,
                2,
                "discussion.send_status",
                discussion_id=room.id,
                client_message_id=first_id,
            )
            assert status["result"] == {"state": "unknown", "message": None}
            if cancel:
                cancelled = request(
                    current,
                    4,
                    "discussion.cancel_send",
                    discussion_id=room.id,
                    client_message_id=first_id,
                )
                assert cancelled["result"] == {"state": "cancelled", "message": None}
                assert store.message_count(room.id) == 0
            sent = request(
                current,
                3,
                "discussion.send",
                discussion_id=room.id,
                body="Pending message",
                client_message_id=second_id,
            )
            assert "result" in sent
            assert store.message_count(room.id) == 1
            release.set()
            assert completed.wait(5)
            expected = 1 if reuse_id or cancel else 2
            assert store.message_count(room.id) == expected
            assert [message.body for message in store.messages(room.id)] == [
                "Pending message"
            ] * expected
            final = request(
                current,
                5,
                "discussion.send_status",
                discussion_id=room.id,
                client_message_id=first_id,
            )
            assert final["result"]["state"] == ("cancelled" if cancel else "sent")
    finally:
        release.set()
        web.stop()


@pytest.mark.parametrize("during_validation", [False, True])
def test_cancelled_send_after_attachment_cleanup(
    server, tmp_path: Path, monkeypatch, during_validation: bool
) -> None:
    dispatcher, output, deps = server
    store = deps.store
    reader = store.create_member("agent", "Reader")
    room = store.create_discussion("Files", [HUMAN_ID, reader.id])
    files = DirectoryUploads(tmp_path / "uploads")
    uploads = Uploads(store, files)
    deps.uploads = uploads
    data = b"A real attachment"
    record = uploads.create(
        room.id, HUMAN_ID, str(uuid.uuid4()), "file.txt", len(data), "text/plain"
    )
    active, target = uploads.begin(record.id, HUMAN_ID)
    with target:
        target.write(data)
    uploads.complete(active)
    uploads.end(active)
    attempt = str(uuid.uuid4())

    def cancel():
        response = call(
            dispatcher,
            output,
            "discussion.cancel_send",
            discussion_id=room.id,
            client_message_id=attempt,
        )
        assert response["result"] == {"state": "cancelled", "message": None}
        uploads.cancel(record.id, HUMAN_ID)

    if during_validation:

        def status(*args):
            cancel()
            raise DomainError("attachment_missing", "Attachment removed")

        monkeypatch.setattr(uploads, "status", status)
    else:
        cancel()
    result = call(
        dispatcher,
        output,
        "discussion.send",
        discussion_id=room.id,
        body="@Reader",
        attachment_ids=[record.id],
        client_message_id=attempt,
    )
    assert result["error"]["code"] == "send_cancelled"
    assert store.message_count(room.id) == 0
    assert store.pending(reader.id) == ()
    assert store.watermark(room.id, HUMAN_ID) == 0
    assert store.get_upload(record.id, HUMAN_ID).state == "expired"
    assert not list((tmp_path / "uploads").rglob("*.part"))
    assert store.send_outcome(room.id, HUMAN_ID, attempt) == ("cancelled", None)


def test_cancel_response_loss_and_cleanup_failure(
    server, tmp_path: Path, monkeypatch
) -> None:
    dispatcher, output, deps = server
    store = deps.store
    room = store.create_discussion("Cancel", [HUMAN_ID])
    files = DirectoryUploads(tmp_path / "uploads")
    uploads = Uploads(store, files)
    deps.uploads = uploads
    record = uploads.create(
        room.id, HUMAN_ID, str(uuid.uuid4()), "file.txt", 4, "text/plain"
    )
    attempt = str(uuid.uuid4())
    request = parse(
        json.dumps(
            {
                "id": 7,
                "method": "discussion.cancel_send",
                "params": {"discussion_id": room.id, "client_message_id": attempt},
            }
        )
    )
    assert request is not None
    dispatcher.handle(request, lambda response: None)
    repeated = call(
        dispatcher,
        output,
        "discussion.cancel_send",
        discussion_id=room.id,
        client_message_id=attempt,
    )
    assert repeated["result"] == {"state": "cancelled", "message": None}
    original = files.discard

    def fail(upload_id):
        raise OSError("Cannot remove file")

    monkeypatch.setattr(files, "discard", fail)
    with pytest.raises(OSError, match="Cannot remove"):
        uploads.cancel(record.id, HUMAN_ID)
    assert store.send_outcome(room.id, HUMAN_ID, attempt) == ("cancelled", None)
    assert store.get_upload(record.id, HUMAN_ID).state == "deleting"
    monkeypatch.setattr(files, "discard", original)
    uploads.cancel(record.id, HUMAN_ID)
    assert store.get_upload(record.id, HUMAN_ID).state == "expired"
    assert store.message_count(room.id) == 0


def test_voice_settings_keep_key_private_and_validate_updates(server) -> None:
    dispatcher, output, deps = server
    defaults = call(dispatcher, output, "settings.get", section="voice")
    assert defaults["result"]["model"] == "gpt-live-transcribe"
    assert "mode" not in defaults["result"]
    saved = call(
        dispatcher,
        output,
        "settings.update",
        section="voice",
        values={
            "model": "gpt-live-transcribe",
            "api_key": "test-placeholder",
        },
    )
    assert saved["result"]["api_key_set"] is True
    assert "mode" not in saved["result"]
    assert "api_key" not in saved["result"]
    changed = call(
        dispatcher,
        output,
        "settings.update",
        section="voice",
        values={"model": "another-model"},
    )
    assert changed["result"]["model"] == "another-model"
    assert deps.settings.get_settings("voice")["api_key"] == "test-placeholder"
    failed = call(
        dispatcher,
        output,
        "settings.update",
        section="voice",
        values={"address": "http://example.invalid"},
    )
    assert failed["error"]["code"] == "voice_config"
    restored = call(dispatcher, output, "settings.get", section="voice")
    assert restored["result"]["address"].startswith("wss://")
    assert restored["result"]["model"] == "another-model"
    assert "mode" not in restored["result"]
    assert "test-placeholder" not in json.dumps(output.frames())
    deps.settings.set_settings("voice", {"mode": "local", "model": ""})
    migrated = call(dispatcher, output, "settings.get", section="voice")
    assert migrated["result"]["model"] == "gpt-live-transcribe"
    assert "mode" not in migrated["result"]
    assert "mode" not in deps.settings.get_settings("voice")
    rejected_mode = call(
        dispatcher,
        output,
        "settings.update",
        section="voice",
        values={"mode": "local"},
    )
    assert rejected_mode["error"]["code"] == "voice_config"
    rejected_model = call(
        dispatcher,
        output,
        "settings.update",
        section="voice",
        values={"model": " "},
    )
    assert rejected_model["error"]["code"] == "voice_config"


def test_voice_recording_requires_a_saved_api_key() -> None:
    events: list[dict[str, Any]] = []

    class Connection:
        closed = False

        async def __aiter__(self):
            yield SimpleNamespace(
                type=WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "type": "start",
                        "sample_rate": 48000,
                        "channels": 1,
                        "format": "f32le",
                    }
                ),
            )

        async def send_json(self, event: dict[str, Any]) -> None:
            events.append(event)

    async def scenario() -> None:
        endpoint = VoiceEndpoint(lambda: {"model": "gpt-live-transcribe"})
        await endpoint.handle(cast(Any, Connection()))

    asyncio.run(scenario())
    assert events == [
        {
            "type": "error",
            "code": "voice_config",
            "message": "Save an API key before testing transcription",
        }
    ]


def test_info_get_returns_fixed_runtime_information(server) -> None:
    dispatcher, output, _ = server
    first = call(dispatcher, output, "info.get")["result"]
    second = call(dispatcher, output, "info.get")["result"]

    assert first == second
    assert set(first) == {"version", "started_at"}
    assert first["version"]
    assert datetime.fromisoformat(first["started_at"]).tzinfo == UTC


def test_bad_json_produces_an_error_event_not_a_crash(server) -> None:
    dispatcher, output, _ = server
    dispatcher.receive("{not json}", output)
    assert output.frames()[-1]["code"] == "invalid_frame"


def test_internal_methods_are_refused_from_outside(server) -> None:
    dispatcher, output, _ = server
    dispatcher.receive(json.dumps({"id": 1, "method": "system.secret"}), output)
    assert output.frames()[-1]["error"]["code"] == "internal_method"


def test_shutdown_is_refused_over_a_connection(server) -> None:
    dispatcher, output, _ = server
    dispatcher.receive(json.dumps({"id": 1, "method": "system.shutdown"}), output)
    assert output.frames()[-1]["error"]["code"] == "internal_method"


def test_shutdown_on_stdin_stops_the_loop() -> None:
    assert (
        wait_for_shutdown(io.StringIO(json.dumps({"method": "system.shutdown"}) + "\n"))
        == "shutdown"
    )


def test_other_stdin_input_is_ignored_until_eof() -> None:
    assert (
        wait_for_shutdown(io.StringIO('garbage\n{"id":1,"method":"ping"}\n')) == "eof"
    )


def test_responses_go_to_the_requesting_connection_only(server) -> None:
    dispatcher, output, _ = server
    other = Capture()
    dispatcher.attach(other)
    dispatcher.receive(json.dumps({"id": 1, "method": "ping"}), other)
    assert [frame["type"] for frame in other.frames()] == ["response"]
    assert output.frames() == []


def test_events_reach_every_connection(server) -> None:
    dispatcher, output, _ = server
    other = Capture()
    dispatcher.attach(other)
    call(dispatcher, output, "organization.create_agent", name="Main")
    assert [frame["type"] for frame in other.frames()] == ["member.created"]
    assert [frame["type"] for frame in output.frames()] == [
        "member.created",
        "response",
    ]
    dispatcher.detach(other)
    dispatcher.emit("noop")
    assert len(other.frames()) == 1


def test_unknown_methods_return_a_named_error(server) -> None:
    dispatcher, output, _ = server
    assert call(dispatcher, output, "nope.at.all")["error"]["code"] == "unknown_method"


def test_discussion_delete_is_not_registered(server) -> None:
    dispatcher, output, deps = server
    agent = deps.store.create_member("agent", "Main")
    room = deps.store.create_discussion("Keep", [HUMAN_ID, agent.id])
    response = call(dispatcher, output, "discussion.delete", discussion_id=room.id)
    assert response["error"]["code"] == "unknown_method"
    assert deps.store.get_discussion(room.id) == room
    assert [frame["type"] for frame in output.frames()] == ["response"]


def test_domain_errors_become_structured_responses(server) -> None:
    dispatcher, output, _ = server
    response = call(dispatcher, output, "organization.create_agent", name="   ")
    assert response["error"]["code"] == "invalid_name"


def test_notifications_without_an_id_emit_an_error_event(server) -> None:
    dispatcher, output, _ = server
    request = parse(json.dumps({"method": "organization.create_agent", "params": {}}))
    assert request is not None
    dispatcher.handle(request, output)
    last = output.frames()[-1]
    assert last["type"] == "error"
    assert last["code"] == "invalid_name"


def test_creating_an_agent_emits_an_incremental_event(server) -> None:
    dispatcher, output, _ = server
    call(dispatcher, output, "organization.create_agent", name="Main")
    events = [item for item in output.frames() if item.get("type") == "member.created"]
    assert events and events[0]["name"] == "Main"


def test_events_are_deltas_not_whole_organization_snapshots(server) -> None:
    dispatcher, output, _ = server
    call(dispatcher, output, "organization.create_agent", name="Main")
    event = next(item for item in output.frames() if item["type"] == "member.created")
    assert set(event) == {"type", "id", "name", "state"}
    assert "members" not in event
    assert "discussions" not in event


def test_full_human_flow_over_the_protocol(server) -> None:
    dispatcher, output, deps = server
    agent = call(dispatcher, output, "organization.create_agent", name="Main")["result"]
    room = call(
        dispatcher,
        output,
        "discussion.create",
        topic="ship it",
        member_ids=[agent["id"]],
    )["result"]
    call(
        dispatcher, output, "discussion.send", discussion_id=room["id"], body="@Main go"
    )

    assert [item.message_id for item in deps.store.pending(agent["id"])] == [1]
    listed = call(dispatcher, output, "discussion.list")["result"]
    assert listed[0]["topic"] == "ship it"

    read = call(dispatcher, output, "discussion.read", discussion_id=room["id"])[
        "result"
    ]
    assert read["messages"][0]["body"] == "@Main go"
    assert read["messages"][0]["mentions"] == [
        {"member_id": agent["id"], "position": 0, "length": 5}
    ]


def test_ui_entry_tracks_read_boundary_across_own_messages(server) -> None:
    dispatcher, output, deps = server
    other = deps.store.create_member("human", "Other")
    room = deps.store.create_discussion("Entry", [HUMAN_ID, other.id])
    deps.store.append_message(room.id, other.id, "@You earlier pending")
    call(
        dispatcher, output, "discussion.mark_read", discussion_id=room.id, message_id=1
    )
    for index in range(55):
        sent = call(
            dispatcher,
            output,
            "discussion.send",
            discussion_id=room.id,
            body=f"Own message {index}",
            mark_read=False,
        )
        assert "result" in sent
    deps.store.append_message(room.id, other.id, "Later message")
    page = call(
        dispatcher, output, "discussion.page", discussion_id=room.id, entry=True
    )["result"]
    assert page["first_unread_id"] == 2
    assert page["read_through"] == deps.store.watermark(room.id, HUMAN_ID) == 1
    assert [message["id"] for message in page["messages"]] == list(range(1, 51))
    assert page["awaiting_ack"] == [1]
    assert page["pending_count"] == 1
    assert not page["has_before"] and page["has_after"]
    marked = call(
        dispatcher, output, "discussion.mark_read", discussion_id=room.id, message_id=20
    )["result"]
    assert marked["read_through"] == 20
    entered = call(
        dispatcher, output, "discussion.page", discussion_id=room.id, entry=True
    )["result"]
    assert entered["first_unread_id"] == 21
    assert [message["id"] for message in entered["messages"]] == list(range(1, 51))
    assert entered["awaiting_ack"] == [1]
    assert entered["pending_count"] == 1


def test_human_ack_and_revoke_round_trip(server) -> None:
    dispatcher, output, deps = server
    agent = call(dispatcher, output, "organization.create_agent", name="Main")["result"]
    room = call(
        dispatcher, output, "discussion.create", topic="t", member_ids=[agent["id"]]
    )["result"]
    call(
        dispatcher, output, "discussion.send", discussion_id=room["id"], body="@You hi"
    )
    deps.store.append_message(room["id"], agent["id"], "@You please review")

    call(dispatcher, output, "discussion.read", discussion_id=room["id"])
    assert len(deps.store.pending(HUMAN_ID)) == 1
    call(
        dispatcher, output, "discussion.ack", discussion_id=room["id"], message_ids=[2]
    )
    assert deps.store.pending(HUMAN_ID) == ()

    call(
        dispatcher,
        output,
        "discussion.revoke_ack",
        discussion_id=room["id"],
        message_ids=[2],
    )
    assert len(deps.store.pending(HUMAN_ID)) == 1


def test_archiving_stops_pending_and_unarchiving_restores_it(server) -> None:
    dispatcher, output, deps = server
    agent = call(dispatcher, output, "organization.create_agent", name="Main")["result"]
    room = call(
        dispatcher, output, "discussion.create", topic="t", member_ids=[agent["id"]]
    )["result"]
    call(
        dispatcher, output, "discussion.send", discussion_id=room["id"], body="@Main go"
    )

    call(
        dispatcher,
        output,
        "discussion.archive",
        discussion_id=room["id"],
        archived=True,
    )
    assert deps.store.pending(agent["id"]) == ()
    call(
        dispatcher,
        output,
        "discussion.archive",
        discussion_id=room["id"],
        archived=False,
    )
    assert len(deps.store.pending(agent["id"])) == 1


def test_concurrent_creation_and_model_updates_preserve_every_selection(server) -> None:
    from concurrent.futures import ThreadPoolExecutor

    dispatcher, _, deps = server

    def change(index):
        output = Capture()
        method = "organization.create_agent" if index % 2 == 0 else "settings.update"
        params = (
            {"name": f"Agent {index}", "model_config": {"thinking": "default"}}
            if index % 2 == 0
            else {
                "section": "model",
                "values": {
                    "action": "set_defaults",
                    "values": {"model_id": None, "thinking": "default"},
                },
            }
        )
        request = parse(json.dumps({"id": index, "method": method, "params": params}))
        dispatcher.handle(request, output)
        result = next(
            frame for frame in output.frames() if frame.get("type") == "response"
        )
        assert "error" not in result
        return result["result"]["id"] if index % 2 == 0 else None

    with ThreadPoolExecutor(max_workers=8) as pool:
        created = {
            str(agent_id)
            for agent_id in pool.map(change, range(40))
            if agent_id is not None
        }
    selections = deps.settings.get_settings("model")["agent_configs"]
    assert set(selections) == created
    assert len(created) == 20
    assert all(selection["thinking"] == "default" for selection in selections.values())


def test_agent_creation_and_model_configuration_are_atomic(server) -> None:
    dispatcher, output, deps = server
    db = deps.store._db
    before = deps.store.list_members()
    db.executescript(
        "CREATE TRIGGER reject_model_config BEFORE UPDATE ON settings "
        "WHEN NEW.section = 'model' BEGIN "
        "SELECT RAISE(ABORT, 'model configuration rejected'); END;"
    )
    rejected = call(dispatcher, output, "organization.create_agent", name="Rejected")
    assert "error" in rejected
    assert deps.store.list_members() == before
    assert not deps.settings.get_settings("model")["agent_configs"]
    assert not any(frame.get("type") == "member.created" for frame in output.frames())
    db.execute("DROP TRIGGER reject_model_config")
    created = call(dispatcher, output, "organization.create_agent", name="Created")[
        "result"
    ]
    assert deps.settings.get_settings("model")["agent_configs"][str(created["id"])] == {
        "model_id": None,
        "thinking": None,
    }
    created_events = [
        frame for frame in output.frames() if frame.get("type") == "member.created"
    ]
    assert len(created_events) == 1


def test_agent_creation_uses_model_selection_and_keeps_queries_public(server) -> None:
    dispatcher, output, deps = server
    deps.settings.set_settings(
        "model",
        {
            "version": 2,
            "providers": [
                {
                    "id": "provider",
                    "name": "Provider",
                    "api_type": "anthropic",
                    "base_url": "https://provider.invalid",
                    "api_key": "private-key",
                    "enabled": True,
                }
            ],
            "models": [
                {
                    "id": "model",
                    "provider_id": "provider",
                    "name": "Model",
                    "model": "claude-sonnet",
                    "enabled": True,
                    "thinking_budget_tokens": 4096,
                }
            ],
            "default_model_id": "model",
            "default_thinking": "high",
            "agent_configs": {"99": {"model_id": "model", "thinking": "default"}},
        },
    )
    created = call(
        dispatcher,
        output,
        "organization.create_agent",
        name="Configured",
        model_config={"model_id": "model", "thinking": "budget"},
    )["result"]
    stored = deps.settings.get_settings("model")
    assert stored["agent_configs"][str(created["id"])] == {
        "model_id": "model",
        "thinking": "budget",
    }
    assert ModelCatalog.restore(stored).resolve(created["id"]).thinking == "budget"
    selected = deps.settings.agent_model_selection(created["id"])
    assert selected == {
        "agent_id": created["id"],
        "model_config": {"model_id": "model", "thinking": "budget"},
        "effective": {"model_id": "model", "thinking": "budget"},
    }
    catalog = deps.settings.model_catalog()
    assert catalog == {
        "models": [
            {
                "id": "model",
                "name": "Model",
                "enabled": True,
                "thinking_options": [
                    "default",
                    "none",
                    "minimal",
                    "low",
                    "medium",
                    "high",
                    "xhigh",
                    "max",
                    "budget",
                ],
                "thinking_budget_tokens": 4096,
            }
        ],
        "default_model_id": "model",
        "default_thinking": "high",
    }
    assert "agent_configs" not in catalog
    assert "base_url" not in json.dumps(catalog)
    assert "api_key_set" not in json.dumps(catalog)

    before = deps.store.list_members()
    rejected = call(
        dispatcher,
        output,
        "organization.create_agent",
        name="Rejected",
        model_config={"model_id": "missing", "thinking": "default"},
    )
    assert rejected["error"]["code"] == "model_not_found"
    assert deps.store.list_members() == before
    assert not any(
        frame.get("name") == "Rejected"
        for frame in output.frames()
        if frame.get("type") == "member.created"
    )


def test_settings_never_return_the_api_key(server) -> None:
    dispatcher, output, _ = server
    call(
        dispatcher,
        output,
        "settings.update",
        section="model",
        values={
            "action": "save_provider",
            "values": {
                "name": "Provider",
                "api_type": "openai-chat",
                "base_url": "https://example.test/v1",
                "api_key": "super-secret-value",
            },
        },
    )
    result = call(dispatcher, output, "settings.get", section="model")["result"]
    assert "super-secret-value" not in json.dumps(result)
    assert result["providers"][0]["api_key_set"] is True
    assert result["providers"][0]["name"] == "Provider"


def test_model_settings_reject_unknown_api_types_and_keep_the_stored_ones(
    server,
) -> None:
    dispatcher, output, deps = server
    call(
        dispatcher,
        output,
        "settings.update",
        section="model",
        values={
            "action": "save_provider",
            "values": {
                "name": "Google",
                "api_type": "google",
                "base_url": "https://g.invalid",
            },
        },
    )
    rejected = call(
        dispatcher,
        output,
        "settings.update",
        section="model",
        values={
            "action": "save_provider",
            "values": {"name": "Other", "api_type": "azure"},
        },
    )
    assert rejected["error"]["code"] == "invalid_model_config"
    result = call(dispatcher, output, "settings.get", section="model")["result"]
    assert result["providers"][0]["api_type"] == "google"
    assert len(result["providers"]) == 1
    assert deps.settings.get_settings("model")["providers"][0]["api_type"] == "google"
    assert [
        frame for frame in output.frames() if frame.get("type") == "settings.updated"
    ] == [{"type": "settings.updated", "section": "model"}]


def test_agent_settings_fill_defaults_merge_and_emit_events(server) -> None:
    dispatcher, output, deps = server
    defaults = asdict(AgentParameters())
    assert (
        call(dispatcher, output, "settings.get", section="agent")["result"] == defaults
    )
    values = {
        "context_window_tokens": 12345,
        "exchange_nudge_after": 3,
        "max_concurrent_turns": 2,
        "idle_streak_after": 5,
        "no_tool_turns_before_pause": 2,
        "memory_index_bytes": 256,
        "token_limit": 1000,
        "request_limit": 75,
    }
    updated = call(
        dispatcher,
        output,
        "settings.update",
        section="agent",
        values=values,
    )["result"]
    assert updated == values
    assert (
        call(dispatcher, output, "settings.get", section="agent")["result"] == updated
    )
    assert deps.settings.get_settings("agent") == updated
    assert [
        frame for frame in output.frames() if frame["type"] == "settings.updated"
    ] == [{"type": "settings.updated", "section": "agent"}]
    assert (
        call(dispatcher, output, "settings.update", section="agent", values={})[
            "result"
        ]
        == updated
    )
    assert call(
        dispatcher,
        output,
        "settings.update",
        section="agent",
        values={"token_limit": 0},
    )["result"] == {**values, "token_limit": 0}


@pytest.mark.parametrize(
    ("values", "code"),
    [
        *[
            ({"context_window_tokens": value}, "invalid_parameter")
            for value in (0, -1, True, False, "100", None, 1.5)
        ],
        ({"max_concurrent_turns": 0}, "invalid_parameter"),
        ({"token_limit": -1}, "invalid_parameter"),
        ({"context_window_tokens": 100, "unknown": 1}, "invalid_setting"),
    ],
)
def test_agent_settings_validation_is_atomic(server, values, code) -> None:
    dispatcher, output, deps = server
    original = {"context_window_tokens": 12345}
    deps.settings.set_settings("agent", original)
    rejected = call(
        dispatcher, output, "settings.update", section="agent", values=values
    )
    assert rejected["error"]["code"] == code
    assert deps.settings.get_settings("agent") == original
    assert not any(frame["type"] == "settings.updated" for frame in output.frames())


@pytest.mark.parametrize("values", [None, False, 1, "", [], [["token_limit", 0]]])
def test_settings_update_requires_an_object(server, values) -> None:
    dispatcher, output, deps = server
    rejected = call(
        dispatcher, output, "settings.update", section="agent", values=values
    )
    assert rejected["error"]["code"] == "invalid_setting"
    assert deps.settings.get_settings("agent") is None
    assert not any(frame["type"] == "settings.updated" for frame in output.frames())


@pytest.mark.parametrize("section", ["agent", "model", "observability"])
@pytest.mark.parametrize("raw", ["null", "[]", '{"private-value":'])
def test_corrupt_settings_reject_reads_and_updates_without_changes(
    server, section, raw
):
    dispatcher, output, deps = server
    with deps.store._db:
        deps.store._db.execute(
            "INSERT INTO settings (section, values_json) VALUES (?, ?)"
            " ON CONFLICT (section) DO UPDATE SET values_json = excluded.values_json",
            (section, raw),
        )
    for method in ("settings.get", "settings.update"):
        result = call(dispatcher, output, method, section=section, values={})
        assert result["error"]["code"] == "invalid_setting"
        assert section in result["error"]["message"]
        assert "private-value" not in result["error"]["message"]
    stored = deps.store._db.execute(
        "SELECT values_json FROM settings WHERE section = ?", (section,)
    )
    assert stored[0]["values_json"] == raw
    assert not any(frame["type"] == "settings.updated" for frame in output.frames())


def test_agent_settings_validate_the_complete_update_before_saving(server) -> None:
    dispatcher, output, deps = server
    original = {"request_limit": -1, "token_limit": "1000"}
    deps.settings.set_settings("agent", original)
    rejected = call(dispatcher, output, "settings.get", section="agent")
    assert rejected["error"]["code"] == "invalid_parameter"
    assert "request_limit" in rejected["error"]["message"]
    rejected = call(
        dispatcher,
        output,
        "settings.update",
        section="agent",
        values={"request_limit": 5},
    )
    assert rejected["error"]["code"] == "invalid_parameter"
    assert "token_limit" in rejected["error"]["message"]
    assert deps.settings.get_settings("agent") == original
    assert not any(frame["type"] == "settings.updated" for frame in output.frames())
    corrected = {"request_limit": 5, "token_limit": 1000}
    result = call(
        dispatcher, output, "settings.update", section="agent", values=corrected
    )["result"]
    assert result == {**asdict(AgentParameters()), **corrected}
    assert deps.settings.get_settings("agent") == corrected
    assert call(dispatcher, output, "settings.get", section="agent")["result"] == result


@pytest.mark.parametrize("configured", [False, True])
def test_model_settings_reject_unknown_operations(server, configured) -> None:
    dispatcher, output, deps = server
    if configured:
        deps.settings.set_settings(
            "model",
            {"model": "m", "api_key": "unused", "base_url": "https://example.invalid"},
        )
    initial = deps.settings.get_settings("model")
    result = call(
        dispatcher,
        output,
        "settings.update",
        section="model",
        values={"action": "unknown"},
    )
    assert result["error"]["code"] == "invalid_model_action"
    assert deps.settings.get_settings("model") == initial


def test_model_listing_and_testing_reach_the_injected_probe(server) -> None:
    dispatcher, output, deps = server
    call(
        dispatcher,
        output,
        "settings.update",
        section="model",
        values={
            "action": "save_provider",
            "values": {
                "name": "Provider",
                "api_type": "openai-chat",
                "base_url": "https://stored.invalid/v1",
                "api_key": "stored-key",
            },
        },
    )
    stored = deps.settings.get_settings("model")
    listed = call(
        dispatcher,
        output,
        "settings.list_models",
        api_type="anthropic",
        base_url="https://a.invalid",
    )
    assert listed["result"] == {"models": ["b", "a"]}
    tested = call(
        dispatcher,
        output,
        "settings.test_model",
        api_type="anthropic",
        base_url="https://a.invalid",
        api_key="typed-key",
        model="candidate",
    )
    assert tested["result"] == {"ok": True, "latency_ms": 12, "reply": "OK"}
    assert deps.__dict__["probe"].calls == [
        ("list", {"api_type": "anthropic", "base_url": "https://a.invalid"}, stored),
        (
            "test",
            {
                "api_type": "anthropic",
                "base_url": "https://a.invalid",
                "api_key": "typed-key",
                "model": "candidate",
            },
            stored,
        ),
    ]
    updated = [f for f in output.frames() if f.get("type") == "settings.updated"]
    assert len(updated) == 1
    assert deps.settings.get_settings("model") == stored


def test_observability_keys_are_never_returned(server) -> None:
    dispatcher, output, _ = server
    call(
        dispatcher,
        output,
        "settings.update",
        section="observability",
        values={
            "enabled": True,
            "base_url": "u",
            "public_key": "pk",
            "secret_key": "sk",
        },
    )
    result = call(dispatcher, output, "settings.get", section="observability")["result"]
    assert "sk" not in json.dumps(result)
    assert "pk" not in json.dumps(result)
    assert result["keys_set"] is True


def test_execution_settings_update_the_live_sandbox(server, tmp_path: Path) -> None:
    dispatcher, output, deps = server
    target = tmp_path / "workspace"
    target.mkdir()
    call(
        dispatcher,
        output,
        "settings.update",
        section="execution",
        values={"write_directories": [str(target)]},
    )
    assert deps.execution.snapshot().write_directories == (str(target.resolve()),)
    assert deps.settings.get_settings("execution") == {
        "write_directories": [str(target.resolve())]
    }


def test_summary_paths_read_only_metadata(server) -> None:
    dispatcher, output, deps = server
    agent_id = call(dispatcher, output, "organization.create_agent", name="Main")[
        "result"
    ]["id"]
    payload = json.dumps([{"text": "x" * 1048576}])
    for _ in range(32):
        run = deps.history.start_run(agent_id)
        deps.history.finish_run(
            agent_id,
            run.sequence,
            status="completed",
            messages_json=payload,
            usage_json='{"input_tokens":100,"output_tokens":20,"last_input_tokens":200000,"tool_calls":1}',
        )
        deps.history.record_effect(agent_id, run.sequence, "send", "message")
    scheduler = Scheduler(deps, PydanticModelRunner(deps.history))
    connection = deps.store._db._connection

    def authorize(action, table, column, database, source):
        if (
            action == sqlite3.SQLITE_READ
            and table == "agent_runs"
            and column == "messages_json"
        ):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(authorize)
    try:
        detail = call(dispatcher, output, "agent.detail", agent_id=agent_id)["result"]
        organization = call(dispatcher, output, "organization.get")["result"]
        assert scheduler.preparation_due(agent_id)
    finally:
        connection.set_authorizer(None)
    assert len(detail["runs"]) == 30
    assert [run["sequence"] for run in detail["runs"]] == list(range(32, 2, -1))
    assert detail["runs"][0]["effects"] == [
        {"ordinal": 1, "tool": "send", "summary": "message"}
    ]
    assert detail["idle_streak"] == 0
    assert detail["usage"]["total_tokens"] == 3840
    member = next(item for item in organization["members"] if item["id"] == agent_id)
    assert member["tokens"] == 3840
    assert deps.history.latest_messages(agent_id) == payload


def test_agent_detail_reports_runs(server) -> None:
    dispatcher, output, deps = server
    agent = call(dispatcher, output, "organization.create_agent", name="Main")["result"]
    run = deps.history.start_run(agent["id"])
    deps.history.finish_run(
        agent["id"], run.sequence, status="completed", messages_json="[]"
    )

    detail = call(dispatcher, output, "agent.detail", agent_id=agent["id"])["result"]
    assert "todos" not in detail
    assert detail["runs"][0]["status"] == "completed"
    assert detail["window"] == {
        "number": 1,
        "since_sequence": 1,
        "reset_at": None,
        "reason": None,
    }
    state = deps.history.reset_window(agent["id"], "overflow")
    detail = call(dispatcher, output, "agent.detail", agent_id=agent["id"])["result"]
    assert detail["window"] == {
        "number": 2,
        "since_sequence": 2,
        "reset_at": state.reset_at,
        "reason": "overflow",
    }
    assert detail["runs"][0]["sequence"] == run.sequence


def test_agent_history_protocol_reads_turn_requests_and_legacy_data(server) -> None:
    dispatcher, output, deps = server
    agent = call(dispatcher, output, "organization.create_agent", name="Main")["result"]
    agent_id = agent["id"]
    run = deps.history.start_run(agent_id)
    handle = deps.history.start_model_request(
        agent_id,
        run.sequence,
        run.run_id,
        2,
        "[]",
        '{"function_tools":[{"name":"tool"}]}',
        '{"temperature":0.2}',
        '{"model":"test-model"}',
        False,
    )
    deps.history.finish_model_request(agent_id, run.sequence, handle, "[]")
    deps.history.finish_run(
        agent_id, run.sequence, status="completed", messages_json="[]"
    )
    page = call(dispatcher, output, "agent.history", agent_id=agent_id, limit=1)[
        "result"
    ]
    assert page["runs"][0]["request_count"] == 1
    assert page["runs"][0]["run_id"] == run.run_id
    read = call(
        dispatcher,
        output,
        "agent.history.read",
        agent_id=agent_id,
        sequence=run.sequence,
    )["result"]
    assert read["requests"][0]["request_id"] == handle.request_id
    assert read["request"]["parameters"] == {"function_tools": [{"name": "tool"}]}
    assert read["request"]["model"] == {"model": "test-model"}
    assert read["request"]["response"] is None

    legacy = deps.history.start_run(agent_id)
    deps.history.finish_run(
        agent_id,
        legacy.sequence,
        status="failed",
        messages_json="[]",
        error="provider failure",
    )
    legacy_read = call(
        dispatcher,
        output,
        "agent.history.read",
        agent_id=agent_id,
        sequence=legacy.sequence,
    )["result"]
    assert legacy_read["missing"] == [
        "provider_system_instructions",
        "model_input",
        "model_request_parameters",
        "model_settings",
        "model_identity",
        "request_window",
    ]
    assert legacy_read["run"]["legacy"] is True
    assert legacy_read["run"]["error"] == "provider failure"


def test_agent_history_requires_human_history_permission_before_body_reads(
    server,
) -> None:
    dispatcher, output, deps = server
    agent_id = call(dispatcher, output, "organization.create_agent", name="Main")[
        "result"
    ]["id"]
    run = deps.history.start_run(agent_id)
    deps.history.finish_run(
        agent_id,
        run.sequence,
        status="completed",
        messages_json="[]",
    )
    deps.scheduler._authorizer = Authorizer(
        lambda actor, capability, target: (
            "deny" if capability == "agent.history" else "allow"
        )
    )
    connection = deps.store._db._connection
    reads = []

    def authorize(action, table, column, database, source):
        if action == sqlite3.SQLITE_READ:
            reads.append((table, column))
            if table in {
                "agent_runs",
                "agent_model_requests",
                "agent_history_message_blobs",
            }:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection.set_authorizer(authorize)
    try:
        response = call(
            dispatcher,
            output,
            "agent.history.read",
            agent_id=agent_id,
            sequence=run.sequence,
        )
    finally:
        connection.set_authorizer(None)
    assert response["error"]["code"] == "not_permitted"
    assert not any(
        table in {"agent_runs", "agent_model_requests"} for table, _ in reads
    )


def test_agent_history_groups_tool_results_after_their_model_response(server) -> None:
    dispatcher, output, deps = server
    agent_id = call(dispatcher, output, "organization.create_agent", name="Main")[
        "result"
    ]["id"]
    messages = [
        ModelRequest(parts=[UserPromptPart("input")]),
        ModelResponse(
            parts=[ToolCallPart("tool", {"value": 1}, tool_call_id="call-1")]
        ),
        ModelRequest(parts=[ToolReturnPart("tool", "result", tool_call_id="call-1")]),
        ModelResponse(parts=[TextPart("done")]),
    ]
    raw = ModelMessagesTypeAdapter.dump_json(messages).decode()
    response_raw = ModelMessagesTypeAdapter.dump_json([messages[1]]).decode()
    run = deps.history.start_run(agent_id)
    handle = deps.history.start_model_request(
        agent_id,
        run.sequence,
        run.run_id,
        1,
        "[]",
        "{}",
        "{}",
        "{}",
        False,
    )
    deps.history.finish_model_request(agent_id, run.sequence, handle, response_raw)
    deps.history.link_model_request(
        agent_id,
        run.sequence,
        handle,
        (ModelMessagesTypeAdapter.dump_json([messages[2]]).decode(),),
    )
    deps.history.finish_run(
        agent_id, run.sequence, status="completed", messages_json=raw
    )
    read = call(
        dispatcher,
        output,
        "agent.history.read",
        agent_id=agent_id,
        sequence=run.sequence,
        ordinal=handle.ordinal,
    )["result"]
    related = read["request"]["related"]["messages"]
    assert len(related) == 1
    assert related[0]["parts"][0]["part_kind"] == "tool-return"


def test_agent_history_image_is_read_as_a_separate_binary_response(server) -> None:
    dispatcher, output, deps = server
    agent_id = call(dispatcher, output, "organization.create_agent", name="Main")[
        "result"
    ]["id"]
    raw = ModelMessagesTypeAdapter.dump_json(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        [BinaryContent(data=b"image-bytes", media_type="image/png")]
                    )
                ]
            )
        ]
    ).decode()
    run = deps.history.start_run(agent_id)
    handle = deps.history.start_model_request(
        agent_id,
        run.sequence,
        run.run_id,
        1,
        raw,
        "{}",
        "{}",
        "{}",
        False,
    )
    deps.history.finish_model_request(agent_id, run.sequence, handle, "[]")
    deps.history.finish_run(
        agent_id, run.sequence, status="completed", messages_json=raw
    )
    image = call(
        dispatcher,
        output,
        "agent.history.image",
        agent_id=agent_id,
        sequence=run.sequence,
        ordinal=handle.ordinal,
        source="input",
        message_index=0,
        path=["parts", 0, "content", 0],
    )["result"]
    assert image["media_type"] == "image/png"
    assert image["size"] == len(b"image-bytes")
    assert base64.b64decode(image["data"]) == b"image-bytes"


def test_agent_history_text_is_bounded_and_continuable(server) -> None:
    dispatcher, output, deps = server
    agent_id = call(dispatcher, output, "organization.create_agent", name="Main")[
        "result"
    ]["id"]
    content = "汉" * 12_000
    raw = ModelMessagesTypeAdapter.dump_json(
        [ModelRequest(parts=[UserPromptPart(content)])]
    ).decode()
    run = deps.history.start_run(agent_id)
    handle = deps.history.start_model_request(
        agent_id,
        run.sequence,
        run.run_id,
        1,
        (raw,),
        "{}",
        "{}",
        "{}",
        False,
    )
    deps.history.finish_run(
        agent_id, run.sequence, status="completed", messages_json=raw
    )
    read = call(
        dispatcher,
        output,
        "agent.history.read",
        agent_id=agent_id,
        sequence=run.sequence,
        ordinal=handle.ordinal,
    )["result"]
    value = read["request"]["input"]["messages"][0]["parts"][0]["content"]
    assert value["kind"] == "text"
    assert len(value["value"].encode()) <= 16 * 1024
    assert value["has_more"] is True
    following = call(
        dispatcher,
        output,
        "agent.history.text",
        agent_id=agent_id,
        sequence=run.sequence,
        source="input",
        ordinal=handle.ordinal,
        message_index=0,
        path=["parts", 0, "content"],
        offset=value["next_offset"],
    )["result"]
    assert following["value"]
    assert following["offset"] == value["next_offset"]


def test_agent_detail_reports_each_turn_output_and_the_idle_streak(server) -> None:
    dispatcher, output, deps = server
    agent = call(dispatcher, output, "organization.create_agent", name="Main")["result"]
    agent_id = agent["id"]

    productive = deps.history.start_run(agent_id)
    deps.history.record_effect(agent_id, productive.sequence, "send", "message 4")
    deps.history.finish_run(
        agent_id, productive.sequence, status="completed", messages_json="[]"
    )
    for _ in range(2):
        spinning = deps.history.start_run(agent_id)
        deps.history.record_effect(agent_id, spinning.sequence, "ack", "1 acknowledged")
        deps.history.finish_run(
            agent_id, spinning.sequence, status="completed", messages_json="[]"
        )

    detail = call(dispatcher, output, "agent.detail", agent_id=agent_id)["result"]
    assert detail["idle_streak"] == 2
    assert detail["idle"] is False
    assert detail["runs"][0]["effects"] == [
        {"ordinal": 1, "tool": "ack", "summary": "1 acknowledged"}
    ]
    assert detail["runs"][2]["effects"] == [
        {"ordinal": 1, "tool": "send", "summary": "message 4"}
    ]


def test_agent_idle_flag_follows_the_current_parameter(server) -> None:
    dispatcher, output, deps = server
    agent_id = call(dispatcher, output, "organization.create_agent", name="Main")[
        "result"
    ]["id"]
    for _ in range(3):
        run = deps.history.start_run(agent_id)
        deps.history.finish_run(
            agent_id, run.sequence, status="completed", messages_json="[]"
        )
    detail = call(dispatcher, output, "agent.detail", agent_id=agent_id)["result"]
    assert detail["idle_streak"] == 3
    assert detail["idle"] is True
    call(
        dispatcher,
        output,
        "settings.update",
        section="agent",
        values={"idle_streak_after": 5},
    )
    detail = call(dispatcher, output, "agent.detail", agent_id=agent_id)["result"]
    assert detail["idle_streak"] == 3
    assert detail["idle"] is False


@pytest.mark.parametrize("limit", [0, 100, 101])
def test_token_limit_reports_the_effective_agent_parameter(server, limit) -> None:
    dispatcher, output, deps = server
    agent_id = call(dispatcher, output, "organization.create_agent", name="Main")[
        "result"
    ]["id"]
    run = deps.history.start_run(agent_id)
    deps.history.finish_run(
        agent_id,
        run.sequence,
        status="completed",
        messages_json="[]",
        usage_json='{"input_tokens":100}',
    )
    legacy = {"agent_token_limit": 1}
    assert (
        call(dispatcher, output, "settings.update", section="limits", values=legacy)[
            "result"
        ]
        == legacy
    )
    assert (
        call(dispatcher, output, "settings.get", section="limits")["result"] == legacy
    )
    call(
        dispatcher,
        output,
        "settings.update",
        section="agent",
        values={"token_limit": limit},
    )
    organization = call(dispatcher, output, "organization.get")["result"]
    detail = call(dispatcher, output, "agent.detail", agent_id=agent_id)["result"]
    assert organization["token_limit"] == detail["token_limit"] == limit
    assert detail["over_token_limit"] is (limit == 100)


def test_library_round_trips_over_the_protocol(server) -> None:
    dispatcher, output, _ = server
    written = call(
        dispatcher, output, "library.write", path="notes.md", content="shared"
    )["result"]
    read = call(dispatcher, output, "library.read", path="notes.md")["result"]
    assert read["content"] == "shared"
    assert read["hash"] == written["hash"]


def test_rejected_write_directories_are_not_persisted(server, tmp_path: Path) -> None:
    dispatcher, output, deps = server
    good = tmp_path / "good"
    good.mkdir()
    call(
        dispatcher,
        output,
        "settings.update",
        section="execution",
        values={"write_directories": [str(good)]},
    )

    failed = call(
        dispatcher,
        output,
        "settings.update",
        section="execution",
        values={"write_directories": [str(good), "relative/bad"]},
    )
    assert failed["error"]["code"] == "invalid_directory"

    result = call(dispatcher, output, "settings.get", section="execution")["result"]
    assert result["write_directories"] == [str(good.resolve())]
    assert deps.execution.snapshot().write_directories == (str(good.resolve()),)


def test_accepted_write_directories_are_stored_canonically(
    server, tmp_path: Path
) -> None:
    dispatcher, output, _ = server
    target = tmp_path / "workspace"
    target.mkdir()
    call(
        dispatcher,
        output,
        "settings.update",
        section="execution",
        values={"write_directories": [f"{target}/", str(target)]},
    )
    result = call(dispatcher, output, "settings.get", section="execution")["result"]
    assert result["write_directories"] == [str(target.resolve())]


def test_library_directory_operations_and_workspace_read_protocol(server) -> None:
    dispatcher, output, deps = server
    agent_id = call(dispatcher, output, "organization.create_agent", name="Main")[
        "result"
    ]["id"]
    assert call(dispatcher, output, "library.mkdir", path="folder")["result"] == {
        "path": "folder",
        "hash": None,
    }
    call(
        dispatcher, output, "library.write", path="folder/note.txt", content="old old\n"
    )
    assert (
        call(
            dispatcher,
            output,
            "library.edit",
            path="folder/note.txt",
            old_text="old",
            new_text="new",
        )["error"]["code"]
        == "ambiguous_match"
    )
    edited = call(
        dispatcher,
        output,
        "library.edit",
        path="folder/note.txt",
        old_text="old",
        new_text="new",
        replace_all=True,
    )["result"]
    assert "+new new" in edited["diff"]
    call(dispatcher, output, "library.move", path="folder", destination="renamed")
    entries = call(dispatcher, output, "library.list")["result"]
    assert [(entry["path"], entry["kind"]) for entry in entries] == [
        ("renamed", "directory"),
        ("renamed/note.txt", "file"),
    ]
    assert all("hash" not in entry for entry in entries)
    assert (
        call(dispatcher, output, "library.read", path="renamed/note.txt")["result"][
            "hash"
        ]
        == edited["hash"]
    )
    assert call(dispatcher, output, "library.delete", path="renamed")["result"] == {
        "path": "renamed",
        "deleted": True,
    }
    assert call(dispatcher, output, "library.list")["result"] == []
    deps.workspace_tree_for(agent_id).write("topics/note.md", "private")
    workspace = call(dispatcher, output, "workspace.list", agent_id=agent_id)["result"]
    assert all("hash" not in entry for entry in workspace)
    assert [(entry["path"], entry["kind"]) for entry in workspace] == [
        ("MEMORY.md", "file"),
        ("topics", "directory"),
        ("topics/note.md", "file"),
    ]
    assert (
        call(dispatcher, output, "workspace.list", agent_id=agent_id, path="topics")[
            "result"
        ]
        == workspace[2:]
    )
    assert (
        call(
            dispatcher,
            output,
            "workspace.read",
            agent_id=agent_id,
            path="topics/note.md",
        )["result"]["content"]
        == "private"
    )
    assert (
        call(dispatcher, output, "agent.detail", agent_id=agent_id)["result"][
            "workspace"
        ]
        == workspace
    )
    for method in (
        "library.run",
        "workspace.write",
        "workspace.edit",
        "workspace.mkdir",
        "workspace.move",
        "workspace.delete",
    ):
        assert call(dispatcher, output, method)["error"]["code"] == "unknown_method"
    updates = [frame for frame in output.frames() if frame["type"] == "library.updated"]
    assert len(updates) == 6
    assert updates[-3:] == [
        {"type": "library.updated", "path": "folder", "deleted": True},
        {"type": "library.updated", "path": "renamed", "hash": None},
        {"type": "library.updated", "path": "renamed", "deleted": True},
    ]


class RealtimeSocket:
    def __init__(self, phase: str = "complete") -> None:
        self.incoming: asyncio.Queue[SimpleNamespace] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self.audio = bytearray()
        self.phase = phase
        self.blocked = threading.Event()
        self.closed_event = threading.Event()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        self.closed = True
        self.closed_event.set()

    async def send_json(self, event: dict[str, Any]) -> None:
        self.sent.append(event)
        if event["type"] == "session.update":
            if self.phase == "handshake":
                self.blocked.set()
                await asyncio.Future()
            event_type = "error" if self.phase == "failure" else "session.updated"
            await self.incoming.put(
                SimpleNamespace(
                    type=WSMsgType.TEXT,
                    data=json.dumps({"type": event_type}),
                )
            )
        elif event["type"] == "input_audio_buffer.append":
            self.audio.extend(base64.b64decode(event["audio"]))
            if self.phase == "transfer":
                self.blocked.set()
                await asyncio.Future()
        elif event["type"] == "input_audio_buffer.commit":
            if self.phase == "final":
                self.blocked.set()
                await asyncio.Future()
            for result in (
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "item_id": "i1",
                    "delta": "hello",
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "i1",
                    "transcript": "hello world",
                },
                {"type": "input_audio_buffer.committed", "item_id": "i1"},
            ):
                await self.incoming.put(
                    SimpleNamespace(type=WSMsgType.TEXT, data=json.dumps(result))
                )

    async def receive(self) -> SimpleNamespace:
        return await self.incoming.get()


class RealtimeClient:
    def __init__(self, phase: str = "complete", **kwargs: Any) -> None:
        self.headers: dict[str, str] = {}
        self.socket = RealtimeSocket(phase)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    def ws_connect(
        self, address: str, *, headers: dict[str, str], **kwargs: Any
    ) -> RealtimeSocket:
        self.headers = headers
        return self.socket


def test_remote_recording_streams_pcm_and_waits_for_commit_and_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huddol.adapters.voice import remote

    client = RealtimeClient()
    monkeypatch.setattr(remote.aiohttp, "ClientSession", lambda **kwargs: client)
    config = VoiceConfig.restore(
        {
            "address": "wss://service.invalid/realtime",
            "model": "gpt-live-transcribe",
            "api_key": "test-key",
        }
    )

    async def scenario() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        ready = asyncio.Event()

        async def emit(event: dict[str, Any]) -> None:
            events.append(event)
            if event["type"] == "ready":
                ready.set()

        recording = RemoteRecording(config, 48000, emit)
        worker = recording.start()
        await asyncio.wait_for(ready.wait(), timeout=2)
        recording.feed(bytes(19200))
        recording.stop()
        await asyncio.wait_for(worker, timeout=2)
        return events

    events = asyncio.run(scenario())
    assert client.headers == {"Authorization": "Bearer test-key"}
    assert client.socket.closed
    assert [event["type"] for event in events] == [
        "ready",
        "transcript",
        "transcript",
        "finished",
    ]
    assert events[-2]["text"] == "hello world"
    assert events[-1]["text"] == "hello world"
    assert len(client.socket.audio) == 4800
    assert [event["type"] for event in client.socket.sent] == [
        "session.update",
        "input_audio_buffer.append",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
    ]


@pytest.mark.parametrize("phase", ["handshake", "transfer", "final"])
def test_webserver_stop_closes_remote_recording_resources(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    from huddol.adapters.voice import remote

    clients: list[RealtimeClient] = []
    client_created = threading.Event()

    def create_client(**kwargs: Any) -> RealtimeClient:
        client = RealtimeClient(phase)
        clients.append(client)
        client_created.set()
        return client

    monkeypatch.setattr(remote.aiohttp, "ClientSession", create_client)
    server = WebServer(Dispatcher(), "test-token", None, port=0)
    endpoint = VoiceEndpoint(
        lambda: {
            "address": "wss://service.invalid/realtime",
            "model": "gpt-live-transcribe",
            "api_key": "test-key",
        }
    )
    server.add_websocket_route("/voice", endpoint.handle, max_msg_size=1048576)
    server.start()

    async def scenario() -> None:
        async with ClientSession() as session:
            with pytest.raises(WSServerHandshakeError) as unauthorized:
                await session.ws_connect(
                    f"http://127.0.0.1:{server.port}/voice?token=wrong"
                )
            assert unauthorized.value.status == 401
            async with session.ws_connect(
                f"http://127.0.0.1:{server.port}/voice?token=test-token"
            ) as connection:
                await connection.send_json(
                    {
                        "type": "start",
                        "sample_rate": 48000,
                        "channels": 1,
                        "format": "f32le",
                    }
                )
                assert await asyncio.to_thread(client_created.wait, 3)
                client = clients[0]
                if phase == "handshake":
                    assert await asyncio.to_thread(client.socket.blocked.wait, 3)
                else:
                    ready = await asyncio.wait_for(connection.receive_json(), 3)
                    assert ready == {"type": "ready", "mode": "remote"}
                    await connection.send_bytes(bytes(19200))
                    if phase == "transfer":
                        assert await asyncio.to_thread(client.socket.blocked.wait, 3)
                    else:
                        await connection.send_json({"type": "stop"})
                        assert await asyncio.wait_for(connection.receive_json(), 3) == {
                            "type": "finishing"
                        }
                        assert await asyncio.to_thread(client.socket.blocked.wait, 3)
                await asyncio.wait_for(asyncio.to_thread(server.stop), timeout=8)
                assert await asyncio.to_thread(client.socket.closed_event.wait, 3)

    try:
        asyncio.run(scenario())
    finally:
        if server._thread.is_alive():
            server.stop()
    assert not server._handlers


def test_server_stop_closes_every_active_remote_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huddol.adapters.voice import remote

    clients: list[RealtimeClient] = []
    client_created = threading.Event()

    def create_client(**kwargs: Any) -> RealtimeClient:
        client = RealtimeClient("transfer")
        clients.append(client)
        client_created.set()
        return client

    monkeypatch.setattr(remote.aiohttp, "ClientSession", create_client)
    server = WebServer(Dispatcher(), "test-token", None, port=0)
    endpoint = VoiceEndpoint(
        lambda: {
            "address": "wss://service.invalid/realtime",
            "model": "gpt-live-transcribe",
            "api_key": "test-key",
        }
    )
    server.add_websocket_route("/voice", endpoint.handle, max_msg_size=1048576)
    server.start()

    async def scenario() -> None:
        async with (
            ClientSession() as session,
            session.ws_connect(
                f"http://127.0.0.1:{server.port}/voice?token=test-token"
            ) as first,
            session.ws_connect(
                f"http://127.0.0.1:{server.port}/voice?token=test-token"
            ) as second,
        ):
            for connection in (first, second):
                await connection.send_json(
                    {
                        "type": "start",
                        "sample_rate": 48000,
                        "channels": 1,
                        "format": "f32le",
                    }
                )
            assert await asyncio.to_thread(client_created.wait, 3)
            async with asyncio.timeout(3):
                while len(clients) != 2:
                    await asyncio.sleep(0.005)
            assert await asyncio.wait_for(first.receive_json(), 3) == {
                "type": "ready",
                "mode": "remote",
            }
            assert await asyncio.wait_for(second.receive_json(), 3) == {
                "type": "ready",
                "mode": "remote",
            }
            await first.send_bytes(bytes(19200))
            await second.send_bytes(bytes(19200))
            assert await asyncio.to_thread(clients[0].socket.blocked.wait, 3)
            assert await asyncio.to_thread(clients[1].socket.blocked.wait, 3)
            await asyncio.wait_for(asyncio.to_thread(server.stop), timeout=8)
            assert await asyncio.to_thread(clients[0].socket.closed_event.wait, 3)
            assert await asyncio.to_thread(clients[1].socket.closed_event.wait, 3)

    try:
        asyncio.run(scenario())
    finally:
        if server._thread.is_alive():
            server.stop()
    assert not server._handlers


def test_remote_failure_allows_a_new_recording_on_the_same_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huddol.adapters.voice import remote

    clients: list[RealtimeClient] = []
    client_created = threading.Event()

    def create_client(**kwargs: Any) -> RealtimeClient:
        client = RealtimeClient("failure" if not clients else "complete")
        clients.append(client)
        client_created.set()
        return client

    monkeypatch.setattr(remote.aiohttp, "ClientSession", create_client)
    server = WebServer(Dispatcher(), "test-token", None, port=0)
    endpoint = VoiceEndpoint(
        lambda: {
            "address": "wss://service.invalid/realtime",
            "model": "gpt-live-transcribe",
            "api_key": "test-key",
        }
    )
    server.add_websocket_route("/voice", endpoint.handle, max_msg_size=1048576)
    server.start()

    async def scenario() -> None:
        async with (
            ClientSession() as session,
            session.ws_connect(
                f"http://127.0.0.1:{server.port}/voice?token=test-token"
            ) as connection,
        ):
            start = {
                "type": "start",
                "sample_rate": 48000,
                "channels": 1,
                "format": "f32le",
            }
            await connection.send_json(start)
            assert await asyncio.to_thread(client_created.wait, 3)
            failed = await asyncio.wait_for(connection.receive_json(), 3)
            assert failed["type"] == "error"
            assert failed["code"] == "voice_remote_error"
            assert await asyncio.to_thread(clients[0].socket.closed_event.wait, 3)
            await connection.send_json(start)
            async with asyncio.timeout(3):
                while len(clients) != 2:
                    await asyncio.sleep(0.005)
            assert await asyncio.wait_for(connection.receive_json(), 3) == {
                "type": "ready",
                "mode": "remote",
            }
            await connection.send_bytes(bytes(19200))
            await connection.send_json({"type": "stop"})
            transcripts = []
            async with asyncio.timeout(3):
                while True:
                    event = await connection.receive_json()
                    transcripts.append(event)
                    if event["type"] == "finished":
                        break
            assert any(event["type"] == "transcript" for event in transcripts)
            assert transcripts[-1] == {
                "type": "finished",
                "text": "hello world",
            }
            assert await asyncio.to_thread(clients[1].socket.closed_event.wait, 3)
            await connection.send_json(start)
            async with asyncio.timeout(3):
                while len(clients) != 3:
                    await asyncio.sleep(0.005)
            assert await asyncio.wait_for(connection.receive_json(), 3) == {
                "type": "ready",
                "mode": "remote",
            }
            await connection.send_json({"type": "cancel"})
            assert await asyncio.wait_for(connection.receive_json(), 3) == {
                "type": "cancelled"
            }
            assert await asyncio.to_thread(clients[2].socket.closed_event.wait, 3)
        await asyncio.wait_for(asyncio.to_thread(server.stop), timeout=8)

    try:
        asyncio.run(scenario())
    finally:
        if server._thread.is_alive():
            server.stop()
    assert not server._handlers


def test_remote_recording_cancel_closes_upstream_and_discards_late_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huddol.adapters.voice import remote

    client = RealtimeClient()
    monkeypatch.setattr(remote.aiohttp, "ClientSession", lambda **kwargs: client)
    config = VoiceConfig.restore(
        {
            "address": "wss://service.invalid/realtime",
            "model": "gpt-live-transcribe",
            "api_key": "test-key",
        }
    )

    async def scenario() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        ready = asyncio.Event()

        async def emit(event: dict[str, Any]) -> None:
            events.append(event)
            if event["type"] == "ready":
                ready.set()

        recording = RemoteRecording(config, 48000, emit)
        recording.start()
        await asyncio.wait_for(ready.wait(), timeout=2)
        await recording.close()
        return events

    events = asyncio.run(scenario())
    assert client.socket.closed
    assert [event["type"] for event in events] == ["ready"]
