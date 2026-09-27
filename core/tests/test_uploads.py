from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
import secrets
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from http.client import HTTPConnection
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import httpx
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from pydantic_ai import BinaryContent, ModelMessagesTypeAdapter
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from websockets.sync.client import connect

from huddol.adapters.files.uploads import DirectoryUploads, decode_image
from huddol.adapters.jsonl.protocol import Dispatcher
from huddol.adapters.model.runner import attachment_result
from huddol.adapters.sqlite.agent import SqliteAgentStore
from huddol.adapters.sqlite.store import SqliteStore
from huddol.adapters.websocket.server import ControlOutbox, WebServer, resource
from huddol.core.errors import DomainError
from huddol.services.uploads import Uploads


@pytest.mark.parametrize(
    "phase", ["waiting", "encoding", "conversion", "queued", "sending"]
)
def test_control_outbox_close_ownership(phase: str, monkeypatch) -> None:
    import weakref

    from huddol.adapters.websocket import server

    entered = threading.Event()
    proceed = threading.Event()
    original = server.encode
    references = []

    class Payload(dict):
        pass

    class Encoded(str):
        def encode(self, encoding="utf-8", errors="strict"):
            data = super().encode(encoding, errors)
            entered.set()
            assert proceed.wait(5)
            return data

    def encode(payload):
        if phase == "encoding":
            entered.set()
            assert proceed.wait(5)
        text = original(payload)
        return Encoded(text) if phase == "conversion" else text

    monkeypatch.setattr(server, "encode", encode)

    async def exercise():
        outbox = ControlOutbox()
        if phase == "waiting":
            outbox.encoder.acquire()

        def produce():
            payload = Payload(type="event", text="x" * 1000)
            references.append(weakref.ref(payload))
            outbox.broadcast(payload)

        producer = asyncio.create_task(asyncio.to_thread(produce))
        if phase in ("encoding", "conversion"):
            assert await asyncio.to_thread(entered.wait, 5)
        elif phase == "waiting":
            async with asyncio.timeout(5):
                while outbox.frames != 1:
                    await asyncio.sleep(0.001)
        else:
            await producer
        assert outbox.frames == 1 and outbox.used > 0
        frame = await outbox.take() if phase == "sending" else None
        outbox.close()
        if phase in ("waiting", "encoding", "conversion", "sending"):
            assert outbox.frames == 1 and outbox.used > 0
        else:
            assert outbox.frames == outbox.used == 0
        if phase == "waiting":
            outbox.encoder.release()
        proceed.set()
        await producer
        if frame is not None:
            frame.text = ""
            outbox.release(frame)
        await asyncio.to_thread(outbox.wait_producers)
        assert outbox.frames == outbox.used == outbox.producers == 0
        assert not outbox.items
        assert all(reference() is None for reference in references)

    asyncio.run(exercise())


def test_control_slow_connection_isolated(monkeypatch) -> None:
    import weakref

    from aiohttp import ClientSession, WSMsgType

    from huddol.adapters.websocket import server as module

    references = []
    original = module.OutgoingFrame

    def frame(text, cost, completion):
        value = original(text, cost, completion)
        references.append(weakref.ref(value))
        return value

    monkeypatch.setattr(module, "OutgoingFrame", frame)
    outboxes = []

    def outbox():
        value = ControlOutbox()
        outboxes.append(value)
        return value

    monkeypatch.setattr(module, "ControlOutbox", outbox)
    dispatcher = Dispatcher()
    dispatcher.register("large", lambda params: "x" * (32 * 1024 * 1024))
    server = WebServer(dispatcher, "test-token", None, port=0)
    server.start()

    async def exercise():
        address = f"http://127.0.0.1:{server.port}/ws?token=test-token"
        async with (
            ClientSession() as session,
            session.ws_connect(address, max_msg_size=0) as slow,
            session.ws_connect(address) as normal,
        ):
            transport = slow._response.connection.transport
            transport.pause_reading()
            try:
                await slow.send_json({"id": 1, "method": "large"})
                await asyncio.sleep(0.3)
                for index in range(300):
                    await asyncio.to_thread(dispatcher.emit, "event", {"id": index})
                    await asyncio.sleep(0.002)
                await normal.send_json({"id": 2, "method": "ping"})
                async with asyncio.timeout(10):
                    while True:
                        message = await normal.receive()
                        assert message.type == WSMsgType.TEXT
                        frame = json.loads(message.data)
                        if frame.get("type") == "response":
                            assert frame == {
                                "type": "response",
                                "id": 2,
                                "result": {"pong": None},
                            }
                            break
                    if sys.platform == "win32":
                        transport.abort()
                    while len(server._handlers) != 1:
                        await asyncio.sleep(0.01)
                assert len(dispatcher._sinks) == 1
            finally:
                transport.abort()

    try:
        asyncio.run(exercise())
    finally:
        server.stop()
    assert not server._handlers and not dispatcher._sinks
    assert references and all(reference() is None for reference in references)
    assert len(outboxes) == 2
    for value in outboxes:
        assert value.frames == value.used == value.producers == 0
        assert not value.responses and not value.items


def test_control_heartbeat_during_large_response_and_worker() -> None:
    from aiohttp import ClientSession, WSMsgType

    dispatcher = Dispatcher()
    dispatcher.register("large", lambda params: "x" * (32 * 1024 * 1024))

    def delayed(params):
        time.sleep(45)
        return {"finished": True}

    dispatcher.register("delayed", delayed)
    server = WebServer(dispatcher, "test-token", None, port=0)
    server.start()
    address = f"http://127.0.0.1:{server.port}/ws?token=test-token"

    async def large(session):
        async with session.ws_connect(address, max_msg_size=0) as connection:
            transport = connection._response.connection.transport
            transport.pause_reading()
            await connection.send_json({"id": 1, "method": "large"})
            for _ in range(9):
                await asyncio.sleep(5)
                await connection.pong()
            transport.resume_reading()
            message = await connection.receive()
            assert message.type == WSMsgType.TEXT
            assert len(json.loads(message.data)["result"]) == 32 * 1024 * 1024

    async def long_worker(session):
        async with session.ws_connect(address) as connection:
            await connection.send_json({"id": 2, "method": "delayed"})
            message = await connection.receive()
            assert message.type == WSMsgType.TEXT
            assert json.loads(message.data)["result"] == {"finished": True}

    async def missing_pong(session):
        async with session.ws_connect(address, autoping=False) as connection:
            started = time.monotonic()
            message = await connection.receive()
            assert message.type == WSMsgType.PING
            message = await connection.receive()
            assert message.type in (WSMsgType.CLOSED, WSMsgType.CLOSE, WSMsgType.ERROR)
            assert 29 <= time.monotonic() - started < 36

    async def exercise():
        async with ClientSession() as session, asyncio.timeout(55):
            await asyncio.gather(
                large(session), long_worker(session), missing_pong(session)
            )

    try:
        asyncio.run(exercise())
    finally:
        server.stop()


@pytest.mark.parametrize("mode", ["worker", "sending"])
def test_control_owner_cancellation(mode: str, monkeypatch) -> None:
    import weakref

    from aiohttp import ClientSession

    from huddol.adapters.websocket import server as module

    started = threading.Event()
    sending = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    propagated = threading.Event()
    observed = threading.Event()
    outboxes = []
    references = []
    state = {}
    original_control = WebServer._control
    original_frame = module.OutgoingFrame
    original_send_str = module.web.WebSocketResponse.send_str

    def outbox():
        value = ControlOutbox()
        outboxes.append(value)
        return value

    def frame(text, cost, completion):
        value = original_frame(text, cost, completion)
        references.append(weakref.ref(value))
        return value

    async def control(self, connection):
        state["transport"] = self._control_transports[connection]
        task = asyncio.create_task(original_control(self, connection))
        state["task"] = task
        try:
            await task
        except asyncio.CancelledError:
            assert finished.is_set()
            propagated.set()
            raise

    async def send_str(connection, text):
        if (
            sys.platform == "win32"
            and mode == "sending"
            and len(text) > 32 * 1024 * 1024
        ):
            sending.set()
            await asyncio.Event().wait()
        return await original_send_str(connection, text)

    monkeypatch.setattr(module, "ControlOutbox", outbox)
    monkeypatch.setattr(module, "OutgoingFrame", frame)
    monkeypatch.setattr(module.web.WebSocketResponse, "send_str", send_str)
    monkeypatch.setattr(WebServer, "_control", control)
    dispatcher = Dispatcher()

    def operation(params):
        started.set()
        if mode == "worker":
            assert release.wait(10)
        finished.set()
        return "x" * (32 * 1024 * 1024) if mode == "sending" else "done"

    dispatcher.register("operation", operation)
    server = WebServer(dispatcher, "test-token", None, port=0)
    server.start()

    async def cancel_and_observe():
        task = state["task"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        observed.set()

    async def exercise():
        async with (
            ClientSession() as session,
            session.ws_connect(
                f"http://127.0.0.1:{server.port}/ws?token=test-token", max_msg_size=0
            ) as connection,
        ):
            transport = connection._response.connection.transport
            if mode == "sending":
                transport.pause_reading()
            try:
                await connection.send_json({"id": 1, "method": "operation"})
                assert await asyncio.to_thread(started.wait, 5)
                if mode == "sending":
                    if sys.platform == "win32":
                        assert await asyncio.to_thread(sending.wait, 5)
                    else:
                        async with asyncio.timeout(5):
                            while not state["transport"].get_write_buffer_size():
                                await asyncio.sleep(0.01)
                waiter = asyncio.run_coroutine_threadsafe(
                    cancel_and_observe(), server._loop
                )
                if mode == "worker":
                    await asyncio.sleep(0.1)
                    assert not propagated.is_set() and not observed.is_set()
                    assert server._handlers
                    release.set()
                await asyncio.wrap_future(waiter)
                async with asyncio.timeout(7):
                    while server._handlers:
                        await asyncio.sleep(0.01)
                assert propagated.is_set() and observed.is_set()
                assert state["transport"].is_closing()
                assert state["transport"].get_write_buffer_size() == 0
            finally:
                release.set()
                transport.abort()

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        server.stop()
    assert len(outboxes) == 1
    value = outboxes[0]
    assert value.used == value.frames == value.producers == 0
    assert not value.responses and not value.items
    assert all(reference() is None for reference in references)
    assert not dispatcher._sinks


def test_control_request_completion_sequence() -> None:
    dispatcher = Dispatcher()
    server = WebServer(dispatcher, "test-token", None, port=0)
    server.start()
    try:
        with connect(f"ws://127.0.0.1:{server.port}/ws?token=test-token") as connection:
            for text, expected in (
                (" ", None),
                ("{", "invalid_frame"),
                ('{"method":"missing"}', "unknown_method"),
                ('{"method":"ping"}', None),
            ):
                connection.send(text)
                if expected is None:
                    with pytest.raises(TimeoutError):
                        connection.recv(timeout=0.1)
                else:
                    assert json.loads(connection.recv(timeout=5))["code"] == expected
                connection.send('{"id":1,"method":"ping"}')
                assert json.loads(connection.recv(timeout=5)) == {
                    "type": "response",
                    "id": 1,
                    "result": {"pong": None},
                }
    finally:
        server.stop()


def test_control_outbox_encoding_failure(monkeypatch) -> None:
    from huddol.adapters.websocket import server

    def fail(payload):
        raise ValueError("encoding probe")

    monkeypatch.setattr(server, "encode", fail)

    async def exercise():
        outbox = ControlOutbox()
        completion = outbox.register_response()
        await asyncio.to_thread(outbox.broadcast, {"type": "event"})
        await asyncio.wait_for(outbox.closing.wait(), 1)
        outbox.close()
        with pytest.raises(ConnectionError):
            await completion
        assert outbox.frames == outbox.used == outbox.producers == 0
        assert not outbox.responses

    asyncio.run(exercise())


@pytest.mark.parametrize("reject", [False, True])
def test_attachment_and_model_settings_share_outer_transaction(
    tmp_path: Path, reject: bool
) -> None:
    store = SqliteStore(tmp_path / "transaction.sqlite3")
    try:
        human = store.create_member("human", "You")
        reader = store.create_member("agent", "Reader")
        room = store.create_discussion("Files", [human.id, reader.id])
        settings = SqliteAgentStore(store._db)
        settings.set_settings("model", {"agent_configs": {}})
        uploads = Uploads(store, DirectoryUploads(tmp_path / "uploads"))
        data = (Path(__file__).parents[2] / "app/icons/icon.ico").read_bytes()
        upload = uploads.create(
            room.id, human.id, str(uuid.uuid4()), "icon.ico", len(data), "image/x-icon"
        )
        active, target = uploads.begin(upload.id, human.id)
        with target:
            target.write(data)
        uploads.complete(active)
        uploads.end(active)
        receipt = str(uuid.uuid4())

        def transaction() -> None:
            with store._db:
                agent = store.create_member("agent", "Created")
                settings.set_settings(
                    "model", {"agent_configs": {str(agent.id): {"thinking": None}}}
                )
                message, _, _ = store.submit_message(
                    room.id,
                    human.id,
                    "@Reader",
                    attachment_ids=[upload.id],
                    client_message_id=receipt,
                )
                store.mark_read(room.id, reader.id, message.id)
                store.discussion_page(room.id, reader.id, limit=50)
                if reject:
                    raise ValueError("Reject outer transaction")

        if reject:
            with pytest.raises(ValueError, match="Reject outer transaction"):
                transaction()
            assert store.message_count(room.id) == 0
            assert store.pending(reader.id) == ()
            assert store.watermark(room.id, reader.id) == 0
            assert store.message_receipt(room.id, human.id, receipt) is None
            assert uploads.status(upload.id, human.id).state == "ready"
            assert settings.get_settings("model") == {"agent_configs": {}}
            assert len(store.list_members()) == 2
        else:
            transaction()
            message = store.messages(room.id)[0]
            repeated, _, created = store.submit_message(
                room.id,
                human.id,
                "@Reader",
                attachment_ids=[upload.id],
                client_message_id=receipt,
            )
            assert repeated.id == message.id and not created
            assert len(store.pending(reader.id)) == 1
            assert store.watermark(room.id, reader.id) == message.id
            assert uploads.status(upload.id, human.id).state == "attached"
            assert settings.get_settings("model")["agent_configs"]
            assert len(store.list_members()) == 3
            assert (
                len(store.discussion_page(room.id, reader.id, limit=50).messages) == 1
            )
    finally:
        store.close()


@pytest.mark.parametrize("sent_first", [False, True])
def test_send_cancellation_receipts_survive_restart(
    tmp_path: Path, sent_first: bool
) -> None:
    database = tmp_path / "cancel.sqlite3"
    store = SqliteStore(database)
    human = store.create_member("human", "You")
    reader = store.create_member("agent", "Reader")
    room = store.create_discussion("Cancel", [human.id, reader.id])
    other = store.create_discussion("Other", [human.id])
    attempt = str(uuid.uuid4())
    try:
        assert store.send_outcome(room.id, human.id, attempt) == ("unknown", None)
        if sent_first:
            store.submit_message(
                room.id, human.id, "@Reader", client_message_id=attempt, mark_read=True
            )
        state, message = store.cancel_send(room.id, human.id, attempt)
        assert state == ("sent" if sent_first else "cancelled")
        assert (message is not None) == sent_first
        assert store.cancel_send(room.id, human.id, attempt) == (state, message)
        assert store.message_count(room.id) == int(sent_first)
        assert len(store.pending(reader.id)) == int(sent_first)
        assert store.watermark(room.id, human.id) == int(sent_first)
        assert store.send_outcome(room.id, reader.id, attempt) == ("unknown", None)
        assert store.send_outcome(other.id, human.id, attempt) == ("unknown", None)
        with pytest.raises(DomainError):
            store.send_outcome(other.id, reader.id, attempt)
        with pytest.raises(DomainError):
            store.cancel_send(other.id, reader.id, attempt)
    finally:
        store.close()
    restored = SqliteStore(database)
    try:
        assert restored.send_outcome(room.id, human.id, attempt)[0] == state
        if sent_first:
            repeated, _, created = restored.submit_message(
                room.id, human.id, "@Reader", client_message_id=attempt
            )
            assert not created and repeated == message
            with pytest.raises(DomainError, match="different message content"):
                restored.submit_message(
                    room.id, human.id, "Changed", client_message_id=attempt
                )
        else:
            with pytest.raises(DomainError, match="cancelled"):
                restored.submit_message(
                    room.id, human.id, "@Reader", client_message_id=attempt
                )
        assert restored.message_count(room.id) == int(sent_first)
    finally:
        restored.close()


def _verify_cancelled_send_in_process(
    database: Path, discussion_id: int, owner_id: int, attempt: str
) -> None:
    store = SqliteStore(database)
    try:
        assert store.send_outcome(discussion_id, owner_id, attempt) == (
            "cancelled",
            None,
        )
        with pytest.raises(DomainError, match="cancelled"):
            store.submit_message(
                discussion_id, owner_id, "Late send", client_message_id=attempt
            )
        assert store.message_count(discussion_id) == 0
    finally:
        store.close()


def test_cancelled_send_in_new_process(tmp_path: Path) -> None:
    database = tmp_path / "process.sqlite3"
    store = SqliteStore(database)
    human = store.create_member("human", "You")
    room = store.create_discussion("Cancel", [human.id])
    attempt = str(uuid.uuid4())
    store.cancel_send(room.id, human.id, attempt)
    store.close()
    worker = multiprocessing.get_context("spawn").Process(
        target=_verify_cancelled_send_in_process,
        args=(database, room.id, human.id, attempt),
    )
    worker.start()
    try:
        worker.join(20)
        assert worker.exitcode == 0
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(5)
        worker.close()


@pytest.mark.parametrize("phase", ["BEFORE", "AFTER"])
def test_cancel_receipt_write_failure_is_atomic(tmp_path: Path, phase: str) -> None:
    store = SqliteStore(tmp_path / "rollback.sqlite3")
    try:
        human = store.create_member("human", "You")
        room = store.create_discussion("Cancel", [human.id])
        attempt = str(uuid.uuid4())
        store._db.execute(
            f"CREATE TRIGGER reject_cancel {phase} INSERT ON message_receipts "
            "WHEN NEW.state = 'cancelled' BEGIN SELECT RAISE(ABORT, 'cancel failed'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="cancel failed"):
            store.cancel_send(room.id, human.id, attempt)
        assert store.send_outcome(room.id, human.id, attempt) == ("unknown", None)
        assert store.message_count(room.id) == 0
        store._db.execute("DROP TRIGGER reject_cancel")
        assert store.cancel_send(room.id, human.id, attempt) == ("cancelled", None)
    finally:
        store.close()


def test_existing_sent_receipts_migrate(tmp_path: Path) -> None:
    database = tmp_path / "migrate.sqlite3"
    store = SqliteStore(database)
    human = store.create_member("human", "You")
    room = store.create_discussion("Migration", [human.id])
    attempt = str(uuid.uuid4())
    message, _, _ = store.submit_message(
        room.id, human.id, "Saved", client_message_id=attempt
    )
    with store._db:
        store._db.execute("ALTER TABLE message_receipts RENAME TO current_receipts")
        store._db.execute(
            "CREATE TABLE message_receipts (discussion_id INTEGER NOT NULL, "
            "owner_id INTEGER NOT NULL, client_message_id TEXT NOT NULL, message_id INTEGER NOT NULL, "
            "PRIMARY KEY(discussion_id, owner_id, client_message_id), "
            "FOREIGN KEY(discussion_id, message_id) REFERENCES messages(discussion_id, id))"
        )
        store._db.execute(
            "INSERT INTO message_receipts SELECT discussion_id, owner_id, client_message_id, message_id FROM current_receipts"
        )
        store._db.execute("DROP TABLE current_receipts")
    store.close()
    restored = SqliteStore(database)
    try:
        assert restored.send_outcome(room.id, human.id, attempt) == ("sent", message)
        assert restored.cancel_send(room.id, human.id, str(uuid.uuid4())) == (
            "cancelled",
            None,
        )
        assert restored._db.execute("PRAGMA foreign_key_check") == []
    finally:
        restored.close()


def test_real_upload_transaction_and_restart(tmp_path: Path) -> None:
    source = Path(__file__).parents[2] / "app/icons/icon.ico"
    image_path = tmp_path / "picture.png"
    with Image.open(source) as image:
        image.convert("RGB").save(image_path)
    data = image_path.read_bytes()
    database = tmp_path / "organization.sqlite3"
    store = SqliteStore(database)
    human = store.create_member("human", "You")
    agent = store.create_member("agent", "Reader")
    room = store.create_discussion("Files", [human.id, agent.id])
    files = DirectoryUploads(tmp_path / "uploads")
    uploads = Uploads(store, files)
    dispatcher = Dispatcher()
    token = secrets.token_urlsafe(32)
    server = WebServer(dispatcher, token, source.parent, uploads=uploads)
    server.start()
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            base = f"http://127.0.0.1:{server.port}"
            assert client.get(f"{base}/icon.ico").content == source.read_bytes()
            record = uploads.create(
                room.id, human.id, str(uuid.uuid4()), "图片.png", len(data), "image/png"
            )
            url = f"{base}/uploads/{record.id}/content"
            assert client.put(url, content=data).status_code == 401
            response = client.put(
                url, content=data, headers={"Authorization": f"Bearer {token}"}
            )
            assert response.status_code == 200, response.text
            attachment = response.json()
            assert attachment["sha256"] == hashlib.sha256(data).hexdigest()
            send_id = str(uuid.uuid4())
            message, mentions, created = store.submit_message(
                room.id,
                human.id,
                "@Reader picture",
                attachment_ids=[record.id],
                client_message_id=send_id,
            )
            assert created and len(mentions) == 1
            duplicate, _, created = store.submit_message(
                room.id,
                human.id,
                "@Reader picture",
                attachment_ids=[record.id],
                client_message_id=send_id,
            )
            assert not created and duplicate.id == message.id
            assert len(store.pending(agent.id)) == 1
            download = f"{base}/discussions/{room.id}/messages/{message.id}/attachments/{record.id}/content"
            result = client.get(
                download,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Origin": "tauri://localhost",
                },
            )
            assert result.status_code == 200
            assert result.content == data
            assert result.headers["Access-Control-Allow-Origin"] == "tauri://localhost"
            assert result.headers["Cache-Control"] == "no-store"
            store.change_discussion_members(room.id, [human.id], remove=True)
            assert (
                client.get(
                    download, headers={"Authorization": f"Bearer {token}"}
                ).status_code
                == 400
            )
            store.change_discussion_members(room.id, [human.id])
            store.set_archived(room.id, True)
            assert (
                client.get(
                    download, headers={"Authorization": f"Bearer {token}"}
                ).content
                == data
            )
        organization_uuid = store.organization_uuid()
    finally:
        server.stop()
        store.close()
    restored = SqliteStore(database)
    try:
        assert restored.organization_uuid() == organization_uuid
        assert restored.message_receipt(room.id, human.id, send_id).id == message.id
        assert restored.messages(room.id)[0].attachments[0].id == record.id
        assert Path(files.path(record.id)).read_bytes() == data
    finally:
        restored.close()


def test_cleanup_cancel_and_concurrent_sends(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "db.sqlite3")
    human = store.create_member("human", "You")
    room = store.create_discussion("Files", [human.id])
    uploads = Uploads(store, DirectoryUploads(tmp_path / "uploads"))
    try:
        record = uploads.create(room.id, human.id, str(uuid.uuid4()), "file.bin", 4, "")
        active, target = uploads.begin(record.id, human.id)
        target.write(b"ab")
        target.flush()
        uploads.cancel(record.id, human.id)
        target.close()
        uploads.end(active)
        assert store.get_upload(record.id, human.id).state == "expired"
        assert not Path(uploads.files.path(record.id) + ".part").exists()
        record = uploads.create(room.id, human.id, str(uuid.uuid4()), "file.bin", 4, "")
        active, target = uploads.begin(record.id, human.id)
        target.write(b"abcd")
        target.flush()
        os.fsync(target.fileno())
        target.close()
        uploads.complete(active)
        uploads.end(active)
        send_id = str(uuid.uuid4())
        results = []

        def send() -> None:
            results.append(
                store.submit_message(
                    room.id,
                    human.id,
                    "",
                    attachment_ids=[record.id],
                    client_message_id=send_id,
                )
            )

        threads = [threading.Thread(target=send) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == 2 and sum(result[2] for result in results) == 1
        assert store.message_count(room.id) == 1
        with pytest.raises(DomainError, match="sent message"):
            uploads.cancel(record.id, human.id)
    finally:
        store.close()


def test_websocket_control_and_private_route_shutdown() -> None:
    from aiohttp import WSMsgType, web

    finished = threading.Event()
    dispatcher = Dispatcher()
    token = secrets.token_urlsafe(32)
    server = WebServer(dispatcher, token, None)

    async def private(connection: web.WebSocketResponse) -> None:
        try:
            async for message in connection:
                if message.type == WSMsgType.BINARY:
                    await connection.send_bytes(message.data)
        finally:
            await asyncio.to_thread(time.sleep, 0.05)
            finished.set()

    server.add_websocket_route("/private", private, max_msg_size=65536)
    server.start()
    with (
        connect(f"ws://127.0.0.1:{server.port}/ws?token={token}") as control,
        connect(f"ws://127.0.0.1:{server.port}/private?token={token}") as separate,
    ):
        control.send(
            json.dumps({"id": 1, "method": "ping", "params": {"token": "test"}})
        )
        assert json.loads(control.recv(timeout=5))["result"] == {"pong": "test"}
        separate.send(b"audio bytes")
        assert separate.recv(timeout=5) == b"audio bytes"
        dispatcher.emit("test.event")
        assert json.loads(control.recv(timeout=5))["type"] == "test.event"
        with pytest.raises(TimeoutError):
            separate.recv(timeout=0.1)
        server.stop()
        assert finished.is_set()


def test_http_websocket_diagnosis() -> None:
    token = secrets.token_urlsafe(32)
    server = WebServer(Dispatcher(), token, None)
    server.start()
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            url = f"http://127.0.0.1:{server.port}/ws"
            for origin in (
                "tauri://localhost",
                "http://tauri.localhost",
                "http://localhost:1420",
            ):
                response = client.get(url, headers={"Origin": origin})
                assert response.status_code == 401
                assert "Access to Huddol requires authentication" in response.text
                assert response.headers["Content-Type"] == "text/html; charset=utf-8"
                assert response.headers["Cache-Control"] == "no-store"
                assert response.headers["Access-Control-Allow-Origin"] == origin
                response = client.get(
                    url, params={"token": token}, headers={"Origin": origin}
                )
                assert response.status_code == 204
                assert not response.content
                assert response.headers["Access-Control-Allow-Origin"] == origin
            response = client.get(
                url,
                params={"token": token},
                headers={"Origin": "https://untrusted.example"},
            )
            assert response.status_code == 204
            assert "Access-Control-Allow-Origin" not in response.headers
    finally:
        server.stop()


def test_resource_acquisition_cancel_releases_file(tmp_path: Path) -> None:
    entered = threading.Event()
    proceed = threading.Event()
    handles = []

    def acquire() -> BinaryIO:
        source = (tmp_path / "resource.bin").open("wb")
        handles.append(source)
        entered.set()
        assert proceed.wait(5)
        return source

    async def use() -> None:
        async with resource(acquire, lambda source: source.close()):
            pytest.fail("Cancelled acquisition must not enter the body")

    async def exercise() -> None:
        task = asyncio.create_task(use())
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert len(handles) == 1 and handles[0].closed


def restore_image_history(database: Path, report: Path, expected_digest: str) -> None:
    started = time.perf_counter()
    store = SqliteStore(database)
    try:
        agents = SqliteAgentStore(store._db)
        runs = agents.runs(2)
        assert {run.status for run in runs} == {"running", "completed", "failed"}
        json_sizes = []
        for run in runs:
            restored = ModelMessagesTypeAdapter.validate_json(run.messages_json)
            call = restored[0].parts[0]
            result = restored[1].parts[0]
            assert isinstance(call, ToolCallPart)
            assert isinstance(result, ToolReturnPart)
            assert result.tool_call_id == call.tool_call_id == "attachment-call"
            description, image = result.content
            assert isinstance(image, BinaryContent)
            assert len(image.data) == 5 * 1024 * 1024
            assert hashlib.sha256(image.data).hexdigest() == expected_digest
            assert image.media_type == "image/png"
            attachment = store.messages(1)[0].attachments[0]
            assert description["attachment_id"] == attachment.id
            assert description["discussion_id"] == description["message_id"] == 1
            json_sizes.append(len(run.messages_json.encode()))
        assert agents.latest_messages(2) != "[]"
        agents.reset_window(2, "Test new window")
        assert agents.latest_messages(2) == "[]"
        uploads = Uploads(store, DirectoryUploads(database.parent / "uploads"))
        attachment = store.messages(1)[0].attachments[0]
        assert (
            hashlib.sha256(uploads.view(1, 1, attachment.id, 2).image.data).hexdigest()
            == expected_digest
        )
        elapsed = time.perf_counter() - started
        report.write_text(
            json.dumps({"json_bytes": json_sizes, "restore_seconds": elapsed})
        )
    finally:
        store.close()


def test_image_history_survives_process_restart(tmp_path: Path) -> None:
    original = Path(__file__).parents[2] / "app/icons/icon.ico"
    with Image.open(original) as source:
        image = source.convert("RGB")
    metadata = PngInfo()
    metadata.add_text("budget", "")
    baseline = BytesIO()
    image.save(baseline, format="PNG", pnginfo=metadata)
    metadata = PngInfo()
    metadata.add_text("budget", "x" * (5 * 1024 * 1024 - len(baseline.getvalue())))
    content = BytesIO()
    image.save(content, format="PNG", pnginfo=metadata)
    image.close()
    data = content.getvalue()
    assert len(data) == 5 * 1024 * 1024
    database = tmp_path / "organization.sqlite3"
    store = SqliteStore(database)
    try:
        human = store.create_member("human", "You")
        agent = store.create_member("agent", "Reader")
        discussion = store.create_discussion("Images", [human.id, agent.id])
        uploads = Uploads(store, DirectoryUploads(tmp_path / "uploads"))
        record = uploads.create(
            discussion.id,
            human.id,
            str(uuid.uuid4()),
            "picture.png",
            len(data),
            "image/png",
        )
        active, target = uploads.begin(record.id, human.id)
        with target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        uploads.complete(active)
        uploads.end(active)
        message, _, _ = store.submit_message(
            discussion.id, human.id, "", attachment_ids=[record.id]
        )
        result = attachment_result(
            uploads.view(discussion.id, message.id, record.id, agent.id)
        )
        messages = [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        "view_attachment",
                        {
                            "discussion_id": discussion.id,
                            "message_id": message.id,
                            "attachment_id": record.id,
                        },
                        "attachment-call",
                    )
                ]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        "view_attachment", result.return_value, "attachment-call"
                    )
                ]
            ),
        ]
        raw = ModelMessagesTypeAdapter.dump_json(messages).decode()
        agents = SqliteAgentStore(store._db)
        progress = agents.start_run(agent.id)
        agents.save_progress(agent.id, progress.sequence, raw)
        for status in ("completed", "failed"):
            run = agents.start_run(agent.id)
            agents.finish_run(agent.id, run.sequence, status=status, messages_json=raw)
    finally:
        store.close()
    report = tmp_path / "history-report.json"
    process = multiprocessing.get_context("spawn").Process(
        target=restore_image_history,
        args=(database, report, hashlib.sha256(data).hexdigest()),
    )
    process.start()
    process.join(30)
    if process.is_alive():
        process.terminate()
        process.join(5)
        pytest.fail("History restoration exceeded 30 seconds")
    assert process.exitcode == 0
    process.close()
    metrics = json.loads(report.read_text())
    assert all(size > len(data) for size in metrics["json_bytes"])
    print(metrics)


@pytest.mark.parametrize("format_name", ["PNG", "JPEG", "WEBP"])
def test_image_formats_and_content_validation(format_name: str) -> None:
    original = Path(__file__).parents[2] / "app/icons/icon.ico"
    with Image.open(original) as source:
        image = source.convert("RGB")
    content = BytesIO()
    image.save(content, format=format_name)
    result = decode_image(content.getvalue())
    assert (result.width, result.height) == image.size
    assert result.media_type == Image.MIME[format_name]
    assert result.data == content.getvalue()
    with pytest.raises(DomainError):
        decode_image(content.getvalue()[:30])
    image.close()


@pytest.mark.parametrize(
    ("size", "accepted"),
    [
        ((8192, 1), True),
        ((8193, 1), False),
        ((4000, 5000), True),
        ((4000, 5001), False),
    ],
)
def test_image_dimensions(size: tuple[int, int], accepted: bool) -> None:
    with Image.new("L", size) as image:
        content = BytesIO()
        image.save(content, format="PNG")
    if accepted:
        result = decode_image(content.getvalue())
        assert (result.width, result.height) == size
    else:
        with pytest.raises(DomainError) as failure:
            decode_image(content.getvalue())
        assert failure.value.code == "image_dimensions_exceeded"


@pytest.mark.parametrize("format_name", ["PNG", "WEBP"])
def test_animated_images_are_rejected(format_name: str) -> None:
    original = Path(__file__).parents[2] / "app/icons/icon.ico"
    with Image.open(original) as source:
        image = source.convert("RGB")
    mirrored = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    content = BytesIO()
    image.save(
        content,
        format=format_name,
        save_all=True,
        append_images=[mirrored],
        duration=100,
    )
    image.close()
    mirrored.close()
    with pytest.raises(DomainError) as failure:
        decode_image(content.getvalue())
    assert failure.value.code == "animated_image"


def test_missing_ready_files_can_be_uploaded_again(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "database.sqlite3")
    try:
        human = store.create_member("human", "You")
        discussion = store.create_discussion("Files", [human.id])
        uploads = Uploads(store, DirectoryUploads(tmp_path / "uploads"))
        original = Path(__file__).parents[2] / "app/icons/icon.ico"
        data = original.read_bytes()
        record = uploads.create(
            discussion.id,
            human.id,
            str(uuid.uuid4()),
            "icon.ico",
            len(data),
            "image/x-icon",
        )
        active, target = uploads.begin(record.id, human.id)
        with target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        uploads.complete(active)
        uploads.end(active)
        assert uploads.status(record.id, human.id).state == "ready"
        assert (
            uploads.create(
                discussion.id,
                human.id,
                record.client_upload_id,
                record.name,
                record.size,
                "image/x-icon",
            ).id
            == record.id
        )
        Path(uploads.files.path(record.id)).unlink()
        assert uploads.status(record.id, human.id).state == "expired"
        with pytest.raises(DomainError) as failure:
            store.submit_message(
                discussion.id, human.id, "", attachment_ids=[record.id]
            )
        assert failure.value.code == "attachment_not_ready"
        assert store.message_count(discussion.id) == 0
        replacement = uploads.create(
            discussion.id,
            human.id,
            str(uuid.uuid4()),
            record.name,
            record.size,
            "image/x-icon",
        )
        assert replacement.state == "reserved" and replacement.id != record.id
    finally:
        store.close()


@pytest.mark.parametrize("invalid_settings", [False, True])
def test_process_shutdown_releases_incomplete_upload(
    tmp_path: Path, invalid_settings: bool
) -> None:
    from test_sidecar_process import Client, Kernel

    environment = {name: str(tmp_path) for name in ("TMPDIR", "TEMP", "TMP")}
    data_directory = tmp_path / "organization"
    with Kernel(data_directory, env=environment) as kernel, kernel.connect() as control:
        client = Client(control)
        assert "result" in client.call(
            {
                "id": 1,
                "method": "organization.create_agent",
                "params": {"name": "Reader"},
            }
        )
        room = client.call(
            {
                "id": 2,
                "method": "discussion.create",
                "params": {"topic": "Files", "member_ids": [2]},
            }
        )["result"]["id"]
        upload = client.call(
            {
                "id": 3,
                "method": "upload.create",
                "params": {
                    "discussion_id": room,
                    "client_upload_id": str(uuid.uuid4()),
                    "name": "icon.ico",
                    "size": 6662,
                    "media_type": "image/x-icon",
                },
            }
        )["result"]
        connection = HTTPConnection("127.0.0.1", kernel.port, timeout=5)
        try:
            connection.putrequest("PUT", f"/uploads/{upload['id']}/content")
            connection.putheader("Authorization", f"Bearer {kernel.token}")
            connection.putheader("Content-Length", "6662")
            connection.endheaders()
            connection.send(b"\x00")
            deadline = time.monotonic() + 5
            while True:
                status = client.call(
                    {
                        "id": 4,
                        "method": "upload.status",
                        "params": {"upload_ids": [upload["id"]]},
                    }
                )["result"][0]
                if status["state"] == "receiving":
                    break
                assert time.monotonic() < deadline
                time.sleep(0.01)
            if invalid_settings:
                database = sqlite3.connect(data_directory / "huddol.sqlite3")
                try:
                    with database:
                        database.execute(
                            "INSERT INTO settings (section, values_json) "
                            "VALUES ('agent', ?) ON CONFLICT(section) "
                            "DO UPDATE SET values_json = excluded.values_json",
                            ('{"token_limit":-1}',),
                        )
                    log_path = data_directory / "logs" / "huddol.log"
                    deadline = time.monotonic() + 10
                    while (
                        "Scheduler stopped after a runtime failure"
                        not in log_path.read_text()
                    ):
                        assert time.monotonic() < deadline
                        time.sleep(0.01)
                    result = client.call(
                        {
                            "id": 5,
                            "method": "settings.get",
                            "params": {"section": "agent"},
                        }
                    )
                    assert result["error"]["code"] == "invalid_parameter"
                    diagnosis = httpx.get(
                        f"http://127.0.0.1:{kernel.port}/ws",
                        params={"token": kernel.token},
                    )
                    assert diagnosis.status_code == 204
                    assert (
                        httpx.get(f"http://127.0.0.1:{kernel.port}/ws").status_code
                        == 401
                    )
                    assert client.call({"id": 6, "method": "ping"})["result"] == {
                        "pong": None
                    }
                finally:
                    database.close()
            assert kernel.shutdown() == 0, kernel.stderr
        finally:
            connection.close()
    assert not (data_directory / "uploads" / f"{upload['id']}.part").exists()
    if invalid_settings:
        database = sqlite3.connect(data_directory / "huddol.sqlite3")
        try:
            with database:
                database.execute("DELETE FROM settings WHERE section = 'agent'")
        finally:
            database.close()
    with (
        Kernel(data_directory, env=environment) as restarted,
        restarted.connect() as control,
    ):
        client = Client(control)
        result = client.call(
            {
                "id": 1,
                "method": "upload.status",
                "params": {"upload_ids": [upload["id"]]},
            }
        )
        assert result["result"][0]["state"] == "expired"
        assert restarted.shutdown() == 0, restarted.stderr


def test_process_reports_port_binding_failure(tmp_path: Path) -> None:
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        environment = {
            "PATH": os.environ["PATH"],
            "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
            "HUDDOL_DATA_DIR": str(tmp_path / "organization"),
            "TMPDIR": str(tmp_path),
            "TEMP": str(tmp_path),
            "TMP": str(tmp_path),
        }
        for name in ("SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "PATHEXT"):
            if name in os.environ:
                environment[name] = os.environ[name]
        result = subprocess.run(
            [sys.executable, "-m", "huddol", "--port", str(occupied.getsockname()[1])],
            env=environment,
            capture_output=True,
            timeout=30,
            check=False,
        )
    assert result.returncode != 0
    assert not result.stdout
    assert b"Address already in use" in result.stderr or b"10048" in result.stderr
