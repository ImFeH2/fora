import asyncio
import errno
import json
import secrets
import socket
import sys
import threading
import time
from contextlib import ExitStack
from unittest.mock import patch

import psutil
import pytest
from aiohttp import WSMsgType
from aiohttp._websocket.writer import MSG_SIZE, WebSocketWriter
from aiohttp.base_protocol import BaseProtocol
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from fora.adapters.jsonl.protocol import Dispatcher, RequestScope
from fora.adapters.websocket import server as module


@pytest.mark.parametrize(
    "scenario",
    [
        "drain_success",
        "drain_error",
        "drain_large",
        "queued_duplicate",
        "business_duplicate",
        "old_callback_new_inflight",
        "write_failure",
        "drain_failure",
        "disconnect_during_drain",
        "success",
        "error",
    ],
)
def test_control_response_delivery_and_identifier_lifetime(scenario):
    drain = scenario.startswith("drain_") or scenario == "disconnect_during_drain"
    error_response = scenario in ("drain_error", "error")
    large = scenario == "drain_large"
    queued = scenario == "queued_duplicate"
    duplicate = scenario == "business_duplicate"
    old_cleanup = scenario == "old_callback_new_inflight"
    state = {}
    events = []
    lock = threading.Lock()
    entered = threading.Event()
    second_entered = threading.Event()
    old_waiting = threading.Event()
    business_release = threading.Event()
    business = []
    body = "x" * (MSG_SIZE + 1)
    dispatcher = Dispatcher()

    def record(event, **values):
        with lock:
            events.append((event, values))

    def handler(params):
        sequence = params["sequence"]
        business.append(sequence)
        record("business_entered", sequence=sequence)
        if sequence == 2:
            second_entered.set()
        if duplicate and sequence == 1:
            entered.set()
            assert business_release.wait(10)
        if old_cleanup and sequence == 2:
            assert business_release.wait(10)
        result = {"pong": sequence}
        if large:
            result["body"] = body
        return result

    dispatcher.register("ping", handler, scope=lambda params: RequestScope())
    server = module.WebServer(dispatcher, secrets.token_urlsafe(24), None, port=0)
    original_register = module.ControlOutbox.register_response
    original_put = module.ControlOutbox.put
    original_take = module.ControlOutbox.take
    original_close = module.WebServer._close_socket
    original_write = WebSocketWriter._write_websocket_frame
    original_drain = BaseProtocol._drain_helper
    original_shield = asyncio.shield

    def register(outbox):
        completion = original_register(outbox)
        if "completion" not in state:
            state.update(
                outbox=outbox,
                completion=completion,
                loop=asyncio.get_running_loop(),
                gate=asyncio.Event(),
                old_gate=asyncio.Event(),
                writes=0,
                releases=0,
                frames=0,
            )
        return completion

    def put(outbox, payload, completion=None, on_sent=None):
        if on_sent is None:
            return original_put(outbox, payload, completion)

        def notified():
            state["releases"] += 1
            if state["releases"] == 1:
                assert state["writes"] == (2 if large else 1)
                assert state["frames"] == 1
            record("protocol_id_released", completion_done=completion.done())
            on_sent()

        return original_put(outbox, payload, completion, notified)

    async def take(outbox):
        if queued and "queued" not in state:
            state["queued"] = True
            while not outbox.items:
                outbox.available.clear()
                await outbox.available.wait()
            record("response_queued")
            entered.set()
            await state["gate"].wait()
        return await original_take(outbox)

    def write(writer, message, opcode, rsv):
        if opcode != WSMsgType.TEXT:
            return original_write(writer, message, opcode, rsv)
        assert writer.compress == 0 and not writer.use_mask
        assert rsv == 0
        first = state["frames"] == 0
        if scenario == "write_failure" and first:
            record("write_failed")
            raise ConnectionResetError("Injected transport write failure")
        original_transport_write = writer.transport.write

        def transport_write(data):
            if first:
                assert state["releases"] == 0
                assert not state["completion"].done()
            original_transport_write(data)
            if first:
                state["writes"] += 1
                record("transport_write_completed", bytes=len(data))

        with patch.object(writer.transport, "write", transport_write):
            original_write(writer, message, opcode, rsv)
        state["frames"] += 1
        record("frame_written", bytes=len(message))
        if drain and first:
            state["protocol"] = writer.protocol
            writer._limit = 0
            writer.protocol._paused = True

    async def drain_helper(protocol):
        if protocol is state.get("protocol") and "draining" not in state:
            state["draining"] = True
            state["sending"] = asyncio.current_task()
            protocol._paused = False
            record("drain_wait_entered", completion_done=state["completion"].done())
            entered.set()
            try:
                await state["gate"].wait()
                if scenario == "drain_failure":
                    record("drain_failed")
                    raise ConnectionResetError("Injected drain failure")
                record("drain_wait_finished")
            finally:
                record("drain_finished")
            return
        return await original_drain(protocol)

    async def close(instance, connection, *, code=1000):
        if code in (1002, 1013):
            record("close_branch", code=code)
        return await original_close(instance, connection, code=code)

    def shield(value):
        if old_cleanup and value is state.get("completion"):
            state["old_task"] = asyncio.current_task()

            async def hold():
                result = await original_shield(value)
                old_waiting.set()
                await state["old_gate"].wait()
                return result

            return original_shield(hold())
        return original_shield(value)

    def request(connection, sequence, method="ping"):
        record("request_sent", sequence=sequence)
        connection.send(
            json.dumps({"id": 1, "method": method, "params": {"sequence": sequence}})
        )

    def response(connection):
        payload = json.loads(connection.recv(timeout=10))
        assert payload["type"] == "response" and payload["id"] == 1
        record("response_received")
        return payload

    def closed(connection, code):
        with pytest.raises(ConnectionClosed) as caught:
            connection.recv(timeout=10)
        assert caught.value.rcvd is not None and caught.value.rcvd.code == code
        record("connection_closed", code=code)

    with ExitStack() as patches:
        patches.enter_context(
            patch.object(module.ControlOutbox, "register_response", register)
        )
        patches.enter_context(patch.object(module.ControlOutbox, "put", put))
        patches.enter_context(patch.object(module.ControlOutbox, "take", take))
        patches.enter_context(patch.object(module.WebServer, "_close_socket", close))
        patches.enter_context(
            patch.object(WebSocketWriter, "_write_websocket_frame", write)
        )
        patches.enter_context(patch.object(BaseProtocol, "_drain_helper", drain_helper))
        patches.enter_context(patch.object(module.asyncio, "shield", shield))
        server.start()
        try:
            with connect(
                f"ws://127.0.0.1:{server.port}/ws?token={server._token}",
                proxy=None,
                open_timeout=10,
            ) as connection:
                request(connection, 1, "missing.method" if error_response else "ping")
                if scenario == "write_failure":
                    closed(connection, 1013)
                elif queued or duplicate:
                    assert entered.wait(10)
                    request(connection, 2)
                    closed(connection, 1002)
                else:
                    first = response(connection)
                    if error_response:
                        assert first["error"]["code"] == "unknown_method"
                    else:
                        expected = {"pong": 1, "body": body} if large else {"pong": 1}
                        assert first["result"] == expected
                    if drain:
                        assert entered.wait(10)
                        assert not state["completion"].done()
                        assert not state["sending"].done()
                    if scenario == "disconnect_during_drain":
                        connection.close()
                    elif scenario == "drain_failure":
                        state["loop"].call_soon_threadsafe(state["gate"].set)
                        closed(connection, 1013)
                    elif old_cleanup:
                        assert old_waiting.wait(10)
                        old = state["old_task"]
                        assert not old.done()
                        request(connection, 2)
                        assert second_entered.wait(10)
                        state["loop"].call_soon_threadsafe(state["old_gate"].set)

                        async def wait_old():
                            await asyncio.gather(old)
                            await asyncio.sleep(0)
                            assert len(server._requests) == 1
                            record("old_task_cleaned_new_request_pending")

                        asyncio.run_coroutine_threadsafe(
                            wait_old(), state["loop"]
                        ).result(10)
                        request(connection, 3)
                        closed(connection, 1002)
                    else:
                        request(connection, 2)
                        if drain:
                            assert second_entered.wait(10)
                            assert not state["completion"].done()
                            record("second_request_accepted_during_drain")
                            state["loop"].call_soon_threadsafe(state["gate"].set)
                        expected = {"pong": 2, "body": body} if large else {"pong": 2}
                        assert response(connection)["result"] == expected
        finally:
            business_release.set()
            if "loop" in state:
                state["loop"].call_soon_threadsafe(state["gate"].set)
                state["loop"].call_soon_threadsafe(state["old_gate"].set)
            server.stop()
    assert not server._thread.is_alive()
    assert not server._requests and not server._connections
    outbox = state["outbox"]
    assert outbox.closed and not outbox.items and not outbox.responses
    assert outbox.frames == outbox.used == outbox.producers == 0
    assert state["completion"].done()
    if "sending" in state:
        assert state["sending"].done()
    if scenario == "disconnect_during_drain":
        assert state["sending"].cancelled()
    assert server._socket.fileno() == -1
    deadline = time.monotonic() + 10
    refused = errno.WSAECONNREFUSED if sys.platform == "win32" else errno.ECONNREFUSED
    listening = None
    error_code = None
    port_closed = False
    while time.monotonic() < deadline:
        listening = any(
            item.status == psutil.CONN_LISTEN and item.laddr.port == server.port
            for item in psutil.Process().net_connections(kind="tcp")
        )
        with socket.socket() as probe:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            probe.settimeout(remaining)
            error_code = probe.connect_ex(("127.0.0.1", server.port))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if not listening and error_code == refused:
            port_closed = True
            break
        time.sleep(min(0.01, remaining))
    assert port_closed, f"LISTEN={listening}, connect_ex={error_code}"
    if queued or duplicate or scenario == "write_failure":
        assert state["releases"] == 0
        assert business == [1]
    elif old_cleanup or scenario in ("drain_failure", "disconnect_during_drain"):
        assert state["releases"] == 1
    else:
        assert state["releases"] == 2
        assert business == ([2] if error_response else [1, 2])
    names = [name for name, values in events]
    if large:
        writes = [
            values["bytes"]
            for name, values in events
            if name == "transport_write_completed"
        ]
        assert len(writes) == 2 and writes[0] <= 10 and writes[1] > MSG_SIZE
        assert names.index("frame_written") < names.index("protocol_id_released")
        assert names.index("protocol_id_released") < names.index(
            "second_request_accepted_during_drain"
        )
