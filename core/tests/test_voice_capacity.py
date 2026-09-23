from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from aiohttp import ClientSession, web

from huddol.adapters.voice.endpoint import VoiceEndpoint
from huddol.adapters.voice.local import LocalRecording
from huddol.adapters.voice.model import ModelStore
from huddol.core.errors import DomainError


class SlowEngine:
    def __init__(self, *, automatic: bool = False) -> None:
        self.condition = threading.Condition()
        self.calls = 0
        self.permits = 0
        self.cancelled = False
        self.closed = False
        self.automatic = automatic

    def transcribe(self, pcm: bytes) -> str:
        assert pcm
        with self.condition:
            self.calls += 1
            assert self.condition.wait_for(
                lambda: self.permits > 0 or self.cancelled or self.automatic,
                timeout=5,
            )
            if self.cancelled:
                raise DomainError("voice_cancelled", "Cancelled")
            if not self.automatic:
                self.permits -= 1
            return f"segment-{self.calls}"

    def advance(self) -> None:
        with self.condition:
            self.permits += 1
            self.condition.notify_all()

    def cancel(self) -> None:
        with self.condition:
            self.cancelled = True
            self.condition.notify_all()

    def close(self) -> None:
        self.closed = True


async def wait_until(predicate) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


def feed_seconds(recording: LocalRecording, start: int, count: int, rate: int) -> None:
    for second in range(start, start + count):
        value = 0.25 if second % 3 < 2 else 0.0
        recording.feed(np.full(rate, value, dtype="<f4").tobytes())
        pending = (
            recording._received_samples * 16000 - recording._retired_samples * rate
        )
        assert pending <= recording._limit
        if recording._failure is not None:
            return


@pytest.mark.parametrize("rate", [16000, 44100, 48000])
@pytest.mark.parametrize("overflow", [False, True])
def test_backlog_capacity_cancellation_and_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rate: int, overflow: bool
) -> None:
    monkeypatch.setattr(
        "huddol.adapters.voice.audio.webrtcvad.Vad",
        lambda _: Mock(
            is_speech=lambda pcm, _: bool(np.any(np.frombuffer(pcm, dtype="<i2")))
        ),
    )
    engine = SlowEngine()
    monkeypatch.setattr(LocalRecording, "_load", lambda _: (engine, 0.0, 0.0))

    async def scenario() -> None:
        events: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            events.append(event)
            if event.get("code") == "voice_capacity":
                assert engine.closed

        recording = LocalRecording(ModelStore(tmp_path), rate, emit)
        worker = recording.start()
        await wait_until(lambda: recording._ready)
        feed_seconds(recording, 0, 10, rate)
        await wait_until(lambda: engine.calls == 1)
        assert sum(window.final for window in recording._windows) >= 2
        feed_seconds(recording, 10, 10, rate)
        engine.advance()
        await wait_until(lambda: engine.calls == 2)
        assert sum(window.final for window in recording._windows) >= 4
        feed_seconds(recording, 20, 10, rate)
        engine.advance()
        await wait_until(lambda: engine.calls == 3)
        assert sum(window.final for window in recording._windows) >= 6
        if overflow:
            feed_seconds(recording, 30, 10, rate)
            await asyncio.wait_for(worker, timeout=5)
            assert events[-1]["code"] == "voice_capacity"
        else:
            await asyncio.wait_for(recording.close(), timeout=5)
            assert all(event["type"] != "error" for event in events)
        assert engine.closed
        assert recording._engine is None
        assert recording._incoming.empty()
        assert not recording._windows
        assert recording._received_samples == 0
        assert recording._retired_samples == 0

        next_engine = SlowEngine(automatic=True)
        monkeypatch.setattr(LocalRecording, "_load", lambda _: (next_engine, 0.0, 0.0))
        next_recording = LocalRecording(ModelStore(tmp_path), rate, emit)
        next_worker = next_recording.start()
        await wait_until(lambda: next_recording._ready)
        feed_seconds(next_recording, 0, 3, rate)
        next_recording.stop()
        await asyncio.wait_for(next_worker, timeout=5)
        assert events[-1]["type"] == "finished"
        assert events[-1]["text"]
        assert next_engine.closed

    asyncio.run(scenario())


def test_processed_silence_releases_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = SlowEngine(automatic=True)
    monkeypatch.setattr(LocalRecording, "_load", lambda _: (engine, 0.0, 0.0))

    async def scenario() -> None:
        events: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            events.append(event)

        recording = LocalRecording(ModelStore(tmp_path), 44100, emit)
        worker = recording.start()
        await wait_until(lambda: recording._ready)
        for _ in range(60):
            recording.feed(bytes(44100 * 4))
            await wait_until(recording._incoming.empty)
            assert recording._failure is None
        recording.stop()
        await asyncio.wait_for(worker, timeout=5)
        assert events[-1] == {"type": "finished", "text": ""}
        assert engine.calls == 0
        assert engine.closed

    asyncio.run(scenario())


def test_capacity_error_releases_endpoint_for_next_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = SlowEngine()
    monkeypatch.setattr(LocalRecording, "_load", lambda _: (engine, 0.0, 0.0))
    monkeypatch.setattr(
        "huddol.adapters.voice.audio.webrtcvad.Vad",
        lambda _: Mock(is_speech=lambda *_: True),
    )

    async def scenario() -> None:
        endpoint = VoiceEndpoint(tmp_path, lambda: {"mode": "local"})

        async def route(request: web.Request) -> web.WebSocketResponse:
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            await endpoint.handle(socket)
            return socket

        app = web.Application()
        app.router.add_get("/voice", route)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        start = {
            "type": "start",
            "format": "f32le",
            "sample_rate": 16000,
            "channels": 1,
        }
        try:
            async with (
                ClientSession() as client,
                client.ws_connect(
                    f"http://127.0.0.1:{runner.addresses[0][1]}/voice"
                ) as socket,
            ):
                await socket.send_json(start)
                assert (await socket.receive_json(timeout=5))["type"] == "ready"
                for _ in range(31):
                    await socket.send_bytes(bytes(16000 * 4))
                error = await socket.receive_json(timeout=5)
                assert error["code"] == "voice_capacity"
                assert engine.closed
                assert not endpoint._local_busy.locked()
                next_engine = SlowEngine(automatic=True)
                monkeypatch.setattr(
                    LocalRecording, "_load", lambda _: (next_engine, 0.0, 0.0)
                )
                await socket.send_json(start)
                assert (await socket.receive_json(timeout=5))["type"] == "ready"
                await socket.send_bytes(bytes(16000 * 4))
                await socket.send_json({"type": "stop"})
                while True:
                    event = await socket.receive_json(timeout=5)
                    assert event["type"] != "error"
                    if event["type"] == "finished":
                        assert event["text"]
                        break
                assert next_engine.closed
                assert not endpoint._local_busy.locked()
        finally:
            await runner.cleanup()

    asyncio.run(scenario())
