from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from aiohttp import WSMsgType, web

from fora.adapters.voice.config import VoiceConfig
from fora.adapters.voice.remote import RemoteRecording
from fora.core.errors import DomainError


class VoiceEndpoint:
    def __init__(self, read_config: Callable[[], dict[str, Any] | None]) -> None:
        self._read_config = read_config

    async def handle(self, connection: web.WebSocketResponse) -> None:
        recording: RemoteRecording | None = None
        supervisor: asyncio.Task[None] | None = None
        terminal: dict[str, Any] | None = None

        async def emit(event: dict[str, Any]) -> None:
            if not connection.closed:
                await connection.send_json(event)

        async def recording_event(event: dict[str, Any]) -> None:
            nonlocal terminal
            if event["type"] in ("finished", "error"):
                terminal = event
            else:
                await emit(event)

        async def supervise(worker: asyncio.Task[None]) -> None:
            nonlocal recording
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                if not worker.cancelled():
                    raise
            except Exception:
                await connection.close(code=1011, message=b"Voice processing failed")
                raise
            finally:
                recording = None
            if terminal is not None:
                await emit(terminal)

        try:
            async for message in connection:
                try:
                    if message.type == WSMsgType.BINARY:
                        if recording is None:
                            raise DomainError(
                                "voice_state", "Start recording before sending audio"
                            )
                        recording.feed(message.data)
                        continue
                    if message.type == WSMsgType.ERROR:
                        raise ConnectionError("Voice connection failed")
                    if message.type != WSMsgType.TEXT:
                        continue
                    command = json.loads(message.data)
                    if not isinstance(command, dict):
                        raise DomainError(
                            "voice_command", "Expected a voice command object"
                        )
                    action = command.get("type")
                    if action == "start":
                        if recording is not None:
                            raise DomainError(
                                "voice_state", "This connection already has a recording"
                            )
                        config = VoiceConfig.restore(
                            await asyncio.to_thread(self._read_config)
                        )
                        rate = command.get("sample_rate")
                        if (
                            type(rate) is not int
                            or command.get("format") != "f32le"
                            or command.get("channels") != 1
                        ):
                            raise DomainError(
                                "voice_format",
                                "Declare mono f32le PCM and its sample rate",
                            )
                        if supervisor is not None:
                            await supervisor
                        if not config.api_key.strip():
                            raise DomainError(
                                "voice_config",
                                "Save an API key before testing transcription",
                            )
                        terminal = None
                        recording = RemoteRecording(config, rate, recording_event)
                        supervisor = asyncio.create_task(supervise(recording.start()))
                    elif action == "stop":
                        if recording is None:
                            raise DomainError("voice_state", "No active recording")
                        recording.stop()
                        await emit({"type": "finishing"})
                    elif action == "cancel":
                        terminal = None
                        if recording is not None:
                            await recording.close()
                            recording = None
                        if supervisor is not None:
                            await supervisor
                            supervisor = None
                        await emit({"type": "cancelled"})
                    else:
                        raise DomainError("voice_command", "Unknown voice command")
                except json.JSONDecodeError:
                    await emit(
                        {
                            "type": "error",
                            "code": "voice_command",
                            "message": "Invalid voice command JSON",
                        }
                    )
                except DomainError as error:
                    await emit(
                        {"type": "error", "code": error.code, "message": str(error)}
                    )
        finally:
            try:
                if recording is not None:
                    await recording.close()
            finally:
                if supervisor is not None:
                    await asyncio.shield(supervisor)
