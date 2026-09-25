from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp

from huddol.adapters.voice.audio import AudioConverter
from huddol.adapters.voice.config import VoiceConfig
from huddol.core.errors import DomainError

Emit = Callable[[dict[str, Any]], Awaitable[None]]


class RemoteRecording:
    def __init__(self, config: VoiceConfig, input_rate: int, emit: Emit) -> None:
        self._config = config
        self._emit = emit
        self._converter = AudioConverter(input_rate, 24000)
        self._incoming: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._queued_bytes = 0
        self._limit = input_rate * 4 * 30
        self._ready = False
        self._finishing = False
        self._cancelled = False
        self._task: asyncio.Task[None] | None = None
        self._committed = False

    def start(self) -> asyncio.Task[None]:
        if self._task is not None:
            raise DomainError("voice_state", "Recording has already started")
        self._task = asyncio.create_task(self._run())
        return self._task

    def feed(self, frame: bytes) -> None:
        if not self._ready or self._finishing or self._cancelled:
            raise DomainError("voice_state", "Recording is not accepting audio")
        if not frame or self._queued_bytes + len(frame) > self._limit:
            raise DomainError(
                "voice_capacity", "Audio upload cannot keep up with recording"
            )
        self._queued_bytes += len(frame)
        self._incoming.put_nowait(frame)

    def stop(self) -> None:
        if not self._ready or self._finishing:
            raise DomainError("voice_state", "Recording is not active")
        self._finishing = True
        self._incoming.put_nowait(None)

    async def close(self) -> None:
        self._cancelled = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _event(self, socket: aiohttp.ClientWebSocketResponse) -> dict[str, Any]:
        message = await socket.receive()
        if message.type != aiohttp.WSMsgType.TEXT:
            raise DomainError(
                "voice_remote_closed", "Transcription service connection closed"
            )
        event = json.loads(message.data)
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise DomainError(
                "voice_remote_protocol", "Invalid transcription service event"
            )
        if event["type"] == "error" or event["type"].endswith(".failed"):
            raise DomainError(
                "voice_remote_error",
                "Transcription service rejected the request; check address, model and API key",
            )
        return event

    async def _read(self, socket: aiohttp.ClientWebSocketResponse) -> str:
        item: str | None = None
        text = ""
        committed_item: str | None = None
        final_text: str | None = None
        while True:
            event = await self._event(socket)
            kind = event["type"]
            if kind == "input_audio_buffer.committed":
                if (
                    committed_item is not None
                    or not self._committed
                    or not isinstance(event.get("item_id"), str)
                ):
                    raise DomainError(
                        "voice_remote_protocol", "Unexpected audio commit"
                    )
                committed_item = event["item_id"]
            elif kind in (
                "conversation.item.input_audio_transcription.delta",
                "conversation.item.input_audio_transcription.completed",
            ):
                event_item = event.get("item_id")
                if not isinstance(event_item, str) or (
                    item is not None and item != event_item
                ):
                    raise DomainError(
                        "voice_remote_protocol", "Unexpected transcription item"
                    )
                item = event_item
                if kind.endswith(".delta"):
                    delta = event.get("delta")
                    if not isinstance(delta, str) or final_text is not None:
                        raise DomainError(
                            "voice_remote_protocol", "Invalid transcription delta"
                        )
                    text += delta
                    await self._emit(
                        {"type": "transcript", "text": text, "final": False}
                    )
                else:
                    if not self._committed or not isinstance(
                        event.get("transcript"), str
                    ):
                        raise DomainError(
                            "voice_remote_protocol", "Unexpected final transcript"
                        )
                    final_text = event["transcript"]
            if (
                committed_item is not None
                and item is not None
                and committed_item != item
            ):
                raise DomainError(
                    "voice_remote_protocol", "Audio commit and transcript item differ"
                )
            if committed_item is not None and final_text is not None:
                await self._emit(
                    {"type": "transcript", "text": final_text, "final": True}
                )
                return final_text

    async def _stream(self, socket: aiohttp.ClientWebSocketResponse) -> str:
        reader = asyncio.create_task(self._read(socket))
        pending: asyncio.Task[bytes | None] | None = None
        sent = 0
        try:
            while True:
                pending = asyncio.create_task(self._incoming.get())
                done, _ = await asyncio.wait(
                    (pending, reader), return_when=asyncio.FIRST_COMPLETED
                )
                if reader in done:
                    await reader
                    raise DomainError(
                        "voice_remote_protocol",
                        "Transcription completed before stopping",
                    )
                frame = pending.result()
                pending = None
                if frame is not None:
                    self._queued_bytes -= len(frame)
                pcm = self._converter.feed(frame or b"", final=frame is None)
                if pcm:
                    sent += len(pcm)
                    await socket.send_json(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(pcm).decode("ascii"),
                        }
                    )
                if frame is None:
                    if sent == 0:
                        return ""
                    if sent < 4800:
                        await socket.send_json(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(bytes(4800 - sent)).decode(
                                    "ascii"
                                ),
                            }
                        )
                    self._committed = True
                    await socket.send_json({"type": "input_audio_buffer.commit"})
                    return await asyncio.wait_for(reader, timeout=60)
        finally:
            if pending is not None:
                pending.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader

    async def _run(self) -> None:
        try:
            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=None, sock_connect=30)
                ) as client,
                client.ws_connect(
                    self._config.address,
                    headers={"Authorization": f"Bearer {self._config.api_key}"},
                    heartbeat=20,
                    max_msg_size=1048576,
                ) as socket,
            ):
                await socket.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "transcription",
                            "audio": {
                                "input": {
                                    "format": {"type": "audio/pcm", "rate": 24000},
                                    "transcription": {"model": self._config.model},
                                    "turn_detection": None,
                                }
                            },
                        },
                    }
                )
                async with asyncio.timeout(30):
                    while (await self._event(socket))["type"] != "session.updated":
                        pass
                self._ready = True
                await self._emit({"type": "ready", "mode": "remote"})
                text = await self._stream(socket)
            await self._emit({"type": "finished", "text": text})
        except asyncio.CancelledError:
            if not self._cancelled:
                raise
        except DomainError as error:
            if not self._cancelled:
                await self._emit(
                    {"type": "error", "code": error.code, "message": str(error)}
                )
        except (aiohttp.ClientError, TimeoutError):
            if not self._cancelled:
                await self._emit(
                    {
                        "type": "error",
                        "code": "voice_remote_connection",
                        "message": "Transcription service connection failed or timed out",
                    }
                )
        finally:
            self._ready = False
            self._finishing = True
