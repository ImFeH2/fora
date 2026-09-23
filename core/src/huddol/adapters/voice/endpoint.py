from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from aiohttp import WSMsgType, web

from huddol.adapters.voice.config import VoiceConfig
from huddol.adapters.voice.local import LocalRecording
from huddol.adapters.voice.model import MODEL_REVISION, MODEL_SIZE, ModelStore
from huddol.adapters.voice.remote import RemoteRecording
from huddol.core.errors import DomainError


class VoiceEndpoint:
    def __init__(
        self, data_directory: Path, read_config: Callable[[], dict[str, Any] | None]
    ) -> None:
        self._model = ModelStore(data_directory)
        self._read_config = read_config
        self._local_busy = threading.Lock()

    async def handle(self, connection: web.WebSocketResponse) -> None:
        recording: LocalRecording | RemoteRecording | None = None
        acquired = False
        supervisor: asyncio.Task[None] | None = None
        terminal: dict[str, Any] | None = None
        download: asyncio.Task[None] | None = None
        download_cancel = threading.Event()
        loop = asyncio.get_running_loop()

        async def emit(event: dict[str, Any]) -> None:
            if not connection.closed:
                await connection.send_json(event)

        def progress(received: int, total: int) -> None:
            if not download_cancel.is_set():
                asyncio.run_coroutine_threadsafe(
                    emit(
                        {"type": "model.progress", "received": received, "total": total}
                    ),
                    loop,
                ).result()

        async def download_model() -> None:
            worker = asyncio.create_task(
                asyncio.to_thread(self._model.download, download_cancel, progress)
            )
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                download_cancel.set()
                await worker
                raise
            except DomainError as error:
                if not download_cancel.is_set():
                    await emit(
                        {"type": "error", "code": error.code, "message": str(error)}
                    )
            except (httpx.HTTPError, OSError):
                if not download_cancel.is_set():
                    await emit(
                        {
                            "type": "error",
                            "code": "voice_download",
                            "message": "Model download failed; check network and available storage",
                        }
                    )
            else:
                if not download_cancel.is_set():
                    await emit({"type": "model.downloaded", "revision": MODEL_REVISION})

        async def recording_event(event: dict[str, Any]) -> None:
            nonlocal terminal
            if event["type"] in ("finished", "error"):
                terminal = event
            else:
                await emit(event)

        async def supervise(worker: asyncio.Task[None]) -> None:
            nonlocal acquired, recording
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
                if acquired:
                    self._local_busy.release()
                    acquired = False
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
                    if action == "model.status":
                        await emit(
                            {
                                "type": "model.status",
                                "present": self._model.path.is_file(),
                                "revision": MODEL_REVISION,
                                "bytes": MODEL_SIZE,
                            }
                        )
                    elif action == "model.download":
                        if download is not None and not download.done():
                            raise DomainError(
                                "voice_download_busy",
                                "A model download is already running",
                            )
                        if download is not None:
                            await download
                        download_cancel.clear()
                        download = asyncio.create_task(download_model())
                    elif action == "model.cancel":
                        download_cancel.set()
                        if download is not None:
                            await download
                            download = None
                        await emit({"type": "model.cancelled"})
                    elif action == "start":
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
                        terminal = None
                        if config.mode == "local":
                            candidate = LocalRecording(
                                self._model, rate, recording_event
                            )
                            acquired = self._local_busy.acquire(blocking=False)
                            if not acquired:
                                raise DomainError(
                                    "voice_busy", "Another local recording is active"
                                )
                            recording = candidate
                        else:
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
            download_cancel.set()
            try:
                try:
                    if recording is not None:
                        await recording.close()
                finally:
                    if supervisor is not None:
                        await asyncio.shield(supervisor)
            finally:
                if download is not None:
                    await asyncio.shield(download)
