from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from huddol.adapters.voice.audio import AudioConverter, SpeechSegments, SpeechWindow
from huddol.adapters.voice.model import ModelStore
from huddol.adapters.voice.native import Whisper
from huddol.core.errors import DomainError

Emit = Callable[[dict[str, Any]], Awaitable[None]]


class LocalRecording:
    def __init__(self, model: ModelStore, input_rate: int, emit: Emit) -> None:
        self._model = model
        self._emit = emit
        self._converter = AudioConverter(input_rate, 16000)
        self._segments = SpeechSegments()
        self._incoming: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._windows: deque[SpeechWindow] = deque()
        self._input_rate = input_rate
        self._received_samples = 0
        self._retired_samples = 0
        self._limit = input_rate * 16000 * 30
        self._failure: DomainError | None = None
        self._cancel = threading.Event()
        self._engine: Whisper | None = None
        self._finishing = False
        self._ready = False
        self._task: asyncio.Task[None] | None = None

    def start(self) -> asyncio.Task[None]:
        if self._task is not None:
            raise DomainError("voice_state", "Recording has already started")
        self._task = asyncio.create_task(self._run())
        return self._task

    def feed(self, frame: bytes) -> None:
        if self._failure is not None:
            return
        if not self._ready or self._finishing or self._cancel.is_set():
            raise DomainError("voice_state", "Recording is not accepting audio")
        if not frame or len(frame) % 4 or len(frame) > self._input_rate * 4:
            raise DomainError(
                "voice_format", "Expected at most one second of float32 PCM"
            )
        received = self._received_samples + len(frame) // 4
        if received * 16000 - self._retired_samples * self._input_rate > self._limit:
            self._failure = DomainError(
                "voice_capacity", "Audio processing cannot keep up with recording"
            )
            self._cancel.set()
            if self._engine is not None:
                self._engine.cancel()
            self._incoming.put_nowait(None)
            return
        self._received_samples = received
        self._incoming.put_nowait(frame)

    def stop(self) -> None:
        if not self._ready or self._finishing:
            raise DomainError("voice_state", "Recording is not active")
        self._finishing = True
        self._incoming.put_nowait(None)

    async def close(self) -> None:
        self._failure = None
        self._cancel.set()
        if self._engine is not None:
            self._engine.cancel()
        self._incoming.put_nowait(None)
        if self._task is not None:
            await asyncio.shield(self._task)

    def _load(self) -> tuple[Whisper, float, float]:
        verified = self._model.verify(self._cancel)
        if self._cancel.is_set():
            raise DomainError("voice_cancelled", "Model loading cancelled")
        started = time.monotonic()
        engine = Whisper(self._model.path)
        return engine, verified, time.monotonic() - started

    async def _run(self) -> None:
        try:
            loading = asyncio.create_task(asyncio.to_thread(self._load))
            try:
                self._engine, verified, loaded = await asyncio.shield(loading)
            except asyncio.CancelledError:
                self._cancel.set()
                self._engine, verified, loaded = await loading
                raise
            if self._cancel.is_set():
                return
            assert self._engine is not None
            engine = self._engine
            self._ready = True
            await self._emit(
                {
                    "type": "ready",
                    "mode": "local",
                    "verify_seconds": verified,
                    "load_seconds": loaded,
                }
            )
            completed: list[str] = []
            windows = self._windows
            ended = False
            while not self._cancel.is_set():
                if not windows and not ended:
                    frame = await self._incoming.get()
                    ended = self._consume(frame, windows)
                while not self._incoming.empty() and not ended:
                    ended = self._consume(self._incoming.get_nowait(), windows)
                if self._cancel.is_set():
                    break
                if not windows:
                    if ended:
                        await self._emit(
                            {"type": "finished", "text": "".join(completed)}
                        )
                        return
                    continue
                window = windows.popleft()
                inference = asyncio.create_task(
                    asyncio.to_thread(engine.transcribe, window.pcm)
                )
                try:
                    text = await asyncio.shield(inference)
                except asyncio.CancelledError:
                    engine.cancel()
                    await inference
                    raise
                if self._cancel.is_set():
                    break
                self._retire(windows)
                if window.final:
                    completed.append(text)
                    snapshot = "".join(completed)
                else:
                    snapshot = "".join(completed) + text
                await self._emit(
                    {"type": "transcript", "text": snapshot, "final": window.final}
                )
        except DomainError as error:
            if not self._cancel.is_set():
                await self._emit(
                    {"type": "error", "code": error.code, "message": str(error)}
                )
        finally:
            self._ready = False
            self._finishing = True
            try:
                if self._engine is not None:
                    await asyncio.to_thread(self._engine.close)
                    self._engine = None
            finally:
                self._windows.clear()
                while not self._incoming.empty():
                    self._incoming.get_nowait()
                self._segments = SpeechSegments()
                self._converter = AudioConverter(self._input_rate, 16000)
                self._received_samples = 0
                self._retired_samples = 0
            if self._failure is not None:
                await self._emit(
                    {
                        "type": "error",
                        "code": self._failure.code,
                        "message": str(self._failure),
                    }
                )

    def _retire(self, windows: deque[SpeechWindow]) -> None:
        self._retired_samples = min(
            self._segments.retained_from,
            min(
                (window.start_sample for window in windows),
                default=self._segments.retained_from,
            ),
        )

    def _consume(self, frame: bytes | None, windows: deque[SpeechWindow]) -> bool:
        pcm = self._converter.feed(frame or b"", final=frame is None)
        for window in self._segments.feed(pcm, final=frame is None):
            if (
                windows
                and not windows[-1].final
                and windows[-1].segment == window.segment
            ):
                windows.pop()
            windows.append(window)
        self._retire(windows)
        return frame is None
