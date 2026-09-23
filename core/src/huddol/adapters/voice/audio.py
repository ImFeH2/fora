from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import soxr  # type: ignore[import-untyped]
import webrtcvad  # type: ignore[import-untyped]

from huddol.core.errors import DomainError


class AudioConverter:
    def __init__(self, input_rate: int, output_rate: int) -> None:
        if not 8000 <= input_rate <= 192000 or output_rate not in (16000, 24000):
            raise DomainError("voice_format", "Unsupported audio sample rate")
        self._resampler = soxr.ResampleStream(
            input_rate, output_rate, 1, dtype="float32", quality="HQ"
        )
        self._input_rate = input_rate
        self._closed = False

    def feed(self, pcm: bytes, *, final: bool = False) -> bytes:
        if self._closed:
            raise DomainError("voice_state", "Audio conversion has already finished")
        if len(pcm) % 4 or len(pcm) > self._input_rate * 4:
            raise DomainError(
                "voice_format", "Expected at most one second of float32 PCM"
            )
        samples = np.frombuffer(pcm, dtype="<f4")
        if not np.isfinite(samples).all() or np.any(np.abs(samples) > 1.0):
            raise DomainError(
                "voice_format", "PCM samples must be finite and within [-1, 1]"
            )
        converted = self._resampler.resample_chunk(samples, last=final)
        self._closed = final
        return (
            np.clip(np.rint(converted * 32768), -32768, 32767).astype("<i2").tobytes()
        )


@dataclass(frozen=True)
class SpeechWindow:
    segment: int
    pcm: bytes
    final: bool
    start_sample: int


class SpeechSegments:
    sample_rate = 16000
    frame_bytes = 640

    def __init__(self) -> None:
        self._vad = webrtcvad.Vad(2)
        self._pending = bytearray()
        self._preroll: deque[bytes] = deque(maxlen=10)
        self._frames: list[bytes] = []
        self._silent_frames = 0
        self._partial_frames = 0
        self._segment = 0
        self._processed_samples = 0
        self._closed = False

    @property
    def retained_from(self) -> int:
        retained = self._frames if self._frames else self._preroll
        return self._processed_samples - sum(len(frame) // 2 for frame in retained)

    def _window(self, final: bool) -> SpeechWindow:
        return SpeechWindow(
            self._segment, b"".join(self._frames), final, self.retained_from
        )

    def _finish_segment(self) -> SpeechWindow:
        window = self._window(True)
        self._segment += 1
        self._frames.clear()
        self._partial_frames = 0
        self._silent_frames = 0
        return window

    def feed(self, pcm: bytes, *, final: bool = False) -> list[SpeechWindow]:
        if self._closed:
            raise DomainError("voice_state", "Speech segmentation has already finished")
        if len(pcm) % 2:
            raise DomainError("voice_format", "Expected complete PCM16 samples")
        self._pending.extend(pcm)
        windows: list[SpeechWindow] = []
        while len(self._pending) >= self.frame_bytes:
            frame = bytes(self._pending[: self.frame_bytes])
            del self._pending[: self.frame_bytes]
            self._processed_samples += self.frame_bytes // 2
            speech = self._vad.is_speech(frame, self.sample_rate)
            if not self._frames:
                if not speech:
                    self._preroll.append(frame)
                    continue
                self._frames.extend(self._preroll)
                self._preroll.clear()
            self._frames.append(frame)
            self._silent_frames = 0 if speech else self._silent_frames + 1
            self._partial_frames += 1
            if self._silent_frames >= 30 or len(self._frames) >= 1000:
                windows.append(self._finish_segment())
            elif self._partial_frames >= 50:
                windows.append(self._window(False))
                self._partial_frames = 0
        if final:
            self._processed_samples += len(self._pending) // 2
            if self._frames:
                if self._pending:
                    self._frames.append(bytes(self._pending))
                windows.append(self._finish_segment())
            self._pending.clear()
            self._preroll.clear()
            self._closed = True
        return windows
