from __future__ import annotations

import numpy as np
import soxr  # type: ignore[import-untyped]

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
