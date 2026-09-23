from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from huddol.core.errors import DomainError

MODEL_REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"
MODEL_SIZE = 147951465
MODEL_SHA256 = "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe"
MODEL_URL = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/{MODEL_REVISION}/ggml-base.bin"


class ModelStore:
    def __init__(self, data_directory: Path) -> None:
        self.directory = data_directory / "models" / "whisper"
        self.path = self.directory / MODEL_REVISION / "ggml-base.bin"
        self._download = threading.Lock()

    def verify(self, cancel: threading.Event) -> float:
        started = time.monotonic()
        if not self.path.is_file():
            raise DomainError(
                "voice_model_missing",
                "Download the Whisper base model in Voice settings",
            )
        if self.path.stat().st_size != MODEL_SIZE:
            raise DomainError(
                "voice_model_invalid",
                "Model size verification failed; download it again",
            )
        digest = hashlib.sha256()
        with self.path.open("rb") as source:
            while block := source.read(1024 * 1024):
                if cancel.is_set():
                    raise DomainError("voice_cancelled", "Model verification cancelled")
                digest.update(block)
        if digest.hexdigest() != MODEL_SHA256:
            raise DomainError(
                "voice_model_invalid",
                "Model SHA-256 verification failed; download it again",
            )
        return time.monotonic() - started

    def download(
        self, cancel: threading.Event, progress: Callable[[int, int], None]
    ) -> None:
        if not self._download.acquire(blocking=False):
            raise DomainError(
                "voice_download_busy", "A model download is already running"
            )
        partial = self.directory / f"{MODEL_REVISION}.partial"
        try:
            if partial.exists():
                shutil.rmtree(partial)
            partial.mkdir(parents=True)
            target = partial / "ggml-base.bin"
            digest = hashlib.sha256()
            received = 0
            progress(received, MODEL_SIZE)
            with (
                httpx.Client(follow_redirects=True, timeout=30) as client,
                client.stream("GET", MODEL_URL) as response,
            ):
                response.raise_for_status()
                with target.open("xb") as output:
                    for block in response.iter_bytes(64 * 1024):
                        if cancel.is_set():
                            raise DomainError(
                                "voice_cancelled", "Model download cancelled"
                            )
                        received += len(block)
                        if received > MODEL_SIZE:
                            raise DomainError(
                                "voice_model_invalid",
                                "Downloaded model exceeds the expected size",
                            )
                        output.write(block)
                        digest.update(block)
                        progress(received, MODEL_SIZE)
                    output.flush()
                    os.fsync(output.fileno())
            if cancel.is_set():
                raise DomainError("voice_cancelled", "Model download cancelled")
            if received != MODEL_SIZE or digest.hexdigest() != MODEL_SHA256:
                raise DomainError(
                    "voice_model_invalid",
                    "Downloaded model failed size or SHA-256 verification",
                )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, self.path)
        finally:
            try:
                if partial.exists():
                    shutil.rmtree(partial)
            finally:
                self._download.release()
