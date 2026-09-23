from __future__ import annotations

import ctypes
import os
import sys
import threading
from importlib.metadata import distribution
from pathlib import Path

import numpy as np

from huddol.core.errors import DomainError


def library_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "voice"  # type: ignore[attr-defined]
    return Path(str(distribution("huddol").locate_file("huddol/adapters/voice/lib")))


class Whisper:
    def __init__(self, model: Path, *, directory: Path | None = None) -> None:
        directory = library_directory() if directory is None else directory
        name = {
            "linux": "libvoice_shim.so",
            "darwin": "libvoice_shim.dylib",
            "win32": "voice_shim.dll",
        }[sys.platform]
        self._dll_directory = None
        if sys.platform == "win32":
            self._dll_directory = os.add_dll_directory(str(directory))  # type: ignore[attr-defined]
        self._handle = None
        try:
            self._initialize(model, directory, name)
        finally:
            if not self._handle and self._dll_directory is not None:
                self._dll_directory.close()
                self._dll_directory = None

    def _initialize(self, model: Path, directory: Path, name: str) -> None:
        self._lib = ctypes.CDLL(str(directory / name))
        self._lib.voice_abi_version.argtypes = []
        self._lib.voice_abi_version.restype = ctypes.c_int
        self._lib.voice_engine_version.argtypes = []
        self._lib.voice_engine_version.restype = ctypes.c_char_p
        if (
            self._lib.voice_abi_version() != 1
            or self._lib.voice_engine_version() != b"1.9.2"
        ):
            raise DomainError("voice_abi", "Incompatible voice engine ABI or version")
        self._lib.voice_open.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib.voice_open.restype = ctypes.c_void_p
        self._lib.voice_transcribe.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib.voice_transcribe.restype = ctypes.c_int
        self._lib.voice_text.argtypes = [ctypes.c_void_p]
        self._lib.voice_text.restype = ctypes.c_char_p
        self._lib.voice_cancel.argtypes = [ctypes.c_void_p]
        self._lib.voice_cancel.restype = None
        self._lib.voice_close.argtypes = [ctypes.c_void_p]
        self._lib.voice_close.restype = None
        self._operation = threading.Lock()
        self._lifecycle = threading.Lock()
        error = ctypes.create_string_buffer(1024)
        self._handle = self._lib.voice_open(
            os.fsencode(model),
            os.fsencode(directory),
            min(4, os.cpu_count() or 1),
            error,
            len(error),
        )
        if not self._handle:
            raise DomainError("voice_load", error.value.decode("utf-8"))

    def transcribe(self, pcm: bytes) -> str:
        if len(pcm) % 2:
            raise DomainError("voice_format", "Expected complete PCM16 samples")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        with self._operation:
            if not self._handle:
                raise DomainError("voice_state", "Voice engine has been closed")
            error = ctypes.create_string_buffer(1024)
            status = self._lib.voice_transcribe(
                self._handle,
                samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                len(samples),
                error,
                len(error),
            )
            if status:
                code = "voice_cancelled" if status == 2 else "voice_inference"
                raise DomainError(code, error.value.decode("utf-8"))
            return self._lib.voice_text(self._handle).decode("utf-8").strip()

    def cancel(self) -> None:
        with self._lifecycle:
            if self._handle:
                self._lib.voice_cancel(self._handle)

    def close(self) -> None:
        self.cancel()
        with self._operation, self._lifecycle:
            if self._handle:
                self._lib.voice_close(self._handle)
                self._handle = None
            if self._dll_directory is not None:
                self._dll_directory.close()
                self._dll_directory = None
