from pathlib import Path
from unittest.mock import Mock

import pytest

from huddol.adapters.voice import native
from huddol.core.errors import DomainError


@pytest.mark.parametrize("failure", ["load", "abi", "symbol", "model"])
def test_windows_directory_closes_after_initialization_failure(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    directory = Mock()
    monkeypatch.setattr(native.sys, "platform", "win32")
    monkeypatch.setattr(
        native.os, "add_dll_directory", Mock(return_value=directory), raising=False
    )
    library = Mock()
    library.voice_abi_version.return_value = 2 if failure == "abi" else 1
    library.voice_engine_version.return_value = b"1.9.2"
    library.voice_open.return_value = None
    if failure == "symbol":
        del library.voice_transcribe
    loader = Mock(return_value=library)
    if failure == "load":
        loader.side_effect = OSError("Library unavailable")
    monkeypatch.setattr(native.ctypes, "CDLL", loader)
    expected = {
        "load": OSError,
        "abi": DomainError,
        "symbol": AttributeError,
        "model": DomainError,
    }[failure]
    with pytest.raises(expected):
        native.Whisper(Path("model.bin"), directory=Path("engine"))
    directory.close.assert_called_once_with()
