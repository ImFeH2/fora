from __future__ import annotations

import os
import signal
import stat
import sys
from collections.abc import Callable
from io import TextIOWrapper
from pathlib import Path
from typing import cast


def read_private(path: Path) -> str:
    if os.name == "nt":
        from fora.adapters.windows_files import read_private as windows_read_private

        return windows_read_private(path)
    return path.read_text(encoding="utf-8")


def write_private(path: Path, payload: str) -> None:
    if os.name == "nt":
        from fora.adapters.windows_files import write_private as windows_write_private

        windows_write_private(path, payload)
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
    os.chmod(path, 0o600)


def stdin_is_piped() -> bool:
    try:
        mode = os.fstat(0).st_mode
    except OSError:
        return False
    return stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)


def install_signal_handlers(callback: Callable[[str], None]) -> None:
    for name in ("sigint", "sigterm"):
        number = getattr(signal, name.upper(), None)
        if number is not None:
            signal.signal(number, lambda *_, why=name: callback(why))


def configure_stdio() -> None:
    cast(TextIOWrapper, sys.stdin).reconfigure(encoding="utf-8", errors="strict")
    cast(TextIOWrapper, sys.stdout).reconfigure(
        encoding="utf-8", errors="strict", newline="\n"
    )
