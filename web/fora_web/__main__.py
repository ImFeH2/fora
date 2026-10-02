from __future__ import annotations

import os
from pathlib import Path

from fora.__main__ import WEBUI_DIRECTORY_ENV, main

STATIC = Path(__file__).resolve().parent / "static"


def run() -> int:
    os.environ.setdefault(WEBUI_DIRECTORY_ENV, str(STATIC))
    return main()


if __name__ == "__main__":
    raise SystemExit(run())
