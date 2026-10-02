from __future__ import annotations

import json
import queue
import runpy
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fora.runtime_info import RuntimeInfo


@pytest.fixture
def stdio_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "check-python.py"
    check_stdio = runpy.run_path(str(script))["check_stdio"]
    process = SimpleNamespace(stdin=StringIO())
    frames: queue.Queue[Any] = queue.Queue()
    frames.put({"type": "member.updated", "id": 1, "name": "Renamed"})
    frames.put(
        {
            "type": "response",
            "id": 1,
            "result": {"id": 1, "name": "Renamed"},
        }
    )

    def check(ready: Any) -> None:
        @contextmanager
        def kernel(module: str, directory: Path, *arguments: str):
            assert module == "fora"
            assert directory == tmp_path
            assert arguments == ("--transport", "stdio")
            yield process, frames, ready

        monkeypatch.setitem(check_stdio.__globals__, "kernel", kernel)
        check_stdio(tmp_path)

    return check, process, frames


@pytest.mark.parametrize(
    "extra",
    [
        {},
        RuntimeInfo.capture().as_dict(),
        {"text": "additional", "object": {"enabled": True}, "array": [1, "value"]},
        {"null": None, "number": 2, "boolean": False},
        {"version": None, "started_at": {"custom": [False, 1]}},
    ],
)
def test_stdio_ready_accepts_extra_fields(stdio_check, extra: dict[str, Any]) -> None:
    check, process, frames = stdio_check
    check({"type": "ready", "transport": "stdio", **extra})
    assert json.loads(process.stdin.getvalue()) == {
        "id": 1,
        "method": "organization.rename_member",
        "params": {"member_id": 1, "name": "Renamed"},
    }
    assert frames.empty()


@pytest.mark.parametrize(
    "ready",
    [
        None,
        False,
        1,
        "ready",
        [],
        ["ready", "stdio"],
        {},
        {"type": "ready"},
        {"transport": "stdio"},
        {"type": "response", "transport": "stdio"},
        {"type": "ready", "transport": "websocket"},
    ],
)
def test_stdio_ready_rejects_invalid_response(stdio_check, ready: Any) -> None:
    check, process, frames = stdio_check
    with pytest.raises(AssertionError):
        check(ready)
    assert process.stdin.getvalue() == ""
    assert frames.qsize() == 2


@pytest.mark.parametrize("field", ["type", "transport"])
@pytest.mark.parametrize("value", [None, False, 1, 1.5, [], {}, ""])
def test_stdio_ready_rejects_invalid_core_field(
    stdio_check, field: str, value: Any
) -> None:
    check, process, frames = stdio_check
    ready = {"type": "ready", "transport": "stdio", field: value}
    with pytest.raises(AssertionError):
        check(ready)
    assert process.stdin.getvalue() == ""
    assert frames.qsize() == 2
