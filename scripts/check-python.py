from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from importlib.metadata import requires, version
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import urlopen

from websockets.sync.client import ClientConnection, connect

TIMEOUT = 60
RENAME = {
    "id": 1,
    "method": "organization.rename_member",
    "params": {"member_id": 1, "name": "Renamed"},
}


def read_frames(process: subprocess.Popen[str], frames: queue.Queue[Any]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        frames.put(json.loads(line))


@contextmanager
def kernel(module: str, directory: Path, *arguments: str):
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("FORA_")
    }
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errors:
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-m",
                module,
                "--data-dir",
                str(directory),
                "--port",
                "0",
                *arguments,
            ],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            encoding="utf-8",
        )
        frames: queue.Queue[Any] = queue.Queue()
        reader = threading.Thread(
            target=read_frames, args=(process, frames), daemon=True
        )
        reader.start()
        try:
            yield process, frames, frames.get(timeout=TIMEOUT)
        finally:
            if process.poll() is None:
                assert process.stdin is not None
                process.stdin.close()
                process.wait(timeout=TIMEOUT)
            reader.join(timeout=5)
            errors.seek(0)
            assert process.returncode == 0, errors.read()
            assert not (directory / "run.json").exists()


def status(url: str) -> int:
    try:
        with urlopen(url, timeout=TIMEOUT) as response:
            return response.status
    except HTTPError as error:
        return error.code


def exchange(connection: ClientConnection) -> list[dict[str, Any]]:
    connection.send(json.dumps(RENAME))
    return [json.loads(connection.recv(timeout=TIMEOUT)) for _ in range(2)]


def assert_answered(frames: list[dict[str, Any]]) -> None:
    assert {"type": "member.updated", "id": 1, "name": "Renamed"} in frames, frames
    assert {
        "type": "response",
        "id": 1,
        "result": {"id": 1, "name": "Renamed"},
    } in frames, frames


def check_commands() -> None:
    scripts = Path(sys.executable).parent
    for name in ("fora", "fora-web"):
        executable = scripts / f"{name}.exe"
        path = executable if executable.exists() else scripts / name
        completed = subprocess.run(
            [str(path), "--help"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--data-dir" in completed.stdout
    installed = version("fora")
    assert version("fora-web") == installed
    assert f"fora=={installed}" in (requires("fora-web") or [])


def check_stdio(directory: Path) -> None:
    with kernel("fora", directory, "--transport", "stdio") as (
        process,
        frames,
        ready,
    ):
        assert isinstance(ready, dict), ready
        assert ready.get("type") == "ready", ready
        assert ready.get("transport") == "stdio", ready
        assert process.stdin is not None
        process.stdin.write(json.dumps(RENAME) + "\n")
        process.stdin.flush()
        assert_answered([frames.get(timeout=TIMEOUT) for _ in range(2)])


def check_core(directory: Path) -> None:
    with kernel("fora", directory) as (_process, _frames, ready):
        assert ready["type"] == "ready", ready
        with connect(
            f"ws://127.0.0.1:{ready['port']}/ws?token={ready['token']}",
            open_timeout=TIMEOUT,
        ) as connection:
            assert_answered(exchange(connection))
        base = f"http://127.0.0.1:{ready['port']}"
        assert status(f"{base}/") == 404
        assert status(f"{base}/assets/app.js") == 404


def check_web(directory: Path) -> None:
    with kernel("fora_web", directory) as (_process, _frames, ready):
        assert ready["type"] == "ready", ready
        base = f"http://127.0.0.1:{ready['port']}"
        with urlopen(f"{base}/", timeout=TIMEOUT) as response:
            assert response.status == 200
            body = response.read()
        assert b'<div id="root">' in body, body[:200]
        with connect(
            f"ws://127.0.0.1:{ready['port']}/ws?token={ready['token']}",
            open_timeout=TIMEOUT,
        ) as connection:
            assert_answered(exchange(connection))


def check(directory: Path) -> None:
    check_commands()
    check_stdio(directory / "stdio")
    check_core(directory / "core")
    check_web(directory / "web")


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: check-python.py DIRECTORY\n")
        return 2
    directory = Path(sys.argv[1]).resolve()
    check(directory)
    sys.stdout.write(
        "Installed fora and fora-web passed commands, stdio, WebSocket, "
        "events, and frontend assets.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
