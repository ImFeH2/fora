from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing
from dataclasses import asdict
from http.client import HTTPConnection
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import psutil
import pytest
from websockets.exceptions import InvalidStatus
from websockets.sync.client import ClientConnection, connect

from fora.core.parameters import AgentParameters

TIMEOUT = 90
SOURCE = str(Path(__file__).resolve().parents[1] / "src")


class Kernel:
    def __init__(
        self,
        data_directory: Path,
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        webui_directory: Path | None = None,
        stdin: int = subprocess.PIPE,
        arguments: Sequence[str] = (),
    ) -> None:
        self.data_directory = data_directory
        self._cwd = cwd
        self._stdin = stdin
        self._arguments = list(arguments)
        self._env = {
            "FORA_DATA_DIR": str(data_directory),
            "FORA_PORT": "0",
            "FORA_WEBUI_DIR": str(webui_directory or data_directory / "no-webui"),
            "PATH": os.environ["PATH"]
            if os.name == "nt"
            else "/usr/bin:/bin:/usr/local/bin",
            "PYTHONPATH": SOURCE,
        }
        if os.name == "nt":
            for name in (
                "SYSTEMROOT",
                "SYSTEMDRIVE",
                "COMSPEC",
                "PATHEXT",
                "TEMP",
                "TMP",
            ):
                if name in os.environ:
                    self._env[name] = os.environ[name]
        self._env.update(env or {})
        self.ready: dict[str, Any] = {}
        self.raw_ready = b""
        self.stdout = b""
        self.stderr = ""
        self.returncode: int | None = None

    def __enter__(self) -> Self:
        self._process = subprocess.Popen(
            [sys.executable, "-m", "fora", *self._arguments],
            stdin=self._stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self._cwd or Path.cwd(),
            env=self._env,
        )
        assert self._process.stdout is not None
        stdout = self._process.stdout
        line: list[bytes] = []
        reader = threading.Thread(target=lambda: line.append(stdout.readline()))
        reader.start()
        reader.join(TIMEOUT)
        if reader.is_alive() or not line or not line[0]:
            self._process.kill()
            reader.join()
            self._finish()
            raise AssertionError(f"kernel did not announce itself: {self.stderr}")
        self.raw_ready = line[0]
        self.ready = json.loads(self.raw_ready.decode("utf-8"))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.returncode is None:
            self.shutdown()

    @property
    def port(self) -> int:
        return int(self.ready["port"])

    @property
    def token(self) -> str:
        return str(self.ready["token"])

    @property
    def pid(self) -> int:
        pid = self._process.pid
        if os.name == "nt" and sys.executable != sys._base_executable:
            children = psutil.Process(pid).children()
            if len(children) == 1:
                pid = children[0].pid
        return pid

    def ws_url(self, token: str | None = None) -> str:
        query = (
            f"?token={self.token if token is None else token}" if token != "" else ""
        )
        return f"ws://127.0.0.1:{self.port}/ws{query}"

    def connect(self, token: str | None = None) -> ClientConnection:
        return connect(self.ws_url(token), open_timeout=TIMEOUT)

    def http(self, path: str) -> tuple[int, str | None, bytes]:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=TIMEOUT)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            return response.status, response.getheader("Content-Type"), response.read()
        finally:
            connection.close()

    def read_frame(self) -> dict[str, Any]:
        assert self._process.stdout is not None
        line = self._process.stdout.readline()
        assert line, self.stderr
        return json.loads(line.decode("utf-8"))

    def send_stdin(self, text: str) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(text.encode("utf-8"))
        self._process.stdin.flush()

    def shutdown(self, *, close_only: bool = False) -> int:
        if not close_only:
            self.send_stdin(json.dumps({"method": "system.shutdown"}) + "\n")
        self._finish()
        assert self.returncode is not None
        return self.returncode

    def signal(self, signum: int) -> int:
        os.kill(self.pid, signum)
        self._finish()
        assert self.returncode is not None
        return self.returncode

    def run_second_instance(self) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, "-m", "fora", *self._arguments],
            cwd=self._cwd or Path.cwd(),
            env=self._env,
            capture_output=True,
            timeout=TIMEOUT,
            check=False,
        )

    def _finish(self) -> None:
        stdout, stderr = self._process.communicate(timeout=TIMEOUT)
        self.stdout = stdout
        self.stderr = stderr.decode("utf-8", errors="replace")
        self.returncode = self._process.returncode


class Client:
    def __init__(self, connection: ClientConnection) -> None:
        self._connection = connection
        self.frames: list[dict[str, Any]] = []

    def send(self, payload: dict[str, Any] | str) -> None:
        self._connection.send(
            payload if isinstance(payload, str) else json.dumps(payload)
        )

    def wait_for(
        self, predicate: Callable[[dict[str, Any]], bool], timeout: float = 60
    ) -> dict[str, Any]:
        while True:
            frame = json.loads(self._connection.recv(timeout=timeout))
            self.frames.append(frame)
            if predicate(frame):
                return frame

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        self.send(request)
        return self.wait_for(
            lambda frame: (
                frame.get("type") == "response" and frame.get("id") == request["id"]
            )
        )


def drive(
    data_directory: Path,
    requests: Sequence[dict[str, Any]],
    *,
    cwd: Path | None = None,
) -> tuple[list[dict[str, Any]], int, str]:
    with Kernel(data_directory, cwd=cwd) as kernel:
        with kernel.connect() as connection:
            client = Client(connection)
            for request in requests:
                client.call(request)
        code = kernel.shutdown()
        assert kernel.stdout == b"", kernel.stdout
        return client.frames, code, kernel.stderr


def response(frames: list[dict[str, Any]], request_id: int) -> dict[str, Any]:
    for frame in frames:
        if frame.get("type") == "response" and frame.get("id") == request_id:
            return frame
    raise AssertionError(f"no response for id {request_id}")


def events(frames: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [frame for frame in frames if frame.get("type") == kind]


@pytest.fixture
def dist(tmp_path: Path) -> Iterator[Path]:
    directory = tmp_path / "dist"
    (directory / "assets").mkdir(parents=True)
    (directory / "index.html").write_text("<!doctype html><title>Fora</title>")
    (directory / "assets" / "app.js").write_text("console.log(1)")
    yield directory


@pytest.mark.parametrize("escaped", [True, False])
def test_frames_use_utf8_independently_of_stdio_defaults(
    tmp_path: Path, escaped: bool
) -> None:
    name = "启动验证😀"
    env = {"PYTHONIOENCODING": "gbk", "PYTHONUTF8": "0"}
    for requests in (
        [
            {
                "id": 1,
                "method": "organization.rename_member",
                "params": {"member_id": 1, "name": name},
            },
            {"id": 2, "method": "organization.get"},
        ],
        [{"id": 2, "method": "organization.get"}],
    ):
        with Kernel(tmp_path / "data", cwd=tmp_path, env=env) as kernel:
            assert not kernel.raw_ready.startswith(b"\xef\xbb\xbf")
            with kernel.connect() as connection:
                client = Client(connection)
                for request in requests:
                    client.send(json.dumps(request, ensure_ascii=escaped))
                    expected = request["id"]
                    client.wait_for(
                        lambda frame, expected=expected: frame.get("id") == expected
                    )
            assert kernel.shutdown() == 0, kernel.stderr
        assert not any("error" in frame for frame in client.frames)
        assert response(client.frames, 2)["result"]["members"][0]["name"] == name


def test_discussion_list_limits_are_optional_for_the_desktop(
    tmp_path: Path, monkeypatch
) -> None:
    from fora.adapters.sqlite.store import SqliteStore

    data = tmp_path / "data"
    store = SqliteStore(data / "fora.sqlite3")
    human = store.create_member("human", "You")
    agent = store.create_member("agent", "Main")
    for day in range(1, 26):
        monkeypatch.setattr(
            "fora.adapters.sqlite.store.now",
            lambda day=day: f"2026-01-{day:02}T00:00:00Z",
        )
        room = store.create_discussion(f"Room {day}", [human.id, agent.id])
        store.append_message(room.id, agent.id, "@You needle")
    monkeypatch.setattr(
        "fora.adapters.sqlite.store.now", lambda: "2026-01-31T00:00:00Z"
    )
    store.append_message(1, agent.id, "@You needle again")
    store.close()
    frames, code, stderr = drive(
        data,
        [
            {"id": 1, "method": "discussion.list"},
            {"id": 2, "method": "discussion.list", "params": {"limit": 2}},
            {"id": 3, "method": "discussion.list", "params": {"limit": 0}},
            {"id": 4, "method": "discussion.search", "params": {"query": "needle"}},
            {"id": 5, "method": "discussion.list"},
        ],
    )
    assert code == 0, stderr
    full = response(frames, 1)["result"]
    assert [item["id"] for item in full] == [1, *range(25, 1, -1)]
    assert sum(item["unread"] for item in full) == 26
    assert response(frames, 2)["result"] == full[:2]
    assert response(frames, 3)["error"]["code"] == "invalid_pagination"
    assert len(response(frames, 4)["result"]) == 26
    assert response(frames, 5)["result"] == full


def test_discussion_management_and_pagination_survive_the_pipe(tmp_path: Path) -> None:
    from fora.adapters.sqlite.store import SqliteStore

    data = tmp_path / "data"
    store = SqliteStore(data / "fora.sqlite3")
    store.create_member("human", "You")
    for name in ("Main", "Other"):
        member = store.create_member("agent", name)
        store.set_agent_state(member.id, "paused")
    store.create_discussion("paged", [1, 2])
    store.create_discussion("retained", [1, 3])
    store.create_discussion("hidden", [2, 3])
    for sender, body in [
        (1, "needle first"),
        (2, "needle second @You @Other"),
        (1, "third"),
        (2, "fourth"),
    ]:
        store.append_message(1, sender, body)
    store.append_message(2, 3, "needle retained")
    store.append_message(3, 2, "needle hidden")
    store.close()
    requests = [
        ("discussion.add_members", {"discussion_id": 1, "member_ids": [3]}),
        ("discussion.read", {"discussion_id": 1, "before": 4, "limit": 2}),
        ("discussion.read", {"discussion_id": 1, "after": 1, "limit": 2}),
        ("discussion.read", {"discussion_id": 1, "message_id": 2, "before": 4}),
        ("discussion.search", {"query": "needle", "sender_id": 2, "discussion_id": 1}),
        ("discussion.archive", {"discussion_id": 1}),
        ("discussion.list", {}),
        ("discussion.read", {"discussion_id": 1, "message_id": 2, "limit": 1}),
        ("discussion.remove_members", {"discussion_id": 1, "member_ids": [2]}),
        ("discussion.unarchive", {"discussion_id": 1}),
        ("discussion.search", {"query": "needle", "sender_id": 2}),
    ]
    frames, code, stderr = drive(
        data,
        [
            {"id": index, "method": method, "params": params}
            for index, (method, params) in enumerate(requests, 1)
        ],
    )
    assert code == 0, stderr
    assert response(frames, 1)["result"]["member_ids"] == [1, 2, 3]
    for request_id in (2, 3):
        assert [
            item["id"] for item in response(frames, request_id)["result"]["messages"]
        ] == [2, 3]
    assert response(frames, 4)["error"]["code"] == "invalid_pagination"
    assert [item["id"] for item in response(frames, 5)["result"]] == [2]
    expected_mentions = [{"member_id": 1, "position": 14, "length": 4}]
    assert response(frames, 5)["result"][0]["mentions"] == expected_mentions
    assert response(frames, 2)["result"]["messages"][0]["mentions"] == expected_mentions
    assert response(frames, 2)["result"]["messages"][1]["mentions"] == []
    assert [item["id"] for item in response(frames, 7)["result"]] == [2]
    assert [item["id"] for item in response(frames, 8)["result"]["messages"]] == [2]
    assert response(frames, 9)["result"]["member_ids"] == [1, 3]
    assert response(frames, 10)["result"]["archived"] is False
    assert [item["discussion_id"] for item in response(frames, 11)["result"]] == [1]
    assert len(events(frames, "discussion.updated")) == 4

    frames, code, stderr = drive(
        data,
        [
            {"id": 1, "method": "discussion.read", "params": {"discussion_id": 1}},
            {"id": 2, "method": "discussion.delete", "params": {"discussion_id": 1}},
            {"id": 3, "method": "discussion.read", "params": {"discussion_id": 1}},
            {"id": 4, "method": "discussion.search", "params": {"query": "needle"}},
            {
                "id": 5,
                "method": "discussion.read",
                "params": {"discussion_id": 1, "section": "members"},
            },
        ],
    )
    assert code == 0, stderr
    assert [item["id"] for item in response(frames, 1)["result"]["members"]] == [1]
    assert response(frames, 1)["result"]["members_info"]["total"] == 2
    assert response(frames, 1)["result"]["members_info"]["complete"] is False
    assert (
        response(frames, 1)["result"]["members_info"]["continuation"]["section"]
        == "members"
    )
    assert [item["id"] for item in response(frames, 5)["result"]["members"]] == [1, 3]
    assert response(frames, 5)["result"]["complete"] is True
    assert len(response(frames, 1)["result"]["messages"]) == 4
    assert response(frames, 2)["error"]["code"] == "unknown_method"
    assert (
        response(frames, 3)["result"]["messages"]
        == response(frames, 1)["result"]["messages"]
    )
    assert response(frames, 3)["result"]["messages"][1]["mentions"] == expected_mentions
    assert [item["discussion_id"] for item in response(frames, 4)["result"]] == [
        1,
        1,
        2,
    ]
    assert events(frames, "discussion.deleted") == []


def test_agent_settings_and_window_survive_the_pipe_and_restart(tmp_path: Path) -> None:
    data = tmp_path / "data"
    values = {
        "context_window_tokens": 12345,
        "exchange_nudge_after": 3,
        "max_concurrent_turns": 2,
        "idle_streak_after": 5,
        "no_tool_turns_before_pause": 4,
        "memory_index_bytes": 256,
        "token_limit": 1000,
    }
    frames, code, stderr = drive(
        data,
        [
            {"id": 1, "method": "settings.get", "params": {"section": "agent"}},
            {
                "id": 2,
                "method": "settings.update",
                "params": {
                    "section": "agent",
                    "values": values,
                },
            },
            {
                "id": 3,
                "method": "organization.create_agent",
                "params": {"name": "Main"},
            },
            {"id": 4, "method": "agent.detail", "params": {"agent_id": 2}},
            {"id": 5, "method": "settings.get", "params": {"section": "model"}},
        ],
    )
    assert code == 0, stderr
    assert response(frames, 1)["result"] == asdict(AgentParameters())
    expected = {**asdict(AgentParameters()), **values}
    assert response(frames, 2)["result"] == expected
    assert response(frames, 4)["result"]["idle"] is False
    assert response(frames, 4)["result"]["token_limit"] == values["token_limit"]
    assert response(frames, 4)["result"]["window"] == {
        "number": 1,
        "since_sequence": 1,
        "reset_at": None,
        "reason": None,
    }
    assert "compaction_threshold" not in response(frames, 5)["result"]
    assert events(frames, "settings.updated") == [
        {"type": "settings.updated", "section": "agent"}
    ]
    frames, code, stderr = drive(
        data,
        [{"id": 1, "method": "settings.get", "params": {"section": "agent"}}],
    )
    assert code == 0, stderr
    assert response(frames, 1)["result"] == expected


@pytest.mark.parametrize(
    ("values", "code"),
    [
        *[
            ({"context_window_tokens": value}, "invalid_parameter")
            for value in (0, -1, 1.5, True, None, "32000")
        ],
        ({"max_concurrent_turns": 0}, "invalid_parameter"),
        ({"token_limit": -1}, "invalid_parameter"),
        ({"unknown": 1}, "invalid_setting"),
    ],
)
def test_invalid_agent_settings_do_not_change_stored_values(
    tmp_path: Path, values, code
) -> None:
    from fora.adapters.sqlite.agent import SqliteAgentStore
    from fora.adapters.sqlite.store import SqliteStore

    data = tmp_path / "data"
    store = SqliteStore(data / "fora.sqlite3")
    original = {"context_window_tokens": 64000}
    SqliteAgentStore(store._db).set_settings("agent", original)
    store.close()
    frames, exit_code, stderr = drive(
        data,
        [
            {
                "id": 1,
                "method": "settings.update",
                "params": {"section": "agent", "values": values},
            },
            {"id": 2, "method": "settings.get", "params": {"section": "agent"}},
        ],
    )
    assert exit_code == 0, stderr
    assert response(frames, 1)["error"]["code"] == code
    assert response(frames, 2)["result"] == {**asdict(AgentParameters()), **original}
    assert events(frames, "settings.updated") == []
    store = SqliteStore(data / "fora.sqlite3")
    try:
        assert SqliteAgentStore(store._db).get_settings("agent") == original
    finally:
        store.close()


def test_idle_threshold_changes_over_the_pipe_and_survives_restart(
    tmp_path: Path,
) -> None:
    from fora.adapters.sqlite.agent import SqliteAgentStore
    from fora.adapters.sqlite.store import SqliteStore

    data = tmp_path / "data"
    store = SqliteStore(data / "fora.sqlite3")
    try:
        store.create_member("human", "You")
        agent_id = store.create_member("agent", "Main").id
        history = SqliteAgentStore(store._db)
        for _ in range(3):
            run = history.start_run(agent_id)
            history.finish_run(
                agent_id, run.sequence, status="completed", messages_json="[]"
            )
    finally:
        store.close()
    frames, code, stderr = drive(
        data,
        [
            {"id": 1, "method": "agent.detail", "params": {"agent_id": agent_id}},
            {
                "id": 2,
                "method": "settings.update",
                "params": {
                    "section": "agent",
                    "values": {"idle_streak_after": 5, "token_limit": 0},
                },
            },
            {"id": 3, "method": "agent.detail", "params": {"agent_id": agent_id}},
        ],
    )
    assert code == 0, stderr
    assert response(frames, 1)["result"]["idle_streak"] == 3
    assert response(frames, 1)["result"]["idle"] is True
    assert response(frames, 2)["result"] == {
        **asdict(AgentParameters()),
        "idle_streak_after": 5,
    }
    assert response(frames, 3)["result"]["idle_streak"] == 3
    assert response(frames, 3)["result"]["idle"] is False
    frames, code, stderr = drive(
        data,
        [{"id": 1, "method": "agent.detail", "params": {"agent_id": agent_id}}],
    )
    assert code == 0, stderr
    assert response(frames, 1)["result"]["idle_streak"] == 3
    assert response(frames, 1)["result"]["idle"] is False


def test_incomplete_model_configuration_is_migrated_over_the_pipe(
    tmp_path: Path,
) -> None:
    from fora.adapters.sqlite.agent import SqliteAgentStore
    from fora.adapters.sqlite.store import SqliteStore

    data = tmp_path / "data"
    store = SqliteStore(data / "fora.sqlite3")
    original = {
        "base_url": "https://example.invalid",
        "api_key": "retained-test-key",
        "compaction_threshold": 64000,
    }
    SqliteAgentStore(store._db).set_settings("model", original)
    store.close()
    frames, code, stderr = drive(
        data,
        [
            {"id": 1, "method": "settings.get", "params": {"section": "model"}},
            {
                "id": 2,
                "method": "settings.update",
                "params": {
                    "section": "model",
                    "values": {
                        "action": "set_defaults",
                        "values": {"model_id": None, "thinking": "default"},
                    },
                },
            },
        ],
    )
    assert code == 0, stderr
    assert "compaction_threshold" not in response(frames, 1)["result"]
    assert "compaction_threshold" not in response(frames, 2)["result"]
    assert "retained-test-key" not in json.dumps(frames) + stderr
    store = SqliteStore(data / "fora.sqlite3")
    try:
        migrated = SqliteAgentStore(store._db).get_settings("model")
        assert migrated["version"] == 2
        assert migrated["models"] == []
        assert migrated["providers"][0]["api_key"] == original["api_key"]
        assert migrated["providers"][0]["base_url"] == original["base_url"]
    finally:
        store.close()


def test_model_api_type_round_trips_and_is_validated_over_the_pipe(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    frames, code, stderr = drive(
        data,
        [
            {"id": 1, "method": "settings.get", "params": {"section": "model"}},
            {
                "id": 2,
                "method": "settings.update",
                "params": {
                    "section": "model",
                    "values": {
                        "action": "save_provider",
                        "values": {
                            "name": "Google",
                            "api_type": "google",
                            "base_url": "https://generativelanguage.googleapis.invalid",
                            "api_key": "SECRET-GOOGLE-KEY",
                        },
                    },
                },
            },
            {
                "id": 3,
                "method": "settings.update",
                "params": {
                    "section": "model",
                    "values": {
                        "action": "save_provider",
                        "values": {"name": "Invalid", "api_type": "made-up"},
                    },
                },
            },
            {"id": 4, "method": "settings.get", "params": {"section": "model"}},
        ],
    )
    assert code == 0, stderr
    assert response(frames, 1)["result"]["providers"] == []
    assert response(frames, 2)["result"]["providers"][0]["api_type"] == "google"
    assert response(frames, 3)["error"]["code"] == "invalid_model_config"
    provider = response(frames, 4)["result"]["providers"][0]
    assert provider["api_type"] == "google"
    assert provider["api_key_set"] is True
    assert provider["base_url"] == "https://generativelanguage.googleapis.invalid"
    assert len(events(frames, "settings.updated")) == 1
    assert "SECRET-GOOGLE-KEY" not in json.dumps(frames) + stderr

    frames, code, stderr = drive(
        data, [{"id": 1, "method": "settings.get", "params": {"section": "model"}}]
    )
    assert code == 0, stderr
    assert response(frames, 1)["result"]["providers"][0] == provider
    assert "SECRET-GOOGLE-KEY" not in json.dumps(frames) + stderr


@pytest.fixture
def local_model():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps(
                {
                    "id": "test-completion",
                    "object": "chat.completion",
                    "created": 1,
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Completed"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield {
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
            "api_key": "test-only",
            "model": "test-model",
        }
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()


def configure_local_model(data: Path, config: dict[str, str]) -> None:
    from fora.adapters.sqlite.agent import SqliteAgentStore
    from fora.adapters.sqlite.store import SqliteStore

    store = SqliteStore(data / "fora.sqlite3")
    try:
        SqliteAgentStore(store._db).set_settings("model", config)
    finally:
        store.close()


def test_missing_model_blocks_turns_with_guidance_and_keeps_kernel_running(
    tmp_path: Path,
) -> None:
    with Kernel(tmp_path / "data") as kernel:
        with kernel.connect() as connection:
            client = Client(connection)
            client.call(
                {
                    "id": 1,
                    "method": "organization.create_agent",
                    "params": {"name": "Main"},
                }
            )
            client.call(
                {
                    "id": 2,
                    "method": "discussion.create",
                    "params": {"topic": "no model yet", "member_ids": [2]},
                }
            )
            client.send(
                {
                    "id": 3,
                    "method": "discussion.send",
                    "params": {"discussion_id": 1, "body": "@Main go"},
                }
            )
            detail = client.call(
                {"id": 4, "method": "agent.detail", "params": {"agent_id": 2}}
            )
            assert detail["result"]["runs"] == []
            assert detail["result"]["state"] == "blocked"
            assert any(
                reason["message"]
                == "Configure a model in Settings before running Agents"
                for reason in detail["result"]["reasons"]
            )
            assert client.call({"id": 5, "method": "ping"})["result"] == {"pong": None}
        assert kernel.shutdown() == 0, kernel.stderr


def test_the_process_announces_itself_once_on_stdout(tmp_path: Path) -> None:
    with Kernel(tmp_path / "data") as kernel:
        assert set(kernel.ready) == {
            "type",
            "port",
            "token",
            "url",
            "version",
            "started_at",
        }
        assert kernel.ready["type"] == "ready"
        assert kernel.ready["url"] == (
            f"http://127.0.0.1:{kernel.port}/?token={kernel.token}"
        )
        with kernel.connect() as connection:
            client = Client(connection)
            organization = client.call({"id": 1, "method": "organization.get"})
            info = client.call({"id": 2, "method": "info.get"})
        with kernel.connect() as connection:
            reconnected = Client(connection)
            info_after_reconnect = reconnected.call({"id": 3, "method": "info.get"})
        assert organization["result"]["human_id"] == 1
        assert info["result"] == {
            key: kernel.ready[key] for key in ("version", "started_at")
        }
        assert info_after_reconnect["result"] == info["result"]
        assert kernel.shutdown() == 0, kernel.stderr
        assert kernel.stdout == b""


def test_runtime_info_changes_after_restart(tmp_path: Path) -> None:
    data = tmp_path / "data"
    with Kernel(data) as first:
        first_info = {key: first.ready[key] for key in ("version", "started_at")}
        with first.connect() as connection:
            info = Client(connection).call({"id": 1, "method": "info.get"})
        assert info["result"] == first_info
        assert first.shutdown() == 0, first.stderr

    with Kernel(data) as second:
        assert second.ready["version"] == first_info["version"]
        assert second.ready["started_at"] != first_info["started_at"]
        with second.connect() as connection:
            info = Client(connection).call({"id": 1, "method": "info.get"})
        assert info["result"] == {
            key: second.ready[key] for key in ("version", "started_at")
        }
        assert second.shutdown() == 0, second.stderr


def test_port_and_token_can_be_pinned_by_the_environment(tmp_path: Path) -> None:
    with Kernel(tmp_path / "data", env={"FORA_PORT": "0"}) as probe:
        port = probe.port
        assert probe.shutdown() == 0
    env = {"FORA_PORT": str(port), "FORA_TOKEN": "pinned-token"}
    with Kernel(tmp_path / "data", env=env) as kernel:
        assert kernel.port == port
        assert kernel.token == "pinned-token"
        with kernel.connect() as connection:
            assert Client(connection).call({"id": 1, "method": "ping"})["result"] == {
                "pong": None
            }
        assert kernel.shutdown() == 0, kernel.stderr


def test_the_data_directory_variable_is_honoured(tmp_path: Path) -> None:
    target = tmp_path / "somewhere" / "else"
    with Kernel(target) as kernel:
        assert (target / "fora.sqlite3").is_file()
        assert kernel.shutdown() == 0, kernel.stderr


def test_startup_has_no_workspace_or_working_directory_setting(
    tmp_path: Path,
) -> None:
    target = tmp_path / "somewhere" / "else"
    frames, code, stderr = drive(
        target,
        [{"id": 1, "method": "settings.get", "params": {"section": "execution"}}],
        cwd=tmp_path,
    )
    assert code == 0, stderr
    assert "working_directory" not in response(frames, 1)["result"]
    assert not (target / "workspace").exists()


def test_concurrent_model_changes_and_member_queries_allow_shutdown(
    tmp_path: Path,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    with Kernel(tmp_path) as kernel:
        ready = threading.Barrier(3)

        def requests(kind: str) -> None:
            with kernel.connect() as connection:
                client = Client(connection)

                def call(method: str, **params: Any) -> Any:
                    frame = client.call({"id": 1, "method": method, "params": params})
                    assert "result" in frame, frame
                    return frame["result"]

                ready.wait(timeout=10)
                for index in range(10):
                    if kind == "settings":
                        call(
                            "settings.update",
                            section="model",
                            values={
                                "action": "set_defaults",
                                "values": {"model_id": None, "thinking": "default"},
                            },
                        )
                    elif kind == "members":
                        member = call(
                            "organization.create_agent", name=f"Concurrent {index}"
                        )
                        call("organization.delete_agent", agent_id=member["id"])
                    else:
                        organization = call("organization.get")
                        assert all(
                            "reasons" in member
                            for member in organization["members"]
                            if member["type"] == "agent"
                        )

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(requests, kind)
                for kind in ("settings", "members", "queries")
            ]
            for future in futures:
                future.result(timeout=TIMEOUT)
        with kernel.connect() as connection:
            client = Client(connection)
            model = client.call(
                {"id": 1, "method": "settings.get", "params": {"section": "model"}}
            )
            assert model["result"]["agent_configs"] == {}
        assert kernel.shutdown() == 0
        assert not (tmp_path / "run.json").exists()
        assert "Lock order violation" not in kernel.stderr


def test_run_file_lives_only_while_the_kernel_runs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    run_file = data / "run.json"
    with Kernel(data) as kernel:
        assert run_file.is_file()
        if os.name != "nt":
            assert oct(run_file.stat().st_mode & 0o777) == "0o600"
        assert json.loads(run_file.read_text()) == {
            "port": kernel.port,
            "token": kernel.token,
            "pid": kernel.pid,
        }
        assert kernel.shutdown() == 0, kernel.stderr
    assert not run_file.exists()


def test_closing_stdin_shuts_the_kernel_down(tmp_path: Path) -> None:
    data = tmp_path / "data"
    with Kernel(data) as kernel:
        assert kernel.shutdown(close_only=True) == 0, kernel.stderr
    assert not (data / "run.json").exists()
    assert (data / "logs" / "fora.log").is_file()
    assert "Shutting down (eof)" in kernel.stderr


@pytest.mark.skipif(os.name != "posix", reason="SIGTERM is a POSIX signal")
def test_sigterm_shuts_the_kernel_down_when_stdin_is_not_a_pipe(
    tmp_path: Path,
) -> None:
    import signal

    data = tmp_path / "data"
    with Kernel(data, stdin=subprocess.DEVNULL) as kernel:
        with kernel.connect() as connection:
            assert Client(connection).call({"id": 1, "method": "ping"})["result"] == {
                "pong": None
            }
        assert (data / "run.json").is_file()
        assert kernel.signal(signal.SIGTERM) == 0, kernel.stderr
    assert not (data / "run.json").exists()
    assert "Shutting down (sigterm)" in kernel.stderr
    assert kernel.stdout == b""


def test_the_token_is_stable_across_starts_of_the_same_data_directory(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    token_file = data / "token"
    with Kernel(data) as first:
        assert token_file.read_text(encoding="utf-8") == first.token
        if os.name != "nt":
            assert oct(token_file.stat().st_mode & 0o777) == "0o600"
        assert first.shutdown() == 0, first.stderr
    assert token_file.is_file()
    with Kernel(data) as second:
        assert second.token == first.token
        assert second.shutdown() == 0, second.stderr
    assert token_file.read_text(encoding="utf-8") == first.token


def test_a_second_instance_on_the_same_data_directory_refuses_to_start(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    run_file = data / "run.json"
    with Kernel(data) as kernel:
        before = run_file.read_bytes()
        second = kernel.run_second_instance()
        assert second.returncode == 2
        assert second.stdout == b""
        assert second.stderr.decode("utf-8").splitlines() == [
            f"Fora is already running on port {kernel.port} for {data}"
        ]
        assert run_file.read_bytes() == before
        with kernel.connect() as connection:
            assert Client(connection).call({"id": 1, "method": "ping"})["result"] == {
                "pong": None
            }
        assert kernel.shutdown() == 0, kernel.stderr
    assert not run_file.exists()


def test_a_stale_run_file_does_not_block_startup(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    run_file = data / "run.json"
    run_file.write_text(json.dumps({"port": 1, "token": "old", "pid": 2**22 + 7}))
    with Kernel(data) as kernel:
        assert json.loads(run_file.read_text())["pid"] == kernel.pid
        assert kernel.shutdown() == 0, kernel.stderr
    assert not run_file.exists()


def test_wrong_or_missing_tokens_get_401_without_an_upgrade(tmp_path: Path) -> None:
    with Kernel(tmp_path / "data") as kernel:
        for token in ("wrong", ""):
            status, content_type, body = kernel.http(
                f"/ws?token={token}" if token else "/ws"
            )
            assert status == 401
            assert content_type == "text/html; charset=utf-8"
            assert b"Access to Fora requires authentication" in body
            assert b"access link provided when Fora starts" in body
            assert b'class="access-notice"' in body
            assert b"MIT License" not in body
            assert b"cnippet-dev" not in body
            assert b"github.com/cnippet-dev" not in body
            assert kernel.token.encode() not in body
            with pytest.raises(InvalidStatus) as rejected, kernel.connect(token):
                pass
            assert rejected.value.response.status_code == 401
        with kernel.connect() as connection:
            assert Client(connection).call({"id": 1, "method": "ping"})["result"] == {
                "pong": None
            }
        assert kernel.shutdown() == 0, kernel.stderr


def test_authentication_diagnosis_and_fixed_application_origins(tmp_path: Path) -> None:
    with Kernel(tmp_path / "data", env={"TMPDIR": str(tmp_path)}) as kernel:
        for origin in (
            "http://localhost:1420",
            "http://tauri.localhost",
            "tauri://localhost",
            "https://example.com",
            "null",
        ):
            for token in (kernel.token, "invalid", "%E6%97%A0%E6%95%88", ""):
                with closing(HTTPConnection("127.0.0.1", kernel.port)) as connection:
                    connection.request(
                        "GET", f"/ws?token={token}", headers={"Origin": origin}
                    )
                    response = connection.getresponse()
                    body = response.read()
                    assert response.status == (204 if token == kernel.token else 401)
                    assert response.getheader("Cache-Control") == "no-store"
                    assert response.getheader("Referrer-Policy") == "no-referrer"
                    assert response.getheader("Vary") == "Origin"
                    assert response.getheader("Access-Control-Allow-Origin") == (
                        origin
                        if origin
                        in (
                            "http://localhost:1420",
                            "http://tauri.localhost",
                            "tauri://localhost",
                        )
                        else None
                    )
                    assert (
                        response.getheader("Access-Control-Allow-Credentials") is None
                    )
                    assert kernel.token.encode() not in body
                    if token == kernel.token:
                        assert body == b""
        assert kernel.shutdown() == 0, kernel.stderr
        assert kernel.token not in kernel.stderr
        assert kernel.token not in (tmp_path / "data/logs/fora.log").read_text()


def test_responses_stay_on_their_own_connection(tmp_path: Path) -> None:
    with Kernel(tmp_path / "data") as kernel:
        with kernel.connect() as first, kernel.connect() as second:
            one, two = Client(first), Client(second)
            one.send({"id": 1, "method": "ping", "params": {"token": "one"}})
            two.send({"id": 1, "method": "ping", "params": {"token": "two"}})
            assert one.wait_for(lambda frame: frame.get("id") == 1)["result"] == {
                "pong": "one"
            }
            assert two.wait_for(lambda frame: frame.get("id") == 1)["result"] == {
                "pong": "two"
            }
            with pytest.raises(TimeoutError):
                one.wait_for(lambda frame: True, timeout=0.5)
            assert len(one.frames) == 1
            assert len(two.frames) == 1
        assert kernel.shutdown() == 0, kernel.stderr


def test_events_are_broadcast_to_every_connection(tmp_path: Path) -> None:
    with Kernel(tmp_path / "data") as kernel:
        with kernel.connect() as first, kernel.connect() as second:
            one, two = Client(first), Client(second)
            created = one.call(
                {
                    "id": 1,
                    "method": "organization.create_agent",
                    "params": {"name": "Main"},
                }
            )
            assert created["result"]["name"] == "Main"
            assert events(one.frames, "member.created")[0]["name"] == "Main"
            echoed = two.wait_for(lambda frame: frame.get("type") == "member.created")
            assert echoed == events(one.frames, "member.created")[0]
            with pytest.raises(TimeoutError):
                two.wait_for(lambda frame: True, timeout=0.5)
        assert kernel.shutdown() == 0, kernel.stderr


def test_the_frontend_is_served_when_it_is_built(tmp_path: Path, dist: Path) -> None:
    with Kernel(tmp_path / "data", webui_directory=dist) as kernel:
        index = dist.joinpath("index.html").read_bytes()
        for path in ("/", "/settings", "/discussions/3?x=1"):
            status, content_type, body = kernel.http(path)
            assert (status, body) == (200, index), path
            assert content_type == "text/html; charset=utf-8"
        status, content_type, body = kernel.http("/assets/app.js")
        assert (status, body) == (200, b"console.log(1)")
        assert content_type is not None
        assert content_type.startswith("text/javascript")
        assert kernel.http("/assets/missing.js")[0] == 404
        assert kernel.http("/assets/")[0] == 200
        assert kernel.http("/../fora.spec")[0] == 404
        assert kernel.shutdown() == 0, kernel.stderr


def test_a_missing_frontend_build_answers_503(tmp_path: Path) -> None:
    with Kernel(tmp_path / "data") as kernel:
        assert kernel.http("/") == (
            503,
            "text/plain; charset=utf-8",
            b"Frontend not built",
        )
        assert kernel.shutdown() == 0, kernel.stderr


def test_discussion_ids_keep_increasing_after_archiving_or_restart(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    requests = [
        ("organization.create_agent", {"name": "Main"}),
        ("discussion.create", {"topic": "First", "member_ids": [2]}),
        ("discussion.create", {"topic": "Second", "member_ids": [2]}),
        ("discussion.archive", {"discussion_id": 2}),
        ("discussion.create", {"topic": "Third", "member_ids": [2]}),
        ("discussion.archive", {"discussion_id": 1}),
        ("discussion.archive", {"discussion_id": 3}),
    ]
    frames, code, stderr = drive(
        data,
        [
            {"id": index, "method": method, "params": params}
            for index, (method, params) in enumerate(requests, 1)
        ],
    )
    assert code == 0, stderr
    assert [response(frames, index)["result"]["id"] for index in (2, 3, 5)] == [1, 2, 3]
    assert response(frames, 6)["result"]["archived"] is True
    assert response(frames, 7)["result"]["archived"] is True
    frames, code, stderr = drive(
        data,
        [
            {
                "id": 1,
                "method": "discussion.create",
                "params": {"topic": "Fourth", "member_ids": [2]},
            },
            {"id": 2, "method": "discussion.read", "params": {"discussion_id": 1}},
        ],
    )
    assert code == 0, stderr
    assert response(frames, 1)["result"]["id"] == 4
    assert response(frames, 2)["result"]["archived"] is True


def test_a_full_conversation_survives_the_real_pipe(tmp_path: Path) -> None:
    frames, code, stderr = drive(
        tmp_path / "data",
        [
            {
                "id": 1,
                "method": "organization.create_agent",
                "params": {"name": "Main"},
            },
            {
                "id": 2,
                "method": "discussion.create",
                "params": {"topic": "ship it", "member_ids": [2]},
            },
            {
                "id": 3,
                "method": "discussion.send",
                "params": {"discussion_id": 1, "body": "@Main please start"},
            },
            {"id": 4, "method": "discussion.list", "params": {}},
            {"id": 5, "method": "discussion.read", "params": {"discussion_id": 1}},
        ],
    )
    assert code == 0, stderr
    assert response(frames, 1)["result"]["name"] == "Main"
    assert response(frames, 2)["result"]["member_ids"] == [1, 2]
    assert response(frames, 3)["result"]["mentioned"] == [2]
    assert response(frames, 4)["result"][0]["topic"] == "ship it"
    assert response(frames, 5)["result"]["messages"][0]["body"] == "@Main please start"
    assert response(frames, 5)["result"]["messages"][0]["mentions"] == [
        {"member_id": 2, "position": 0, "length": 5}
    ]


def test_acknowledgement_ownership_survives_the_real_pipe(tmp_path: Path) -> None:
    from fora.adapters.sqlite.store import SqliteStore

    data = tmp_path / "data"
    data.mkdir()
    store = SqliteStore(data / "fora.sqlite3")
    try:
        store.create_member("human", "You")
        store.create_member("agent", "Helper")
        store.create_member("human", "Reporter")
        store.set_agent_state(2, "paused")
        store.create_discussion("review", [1, 2, 3])
        store.append_message(1, 3, "@You @Helper please review")
        for member_id in (1, 2):
            store.ack(1, [1], member_id)
    finally:
        store.close()

    requests = [
        ("discussion.read", {"discussion_id": 1}),
        (
            "discussion.revoke_ack",
            {"discussion_id": 1, "message_ids": [1], "member_id": 2},
        ),
        ("discussion.read", {"discussion_id": 1}),
        ("discussion.ack", {"discussion_id": 1, "message_ids": [1]}),
        ("discussion.read", {"discussion_id": 1}),
        ("discussion.set_members", {"discussion_id": 1, "member_ids": [2, 3]}),
        ("discussion.revoke_ack", {"discussion_id": 1, "message_ids": [1]}),
    ]
    frames, code, stderr = drive(
        data,
        [
            {"id": index, "method": method, "params": params}
            for index, (method, params) in enumerate(requests, 1)
        ],
    )
    assert code == 0, stderr
    assert response(frames, 1)["result"]["acknowledged"] == [1]
    assert response(frames, 2)["result"]["revoked"] == 1
    assert response(frames, 3)["result"]["acknowledged"] == []
    assert response(frames, 3)["result"]["awaiting_ack"] == [1]
    assert response(frames, 5)["result"]["acknowledged"] == [1]
    assert response(frames, 7)["error"]["code"] == "not_a_member"
    assert len(events(frames, "mention.revoked")) == 1
    store = SqliteStore(data / "fora.sqlite3")
    try:
        assert store.acknowledged(1, 2) == (1,)
        assert store.acknowledged(1, 1) == (1,)
    finally:
        store.close()


def test_membership_changes_preserve_history_and_pending(tmp_path: Path) -> None:
    from fora.adapters.sqlite.store import SqliteStore

    data = tmp_path / "data"
    setup = [
        ("organization.create_agent", {"name": "Helper"}),
        ("organization.pause_agent", {"agent_id": 2}),
        ("discussion.create", {"topic": "review", "member_ids": [2]}),
        ("discussion.send", {"discussion_id": 1, "body": "@Helper review this"}),
    ]
    with Kernel(data) as kernel:
        with kernel.connect() as connection:
            client = Client(connection)
            for index, (method, params) in enumerate(setup, 1):
                client.call({"id": index, "method": method, "params": params})
            for index, member_ids in enumerate(([1], [1, 2], [2])):
                request_id = 5 + index * 3
                first = len(client.frames)
                updated = client.call(
                    {
                        "id": request_id,
                        "method": "discussion.set_members",
                        "params": {"discussion_id": 1, "member_ids": member_ids},
                    }
                )
                read = client.call(
                    {
                        "id": request_id + 1,
                        "method": "discussion.read",
                        "params": {"discussion_id": 1},
                    }
                )
                listed = client.call(
                    {"id": request_id + 2, "method": "discussion.list"}
                )
                assert updated["result"]["member_ids"] == member_ids
                if 1 in member_ids:
                    detail = read["result"]
                    assert [member["id"] for member in detail["members"]] == member_ids
                    assert detail["messages"][0]["body"] == "@Helper review this"
                else:
                    assert read["error"]["code"] == "not_a_member"
                    assert listed["result"] == []
                assert len(events(client.frames[first:], "discussion.updated")) == 1
                with closing(SqliteStore(data / "fora.sqlite3")) as store:
                    assert store.message_count(1) == 1
                    assert [item.message_id for item in store.pending(2)] == (
                        [1] if 2 in member_ids else []
                    )
        assert kernel.shutdown() == 0, kernel.stderr
        assert kernel.stdout == b"", kernel.stdout

    frames, code, stderr = drive(
        data,
        [
            {"id": 1, "method": "discussion.read", "params": {"discussion_id": 1}},
            {"id": 2, "method": "discussion.list"},
        ],
    )
    assert code == 0, stderr
    assert response(frames, 1)["error"]["code"] == "not_a_member"
    assert response(frames, 2)["result"] == []
    with closing(SqliteStore(data / "fora.sqlite3")) as store:
        discussion = store.get_discussion(1)
        assert discussion is not None
        assert discussion.member_ids == frozenset({2})
        assert store.message_count(1) == 1
        assert [item.message_id for item in store.pending(2)] == [1]


def test_state_survives_a_restart(tmp_path: Path) -> None:
    data = tmp_path / "data"
    drive(
        data,
        [
            {
                "id": 1,
                "method": "organization.create_agent",
                "params": {"name": "Main"},
            },
            {
                "id": 2,
                "method": "discussion.create",
                "params": {"topic": "persisted", "member_ids": [2]},
            },
            {
                "id": 3,
                "method": "discussion.send",
                "params": {"discussion_id": 1, "body": "@Main remember this"},
            },
        ],
    )

    frames, code, stderr = drive(
        data, [{"id": 1, "method": "discussion.read", "params": {"discussion_id": 1}}]
    )
    assert code == 0, stderr
    result = response(frames, 1)["result"]
    assert result["topic"] == "persisted"
    assert result["messages"][0]["body"] == "@Main remember this"
    assert result["awaiting_ack"] == []


def test_errors_arrive_as_structured_responses(tmp_path: Path) -> None:
    frames, code, _ = drive(
        tmp_path / "data",
        [
            {"id": 1, "method": "organization.create_agent", "params": {"name": "  "}},
            {"id": 2, "method": "no.such.method", "params": {}},
        ],
    )
    assert code == 0
    assert response(frames, 1)["error"]["code"] == "invalid_name"
    assert response(frames, 2)["error"]["code"] == "unknown_method"


def test_malformed_frames_do_not_kill_the_process(tmp_path: Path) -> None:
    with Kernel(tmp_path / "data") as kernel:
        with kernel.connect() as connection:
            client = Client(connection)
            client.send("not json at all")
            client.wait_for(lambda frame: frame.get("code") == "invalid_frame")
            client.send({"id": 1, "method": "system.shutdown"})
            refused = client.wait_for(lambda frame: frame.get("id") == 1)
            assert refused["error"]["code"] == "internal_method"
            got = client.call({"id": 2, "method": "organization.get", "params": {}})
            assert got["result"]["human_id"] == 1
        assert kernel.shutdown() == 0, kernel.stderr


def test_unusable_write_directories_are_reported_not_fatal(tmp_path: Path) -> None:
    data = tmp_path / "data"
    drive(
        data,
        [
            {
                "id": 1,
                "method": "settings.update",
                "params": {
                    "section": "execution",
                    "values": {"write_directories": [str(tmp_path)]},
                },
            }
        ],
    )
    import sqlite3

    connection = sqlite3.connect(data / "fora.sqlite3")
    connection.execute(
        "UPDATE settings SET values_json = ? WHERE section = 'execution'",
        (json.dumps({"write_directories": ["relative/bad"]}),),
    )
    connection.commit()
    connection.close()

    frames, code, stderr = drive(
        data, [{"id": 1, "method": "settings.get", "params": {"section": "execution"}}]
    )
    assert code == 0, stderr
    status = response(frames, 1)["result"]
    assert status["unusable_write_directories"] == [
        {"path": "relative/bad", "reason": "invalid_directory"}
    ]


def test_interrupted_turns_are_marked_on_the_next_start(tmp_path: Path) -> None:
    data = tmp_path / "data"
    drive(
        data,
        [{"id": 1, "method": "organization.create_agent", "params": {"name": "Main"}}],
    )
    import sqlite3

    connection = sqlite3.connect(data / "fora.sqlite3")
    connection.execute(
        "INSERT INTO agent_runs (agent_id, sequence, run_id, status, started_at,"
        " messages_json) VALUES (2, 1, 'r1', 'running', '2026-01-01T00:00:00Z', '[]')"
    )
    connection.commit()
    connection.close()

    _frames, code, stderr = drive(data, [])
    assert code == 0, stderr

    connection = sqlite3.connect(data / "fora.sqlite3")
    status = connection.execute(
        "SELECT status FROM agent_runs WHERE agent_id = 2"
    ).fetchone()[0]
    connection.close()
    assert status == "interrupted"


def test_startup_recovers_running_history_and_lifecycle_state(
    tmp_path: Path,
) -> None:
    from fora.adapters.sqlite.agent import SqliteAgentStore
    from fora.adapters.sqlite.store import SqliteStore
    from fora.ports.agent import AgentLifecycle

    data = tmp_path / "data"
    store = SqliteStore(data / "fora.sqlite3")
    try:
        history = SqliteAgentStore(store._db)
        store.create_member("human", "You")
        store.create_member("agent", "PausedRun")
        store.create_member("agent", "InvalidHistory")

        paused_run = history.start_run(2)
        partial = (
            '[{"kind":"response","parts":[{"part_kind":"text","content":"partial"}]}]'
        )
        history.save_progress(2, paused_run.sequence, partial)
        history.set_lifecycle(2, AgentLifecycle(pause_requested=True))
        store.set_agent_state(2, "running")

        invalid_run = history.start_run(3)
        history.save_progress(3, invalid_run.sequence, "{")
        store.set_agent_state(3, "running")
    finally:
        store.close()

    with Kernel(data) as kernel:
        with kernel.connect() as connection:
            assert Client(connection).call({"id": 1, "method": "ping"})["result"] == {
                "pong": None
            }
        assert kernel.shutdown() == 0, kernel.stderr

    with closing(sqlite3.connect(data / "fora.sqlite3")) as connection:
        assert connection.execute(
            "SELECT status, messages_json FROM agent_runs WHERE agent_id = 2"
        ).fetchone() == ("interrupted", partial)
        assert connection.execute(
            "SELECT status, messages_json FROM agent_runs WHERE agent_id = 3"
        ).fetchone() == ("running", "{")
        assert connection.execute(
            "SELECT state FROM members WHERE id = 2"
        ).fetchone() == ("paused",)
        assert connection.execute(
            "SELECT state FROM members WHERE id = 3"
        ).fetchone() == ("idle",)
        assert connection.execute(
            "SELECT pause_requested FROM agent_lifecycle WHERE agent_id = 2"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT pause_reason FROM agent_safety WHERE agent_id = 3"
        ).fetchone() == ("history_invalid",)
        assert connection.execute(
            "SELECT agent_id, start_after FROM agent_sessions ORDER BY agent_id"
        ).fetchall() == [(2, 1), (3, 1)]


def wait_for_persisted_turn(data: Path, agent_id: int, sequence: int) -> None:
    deadline = time.monotonic() + TIMEOUT
    with closing(sqlite3.connect(data / "fora.sqlite3")) as connection:
        while time.monotonic() < deadline:
            row = connection.execute(
                "SELECT status FROM agent_runs WHERE agent_id = ? AND sequence = ?",
                (agent_id, sequence),
            ).fetchone()
            if row is not None and row[0] != "running":
                assert row[0] == "completed"
                return
            time.sleep(0.05)
    raise AssertionError(f"Agent {agent_id} did not finish Turn {sequence}")


def test_restart_preserves_completed_count_and_reminder_identity(
    tmp_path: Path, local_model
) -> None:
    from fora.adapters.sqlite.agent import SqliteAgentStore
    from fora.adapters.sqlite.store import SqliteStore
    from fora.runtime.reminder import build_reminder

    data = tmp_path / "data"
    with closing(SqliteStore(data / "fora.sqlite3")) as store:
        history = SqliteAgentStore(store._db)
        history.set_settings("model", local_model)
        store.create_member("human", "You")
        store.create_member("agent", "Main")
        store.create_discussion("restart", [1, 2])
        store.append_message(1, 1, "@Main keep going")
        run = history.start_run(2, reminded=[(1, 1)])
        history.finish_run(2, run.sequence, status="completed", messages_json="[]")
        first_reminded_at = store._db.execute(
            "SELECT first_reminded_at FROM reminded WHERE agent_id = 2"
        )[0]["first_reminded_at"]

    for _ in range(2):
        with Kernel(data) as kernel:
            wait_for_persisted_turn(data, 2, 3)
            with kernel.connect() as connection:
                detail = Client(connection).call(
                    {"id": 1, "method": "agent.detail", "params": {"agent_id": 2}}
                )["result"]
                assert detail["state"] == "blocked"
                assert detail["pause_reason"] == "repeated_mentions"
                assert len(detail["runs"]) == 3
            assert kernel.shutdown() == 0, kernel.stderr

    with closing(SqliteStore(data / "fora.sqlite3")) as store:
        history = SqliteAgentStore(store._db)
        assert [(run.sequence, run.status) for run in history.runs(2)] == [
            (3, "completed"),
            (2, "completed"),
            (1, "completed"),
        ]
        assert store.pending(2)
        assert (
            store._db.execute(
                "SELECT first_reminded_at FROM reminded WHERE agent_id = 2"
            )[0]["first_reminded_at"]
            == first_reminded_at
        )
        reminder = build_reminder(store, history, 2, "Main")
        assert reminder is not None
        assert [item.previously_reminded for item in reminder.items] == [True]
        assert "[previously reminded]" in reminder.render()


def test_startup_continues_an_interrupted_turn_without_new_mentions(
    tmp_path: Path,
    local_model,
) -> None:
    from fora.adapters.sqlite.agent import SqliteAgentStore
    from fora.adapters.sqlite.store import SqliteStore

    data = tmp_path / "data"
    store = SqliteStore(data / "fora.sqlite3")
    try:
        history = SqliteAgentStore(store._db)
        history.set_settings("model", local_model)
        store.create_member("human", "You")
        store.create_member("agent", "Main")
        room = store.create_discussion("interrupted", [1, 2])
        store.append_message(room.id, 1, "@Main continue")
        run = history.start_run(2, reminded=[(room.id, 1)])
        history.save_progress(
            2,
            run.sequence,
            '[{"kind":"response","parts":[{"part_kind":"text","content":"partial"}]}]',
        )
        store.set_agent_state(2, "running")
    finally:
        store.close()

    with Kernel(data) as kernel:
        wait_for_persisted_turn(data, 2, 4)
        time.sleep(1)
        assert kernel.shutdown() == 0, kernel.stderr

    store = SqliteStore(data / "fora.sqlite3")
    try:
        history = SqliteAgentStore(store._db)
        runs = history.runs(2)
        assert [(run.sequence, run.status) for run in runs] == [
            (4, "completed"),
            (3, "completed"),
            (2, "completed"),
            (1, "interrupted"),
        ]
        assert all(
            json.loads(run.messages_json)[0]["parts"][0]["content"] == "partial"
            for run in runs
        )
        assert store.get_member(2).state == "blocked"
        assert history.pause_reason(2) == "repeated_mentions"
        assert [(item.discussion_id, item.message_id) for item in store.pending(2)] == [
            (1, 1)
        ]
    finally:
        store.close()


def test_a_restart_starts_only_agents_with_unhandled_mentions(
    tmp_path: Path, local_model
) -> None:
    from fora.adapters.sqlite.agent import SqliteAgentStore
    from fora.adapters.sqlite.store import SqliteStore

    data = tmp_path / "data"
    data.mkdir()
    store = SqliteStore(data / "fora.sqlite3")
    agent_store = SqliteAgentStore(store._db)
    try:
        store.create_member("human", "You")
        store.create_member("agent", "Main")
        store.create_member("agent", "Helper")
        store.create_member("agent", "Safety")
        store.create_member("agent", "Manual")
        store.create_member("agent", "Spent")
        store.create_member("agent", "Idle")
        agent_store.set_settings("model", local_model)
        store.create_discussion("work", [1, 2, 3, 4, 5, 6])
        store.append_message(1, 1, "@Main @Helper @Safety @Manual @Spent please help")
        store.ack(1, [1], 3)
        agent_store.pause_for_safety(4, "runtime_error")
        store.set_agent_state(5, "paused")
        agent_store.set_settings("agent", {"token_limit": 10})
        spent = agent_store.start_run(6)
        agent_store.finish_run(
            6,
            spent.sequence,
            status="completed",
            messages_json="[]",
            usage_json='{"input_tokens": 100, "output_tokens": 0}',
        )
    finally:
        store.close()

    with Kernel(data) as kernel:
        wait_for_persisted_turn(data, 2, 3)
        time.sleep(1)
        with closing(sqlite3.connect(data / "fora.sqlite3")) as connection:
            runs = connection.execute(
                "SELECT agent_id, COUNT(*) FROM agent_runs GROUP BY agent_id"
                " ORDER BY agent_id"
            ).fetchall()
        assert runs == [(2, 3), (6, 1)]
        assert kernel.shutdown() == 0, kernel.stderr


def test_internal_methods_are_refused_over_the_pipe(tmp_path: Path) -> None:
    frames, code, _ = drive(
        tmp_path / "data", [{"id": 1, "method": "system.diagnostics", "params": {}}]
    )
    assert code == 0
    assert response(frames, 1)["error"]["code"] == "internal_method"


def test_agents_wake_and_report_turns_over_the_socket(
    tmp_path: Path, local_model
) -> None:
    configure_local_model(tmp_path / "data", local_model)
    with Kernel(tmp_path / "data") as kernel:
        with kernel.connect() as connection:
            client = Client(connection)
            client.call(
                {
                    "id": 1,
                    "method": "organization.create_agent",
                    "params": {"name": "Main"},
                }
            )
            client.call(
                {
                    "id": 2,
                    "method": "discussion.create",
                    "params": {"topic": "wake up", "member_ids": [2]},
                }
            )
            client.send(
                {
                    "id": 3,
                    "method": "discussion.send",
                    "params": {"discussion_id": 1, "body": "@Main go"},
                }
            )
            started = client.wait_for(lambda frame: frame.get("type") == "turn.started")
            assert started["agent_id"] == 2
            assert started["items"] == 1
            assert started["kind"] == "reminder"
            finished = client.wait_for(
                lambda frame: frame.get("type") == "turn.finished"
            )
            assert finished["agent_id"] == 2
        assert kernel.shutdown() == 0, kernel.stderr


@pytest.mark.parametrize("section", ["model", "observability"])
def test_secrets_never_come_back_over_the_pipe(tmp_path: Path, section: str) -> None:
    values = (
        {
            "api_type": "openai-chat",
            "base_url": "https://example.invalid/v1",
            "model": "m",
            "api_key": "SECRET-API-KEY",
        }
        if section == "model"
        else {
            "enabled": True,
            "base_url": "https://example.invalid",
            "public_key": "SECRET-PUBLIC",
            "secret_key": "SECRET-SECRET",
        }
    )
    frames, code, _ = drive(
        tmp_path / "data",
        [
            {
                "id": 1,
                "method": "settings.update",
                "params": {"section": section, "values": values},
            },
            {"id": 2, "method": "settings.get", "params": {"section": section}},
        ],
    )
    assert code == 0
    rendered = json.dumps(frames)
    assert "SECRET-API-KEY" not in rendered
    assert "SECRET-PUBLIC" not in rendered
    assert "SECRET-SECRET" not in rendered


def test_observability_settings_preserve_keys_across_restarts(tmp_path: Path) -> None:
    from fora.adapters.sqlite.agent import SqliteAgentStore
    from fora.adapters.sqlite.store import SqliteStore

    data = tmp_path / "data"
    credentials = {"public_key": "test-public", "secret_key": "test-secret"}
    frames, code, stderr = drive(
        data,
        [
            {
                "id": 1,
                "method": "settings.update",
                "params": {
                    "section": "observability",
                    "values": {
                        "enabled": True,
                        "base_url": "https://example.invalid",
                        **credentials,
                    },
                },
            }
        ],
    )
    assert code == 0, stderr
    assert response(frames, 1)["result"]["keys_set"] is True

    restarted, code, restart_stderr = drive(
        data,
        [
            {"id": 1, "method": "settings.get", "params": {"section": "observability"}},
            {
                "id": 2,
                "method": "settings.update",
                "params": {
                    "section": "observability",
                    "values": {
                        "enabled": False,
                        "base_url": "https://updated.example.invalid",
                    },
                },
            },
            {"id": 3, "method": "settings.get", "params": {"section": "observability"}},
        ],
    )
    assert code == 0, restart_stderr
    assert response(restarted, 1)["result"] == {
        "enabled": True,
        "base_url": "https://example.invalid",
        "keys_set": True,
    }
    assert response(restarted, 3)["result"] == {
        "enabled": False,
        "base_url": "https://updated.example.invalid",
        "keys_set": True,
    }
    rendered = json.dumps([frames, restarted]) + stderr + restart_stderr
    assert all(key not in rendered for key in credentials.values())
    store = SqliteStore(data / "fora.sqlite3")
    try:
        values = SqliteAgentStore(store._db).get_settings("observability")
        assert values is not None
        assert all(values[name] == key for name, key in credentials.items())
    finally:
        store.close()


def test_unknown_mode_does_not_initialize_business_data(tmp_path: Path) -> None:
    directory = tmp_path / "must-not-exist"
    completed = subprocess.run(
        [sys.executable, "-m", "fora", "--unknown-mode"],
        cwd=tmp_path,
        env={
            **os.environ,
            "FORA_DATA_DIR": str(directory),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode != 0
    assert not directory.exists()


def test_settings_outside_the_execution_contract_keep_the_organization_available(
    tmp_path: Path,
) -> None:
    from fora.adapters.sqlite.agent import SqliteAgentStore
    from fora.adapters.sqlite.store import SqliteStore

    data = tmp_path / "data"
    store = SqliteStore(data / "fora.sqlite3")
    store.create_member("human", "Existing organization")
    SqliteAgentStore(store._db).set_settings(
        "execution", {"directories": {"native": []}}
    )
    store.close()
    frames, code, stderr = drive(
        data,
        [
            {"id": 1, "method": "organization.get"},
            {"id": 2, "method": "settings.get", "params": {"section": "execution"}},
            {
                "id": 3,
                "method": "settings.update",
                "params": {
                    "section": "execution",
                    "values": {"write_directories": []},
                },
            },
        ],
    )
    assert code == 0, stderr
    assert (
        response(frames, 1)["result"]["members"][0]["name"] == "Existing organization"
    )
    assert response(frames, 2)["result"]["error"]
    assert response(frames, 3)["result"]["error"] is None


def test_ping_answers_without_touching_the_domain(tmp_path: Path) -> None:
    frames, code, stderr = drive(
        tmp_path / "data",
        [{"id": 1, "method": "ping", "params": {"token": "abc"}}],
    )
    assert code == 0, stderr
    assert response(frames, 1)["result"] == {"pong": "abc"}


def test_ping_works_without_a_token(tmp_path: Path) -> None:
    frames, code, _ = drive(tmp_path / "data", [{"id": 1, "method": "ping"}])
    assert code == 0
    assert response(frames, 1)["result"] == {"pong": None}


def test_library_trees_and_human_workspace_reads_survive_the_pipe(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    workspace = data / "agents/2/workspace/topics"
    workspace.mkdir(parents=True)
    (workspace / "note.md").write_text("private", encoding="utf-8")
    requests = [
        ("organization.create_agent", {"name": "Main"}),
        ("library.mkdir", {"path": "folder"}),
        ("library.write", {"path": "folder/note.txt", "content": "before\n"}),
        (
            "library.edit",
            {"path": "folder/note.txt", "old_text": "before", "new_text": "after"},
        ),
        ("library.move", {"path": "folder", "destination": "renamed"}),
        ("library.list", {}),
        ("library.read", {"path": "renamed/note.txt"}),
        ("library.delete", {"path": "renamed"}),
        ("library.list", {}),
        ("workspace.list", {"agent_id": 2}),
        ("workspace.read", {"agent_id": 2, "path": "topics/note.md"}),
        ("workspace.read", {"agent_id": 2, "path": "MEMORY.md"}),
        ("agent.detail", {"agent_id": 2}),
        ("workspace.list", {"agent_id": 2, "path": "topics"}),
    ]
    frames, code, stderr = drive(
        data,
        [
            {"id": index, "method": method, "params": params}
            for index, (method, params) in enumerate(requests, 1)
        ],
    )
    assert code == 0, stderr
    assert not any("error" in frame for frame in frames)
    assert response(frames, 2)["result"] == {"path": "folder", "hash": None}
    assert "+after" in response(frames, 4)["result"]["diff"]
    assert [
        (entry["path"], entry["kind"]) for entry in response(frames, 6)["result"]
    ] == [("renamed", "directory"), ("renamed/note.txt", "file")]
    assert response(frames, 7)["result"]["content"] == "after\n"
    assert response(frames, 8)["result"] == {"path": "renamed", "deleted": True}
    assert response(frames, 9)["result"] == []
    workspace_entries = response(frames, 10)["result"]
    assert [(entry["path"], entry["kind"]) for entry in workspace_entries] == [
        ("MEMORY.md", "file"),
        ("topics", "directory"),
        ("topics/note.md", "file"),
    ]
    assert response(frames, 11)["result"]["content"] == "private"
    assert response(frames, 12)["result"]["content"] == ""
    assert response(frames, 13)["result"]["workspace"] == workspace_entries
    assert response(frames, 14)["result"] == workspace_entries[2:]
    assert len(events(frames, "library.updated")) == 6


def test_the_packaging_smoke_sequence_holds(tmp_path: Path) -> None:
    with Kernel(tmp_path / "data") as kernel:
        assert kernel.ready["type"] == "ready"
        with kernel.connect() as connection:
            pong = Client(connection).call(
                {"id": 1, "method": "ping", "params": {"token": "fora-smoke"}}
            )
        assert pong["result"]["pong"] == "fora-smoke"
        assert kernel.shutdown() == 0, kernel.stderr


def test_ui_pagination_read_and_bulk_contract_survive_real_websocket(
    tmp_path: Path,
) -> None:

    data = tmp_path / "ui-pages"
    with Kernel(data) as kernel, kernel.connect() as connection:
        client = Client(connection)
        sequence = 0

        def request(method: str, **params: Any) -> dict[str, Any]:
            nonlocal sequence
            sequence += 1
            response = client.call({"id": sequence, "method": method, "params": params})
            assert "error" not in response, response
            return response["result"]

        agent = request("organization.create_agent", name="Helper")["id"]
        room = request("discussion.create", topic="window", member_ids=[agent])["id"]
        request("discussion.send", discussion_id=room, body="own", mark_read=False)
        page = request("discussion.page", discussion_id=room, entry=True, limit=1)
        assert page["read_through"] == 0
        assert page["first_unread_id"] == 1
        assert page["messages"][0]["id"] == 1
        state = request("discussion.mark_read", discussion_id=room, message_id=1)
        assert state["read_through"] == 1
        assert any(
            frame["type"] == "discussion.read_updated" for frame in client.frames
        )
        assert request(
            "discussion.ack_pending", discussion_id=room, through_message_id=1
        ) == {"acked": 0, "read_through": 1, "pending_count": 0}
        client.send(
            {
                "id": 99,
                "method": "discussion.page",
                "params": {"discussion_id": room, "limit": 101},
            }
        )
        assert (
            client.wait_for(lambda frame: frame.get("id") == 99)["error"]["code"]
            == "invalid_pagination"
        )
    assert kernel.returncode == 0
    with Kernel(data) as restarted, restarted.connect() as connection:
        client = Client(connection)
        result = client.call(
            {
                "id": 1,
                "method": "discussion.page",
                "params": {"discussion_id": room, "entry": True},
            }
        )["result"]
        assert result["read_through"] == 1
        assert result["first_unread_id"] is None
        assert result["messages"][0]["body"] == "own"
