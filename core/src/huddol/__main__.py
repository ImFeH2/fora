from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import sys
import threading
from contextlib import ExitStack
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from huddol.adapters.host import (
    configure_stdio,
    install_signal_handlers,
    stdin_is_piped,
    write_private,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from huddol.adapters.jsonl.protocol import Dispatcher, Sink

DATA_DIRECTORY_ENV = "HUDDOL_DATA_DIR"
PORT_ENV = "HUDDOL_PORT"
TOKEN_ENV = "HUDDOL_TOKEN"
WEBUI_DIRECTORY_ENV = "HUDDOL_WEBUI_DIR"
DEVELOPMENT_PORT = 2461
ALREADY_RUNNING = 2
WEBSOCKET = "websocket"
STDIO = "stdio"


def data_directory(override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".huddol"


def default_port() -> int:
    return 0 if getattr(sys, "frozen", False) else DEVELOPMENT_PORT


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="huddol", description="Run a Huddol kernel.")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get(DATA_DIRECTORY_ENV) or None,
        help="directory holding the database, logs, agents and library (default: ~/.huddol)",
    )
    parser.add_argument(
        "--transport",
        choices=(WEBSOCKET, STDIO),
        default=WEBSOCKET,
        help="websocket listens on a loopback port; stdio reads requests from stdin (default: websocket)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=os.environ.get(PORT_ENV) or default_port(),
        help="websocket port; 0 selects a free port",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get(TOKEN_ENV) or None,
        help="websocket token (default: the data directory's token file)",
    )
    parser.add_argument(
        "--webui-dir",
        default=os.environ.get(WEBUI_DIRECTORY_ENV) or None,
        help="directory with a built interface to serve beside the websocket endpoint",
    )
    return parser.parse_args(argv)


def load_token(directory: Path, override: str | None) -> str:
    if override:
        return override
    path = directory / "token"
    if path.is_file():
        stored = path.read_text(encoding="utf-8").strip()
        if stored:
            return stored
    token = secrets.token_urlsafe(24)
    write_private(path, token)
    return token


def running_instance(run_file: Path) -> dict[str, Any] | None:
    import psutil

    try:
        payload = json.loads(run_file.read_text(encoding="utf-8"))
        pid = int(payload["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return payload if psutil.pid_exists(pid) else None


class Stop:
    def __init__(self) -> None:
        self._stopped = threading.Event()
        self._reason: list[str] = []

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def stop(self, reason: str) -> None:
        if not self._reason:
            self._reason.append(reason)
        self._stopped.set()

    def wait(self) -> str:
        while not self._stopped.wait(0.5):
            pass
        return self._reason[0]


def configure_logging(directory: Path) -> list[logging.Handler]:
    logs = directory / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        RotatingFileHandler(
            logs / "huddol.log", maxBytes=1 << 20, backupCount=3, encoding="utf-8"
        ),
        logging.StreamHandler(sys.stderr),
    ]
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    return handlers


def write_run_file(path: Path, port: int | None, token: str | None) -> None:
    write_private(path, json.dumps({"port": port, "token": token, "pid": os.getpid()}))


def announce(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def stdio_sink(stop: Stop) -> Sink:
    from huddol.adapters.jsonl.protocol import encode

    lock = threading.Lock()

    def sink(payload: dict[str, Any]) -> None:
        with lock:
            try:
                sys.stdout.write(encode(payload) + "\n")
                sys.stdout.flush()
            except BrokenPipeError:
                stop.stop("eof")

    return sink


def read_requests(
    dispatcher: Dispatcher, sink: Sink, stop: Stop, lock: threading.Lock
) -> None:
    from huddol.adapters.jsonl.protocol import SHUTDOWN_METHOD, parse

    try:
        for line in sys.stdin:
            with lock:
                if stop.stopped:
                    return
                request = parse(line)
                if request is not None and request.method == SHUTDOWN_METHOD:
                    stop.stop("shutdown")
                    return
                dispatcher.receive(line, sink)
    finally:
        stop.stop("eof")


def wait_for_stdin_shutdown(stop: Stop) -> None:
    from huddol.adapters.jsonl.protocol import wait_for_shutdown

    stop.stop(wait_for_shutdown())


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)

    from huddol.adapters.execution.platforms import dispatch_helper

    helper_result = dispatch_helper(raw)
    if helper_result is not None:
        return helper_result

    options = parse_arguments(raw)
    stdio = options.transport == STDIO

    configure_stdio()

    from huddol.adapters.execution.manager import ExecutionManager
    from huddol.adapters.files.tree import DirectoryTree
    from huddol.adapters.files.uploads import DirectoryUploads
    from huddol.adapters.jsonl.api import HUMAN_ID, Api
    from huddol.adapters.jsonl.protocol import Dispatcher
    from huddol.adapters.model.runner import PydanticModelRunner
    from huddol.adapters.sqlite.agent import SqliteAgentStore
    from huddol.adapters.sqlite.store import SqliteStore
    from huddol.adapters.voice.endpoint import VoiceEndpoint
    from huddol.adapters.websocket.server import WebServer, webui_directory
    from huddol.runtime.scheduler import Scheduler
    from huddol.services.uploads import Uploads
    from huddol.tools import Dependencies

    directory = data_directory(options.data_dir)
    run_file = directory / "run.json"
    running = running_instance(run_file)
    if running is not None:
        listening = running.get("port")
        where = f" on port {listening}" if listening is not None else ""
        sys.stderr.write(f"Huddol is already running{where} for {directory}\n")
        return ALREADY_RUNNING
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as resources:
        handlers = configure_logging(directory)
        for handler in handlers:
            resources.callback(handler.close)
            resources.callback(logging.getLogger().removeHandler, handler)
        log = logging.getLogger("huddol")
        announced = False

        def remove_run_file() -> None:
            if announced:
                run_file.unlink(missing_ok=True)

        resources.callback(remove_run_file)
        store = SqliteStore(directory / "huddol.sqlite3")
        resources.callback(store.close)
        agent_store = SqliteAgentStore(store._db)

        uploads = Uploads(store, DirectoryUploads(directory / "uploads"))
        uploads.cleanup(restart=True)
        if store.get_member(HUMAN_ID) is None:
            store.create_member("human", "You")

        def agent_directory_for(member_id: int) -> Path:
            path = directory / "agents" / str(member_id)
            path.mkdir(parents=True, exist_ok=True)
            return path

        execution = ExecutionManager(
            settings=agent_store.get_settings("execution"), tolerant=True
        )
        scheduler: Scheduler | None = None
        dispatch_lock = threading.Lock()

        def stop_execution() -> None:
            with dispatch_lock:
                if scheduler is None:
                    execution.close()
                else:
                    scheduler.stop()

        resources.callback(stop_execution)
        deps = Dependencies(
            store=store,
            history=agent_store,
            settings=agent_store,
            execution=execution,
            agent_directory_for=agent_directory_for,
            uploads=uploads,
            library_tree=DirectoryTree(directory / "library"),
            workspace_tree_for=lambda member_id: DirectoryTree(
                directory / "agents" / str(member_id) / "workspace"
            ),
        )

        dispatcher = Dispatcher()
        scheduler = Scheduler(
            deps,
            PydanticModelRunner(agent_store),
            on_event=lambda name, payload: dispatcher.emit(name, payload),
        )
        scheduler.recover()
        Api(scheduler, dispatcher)

        stop = Stop()
        install_signal_handlers(stop.stop)
        scheduler.start()

        if stdio:
            sink = stdio_sink(stop)
            dispatcher.attach(sink)
            write_run_file(run_file, None, None)
            announced = True
            log.info("Reading stdio requests with data directory %s", directory)
            announce({"type": "ready", "transport": STDIO})
            threading.Thread(
                target=read_requests,
                args=(dispatcher, sink, stop, dispatch_lock),
                name="huddol-stdio",
                daemon=True,
            ).start()
        else:
            token = load_token(directory, options.token)
            server = WebServer(
                dispatcher,
                token,
                webui_directory(options.webui_dir),
                port=options.port,
                uploads=uploads,
            )
            voice = VoiceEndpoint(lambda: agent_store.get_settings("voice"))
            server.add_websocket_route("/voice", voice.handle, max_msg_size=1048576)
            resources.callback(server.stop)
            server.start()
            write_run_file(run_file, server.port, token)
            announced = True
            log.info(
                "Listening on http://127.0.0.1:%d with data directory %s",
                server.port,
                directory,
            )
            announce(
                {
                    "type": "ready",
                    "port": server.port,
                    "token": token,
                    "url": f"http://127.0.0.1:{server.port}/?token={quote(token, safe='')}",
                }
            )
            if stdin_is_piped():
                threading.Thread(
                    target=wait_for_stdin_shutdown, args=(stop,), daemon=True
                ).start()

        reason = stop.wait()
        log.info("Shutting down (%s)", reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
