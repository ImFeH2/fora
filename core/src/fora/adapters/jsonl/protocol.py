from __future__ import annotations

import json
import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TextIO

from fora.core.errors import DomainError

INTERNAL_PREFIX = "system."
SHUTDOWN_METHOD = "system.shutdown"

Sink = Callable[[dict[str, Any]], None]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Request:
    id: int | None
    method: str
    params: dict[str, Any]


def encode(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse(line: str) -> Request | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    method = payload.get("method")
    if not isinstance(method, str) or not method:
        return None
    identifier = payload.get("id")
    params = payload.get("params")
    return Request(
        id=identifier if isinstance(identifier, int) else None,
        method=method,
        params=params if isinstance(params, dict) else {},
    )


class Dispatcher:
    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self._sinks: list[Sink] = []
        self._lock = threading.Lock()
        self.register("ping", lambda params: {"pong": params.get("token")})

    def register(self, method: str, handler: Callable[[dict[str, Any]], Any]) -> None:
        self._handlers[method] = handler

    def methods(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))

    def attach(self, sink: Sink) -> None:
        with self._lock:
            self._sinks.append(sink)

    def detach(self, sink: Sink) -> None:
        with self._lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        frame = {"type": event_type, **(payload or {})}
        with self._lock:
            sinks = list(self._sinks)
        for sink in sinks:
            sink(frame)

    def receive(self, line: str, sink: Sink) -> None:
        text = line.strip()
        if not text:
            return
        request = parse(text)
        if request is None:
            sink({"type": "error", "code": "invalid_frame", "message": "bad JSON"})
            return
        if request.method.startswith(INTERNAL_PREFIX):
            self._fail(request, sink, "internal_method", "Internal Fora method")
            return
        self.handle(request, sink)

    def handle(self, request: Request, sink: Sink) -> None:
        handler = self._handlers.get(request.method)
        if handler is None:
            self._fail(
                request, sink, "unknown_method", f"{request.method} is not a method"
            )
            return
        try:
            result = handler(request.params)
        except DomainError as error:
            self._fail(request, sink, error.code, str(error))
        except (TypeError, ValueError, KeyError) as error:
            self._fail(
                request, sink, "invalid_params", f"{type(error).__name__}: {error}"
            )
        except Exception as error:  # noqa: BLE001
            self._fail(
                request, sink, "internal_error", f"{type(error).__name__}: {error}"
            )
        else:
            if request.id is not None:
                sink({"type": "response", "id": request.id, "result": result})

    def _fail(self, request: Request, sink: Sink, code: str, message: str) -> None:
        if request.id is None:
            sink({"type": "error", "code": code, "message": message})
            return
        sink(
            {
                "type": "response",
                "id": request.id,
                "error": {"code": code, "message": message},
            }
        )


def wait_for_shutdown(stream: TextIO | None = None) -> str:
    source = stream if stream is not None else sys.stdin
    for line in source:
        text = line.strip()
        if not text:
            continue
        request = parse(text)
        if request is not None and request.method == SHUTDOWN_METHOD:
            return "shutdown"
        log.warning("Ignoring stdin input; only %s is accepted", SHUTDOWN_METHOD)
    return "eof"
