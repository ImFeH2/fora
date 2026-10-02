from __future__ import annotations

from base64 import b64encode
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

from fora.runtime.reminder import TurnRequest


@dataclass(frozen=True)
class ObservabilityConfig:
    enabled: bool
    base_url: str
    public_key: str
    secret_key: str = field(repr=False)
    environment: str = "development"
    capture_content: bool = True

    @classmethod
    def restore(cls, values: dict[str, Any] | None) -> ObservabilityConfig | None:
        if not values or not values.get("enabled"):
            return None
        base_url = str(values.get("base_url") or "")
        public_key = str(values.get("public_key") or "")
        secret_key = str(values.get("secret_key") or "")
        if not base_url or not public_key or not secret_key:
            return None
        return cls(
            enabled=True,
            base_url=base_url,
            public_key=public_key,
            secret_key=secret_key,
            environment=str(values.get("environment") or "development"),
            capture_content=bool(values.get("capture_content", True)),
        )

    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/public/otel/v1/traces"

    def headers(self) -> dict[str, str]:
        credentials = b64encode(
            f"{self.public_key}:{self.secret_key}".encode()
        ).decode()
        return {
            "Authorization": f"Basic {credentials}",
            "x-langfuse-ingestion-version": "4",
        }

    def redacted(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "base_url": self.base_url,
            "environment": self.environment,
            "capture_content": self.capture_content,
            "keys_set": True,
        }


@dataclass(frozen=True)
class TurnTrace:
    agent_id: int
    agent_name: str
    discussion_ids: tuple[int, ...]
    message_count: int

    @classmethod
    def of(cls, request: TurnRequest) -> TurnTrace:
        items = request.reminder.items if request.reminder is not None else ()
        return cls(
            agent_id=request.agent_id,
            agent_name=request.agent_name,
            discussion_ids=tuple(sorted({item.discussion_id for item in items})),
            message_count=len(items),
        )

    def attributes(self) -> dict[str, Any]:
        return {
            "langfuse.trace.name": "Agent turn",
            "langfuse.trace.tags": ["fora", "agent"],
            "langfuse.trace.metadata.agent_id": self.agent_id,
            "langfuse.trace.metadata.agent_name": self.agent_name,
            "langfuse.trace.metadata.discussion_ids": list(self.discussion_ids),
            "langfuse.trace.metadata.message_count": self.message_count,
            "langfuse.session.id": f"fora-agent-{self.agent_id}",
        }


_active: ContextVar[TurnTrace | None] = ContextVar("fora_active_trace", default=None)


@contextmanager
def active_trace(trace: TurnTrace) -> Iterator[None]:
    token = _active.set(trace)
    try:
        yield
    finally:
        _active.reset(token)


def current_trace() -> TurnTrace | None:
    return _active.get()


class Observability(Protocol):
    def instrumentation(self) -> Any: ...

    def shutdown(self) -> None: ...
