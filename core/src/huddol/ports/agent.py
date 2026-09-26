from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AgentRun:
    agent_id: int
    sequence: int
    run_id: str
    status: str
    started_at: str
    completed_at: str | None
    messages_json: str
    usage_json: str | None
    error: str | None
    pending_revision: int = -1


@dataclass(frozen=True)
class RunSummary:
    sequence: int
    status: str
    started_at: str
    completed_at: str | None
    usage_json: str | None
    error: str | None


@dataclass(frozen=True)
class HistorySlice:
    sequence: int
    status: str
    started_at: str
    messages: str
    offset: int
    total_length: int


@dataclass(frozen=True)
class WindowState:
    number: int
    since_sequence: int
    reset_at: str | None
    reason: str | None


@dataclass(frozen=True)
class TurnEffect:
    sequence: int
    ordinal: int
    tool: str
    summary: str
    created_at: str


@dataclass(frozen=True)
class AgentLifecycle:
    pause_requested: bool = False
    error: str | None = None
    prepared_sequence: int | None = None


class HistoryStore(Protocol):
    def transaction(self) -> AbstractContextManager[object]: ...

    def pending_revision(self, agent_id: int) -> int: ...

    def repeated_turns(self, agent_id: int, keys: Sequence[tuple[int, int]]) -> int: ...

    def lifecycle(self, agent_id: int) -> AgentLifecycle: ...

    def set_lifecycle(self, agent_id: int, value: AgentLifecycle) -> None: ...

    def consume_preparation(
        self, agent_id: int, sequence: int, keys: Sequence[tuple[int, int]]
    ) -> None: ...

    def new_mentions(
        self, agent_id: int, keys: Sequence[tuple[int, int]]
    ) -> frozenset[tuple[int, int]]: ...

    def prepared_mentions(
        self, agent_id: int, sequence: int
    ) -> frozenset[tuple[int, int]]: ...

    def window(self, agent_id: int) -> WindowState: ...

    def reset_window(self, agent_id: int, reason: str) -> WindowState: ...

    def start_run(
        self,
        agent_id: int,
        run_id: str | None = None,
        reminded: Sequence[tuple[int, int]] = (),
    ) -> AgentRun: ...

    def last_reminder(self, agent_id: int) -> frozenset[tuple[int, int]]: ...

    def no_tool_streak(self, agent_id: int) -> int: ...

    def pause_reason(self, agent_id: int) -> str | None: ...

    def pause_for_safety(self, agent_id: int, reason: str) -> None: ...

    def reset_safety(self, agent_id: int) -> None: ...

    def previously_reminded(
        self, agent_id: int, keys: Sequence[tuple[int, int]]
    ) -> frozenset[tuple[int, int]]: ...

    def save_progress(
        self, agent_id: int, sequence: int, messages_json: str
    ) -> None: ...

    def finish_run(
        self,
        agent_id: int,
        sequence: int,
        *,
        status: str,
        messages_json: str,
        usage_json: str | None = None,
        error: str | None = None,
    ) -> None: ...

    def latest_messages(self, agent_id: int) -> str: ...

    def runs(self, agent_id: int, *, limit: int = 50) -> tuple[AgentRun, ...]: ...

    def run_summaries(
        self, agent_id: int, *, limit: int = 50
    ) -> tuple[RunSummary, ...]: ...

    def record_effect(
        self, agent_id: int, sequence: int, tool: str, summary: str
    ) -> None: ...

    def effects(
        self, agent_id: int, *, sequences: Sequence[int] = ()
    ) -> tuple[TurnEffect, ...]: ...

    def usage_total(self, agent_id: int) -> dict[str, int]: ...

    def mark_interrupted(self) -> int: ...

    def mark_session_start(self) -> int: ...

    def search_runs(
        self, agent_id: int, query: str, *, limit: int = 20
    ) -> tuple[AgentRun, ...]: ...

    def read_run_slice(
        self, agent_id: int, sequence: int, offset: int | None, limit: int
    ) -> HistorySlice | None: ...


class SettingsStore(Protocol):
    def get_settings(self, section: str) -> dict[str, object] | None: ...

    def set_settings(self, section: str, values: dict[str, object]) -> None: ...

    def update_settings(
        self,
        section: str,
        update: Callable[[dict[str, object] | None], dict[str, object]],
    ) -> dict[str, object]: ...
