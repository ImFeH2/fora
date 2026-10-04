from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable, Sequence

from pydantic import ValidationError

from fora.adapters.model.config import (
    AgentModelConfig,
    ModelCatalog,
    thinking_options,
)
from fora.adapters.sqlite.store import LockedConnection, first
from fora.core.errors import DomainError
from fora.core.turn import OVERFLOW_CONTEXT_BYTES
from fora.ports.agent import (
    AgentHistoryRun,
    AgentLifecycle,
    AgentMessagePage,
    AgentModelRequest,
    AgentModelRequestSummary,
    AgentRun,
    AgentTextPage,
    HistorySlice,
    ModelRequestHandle,
    RunSummary,
    TurnEffect,
    WindowEvent,
    WindowState,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_safety (
    agent_id INTEGER PRIMARY KEY,
    resumed_after INTEGER NOT NULL DEFAULT 0,
    pause_reason TEXT
);
CREATE TABLE IF NOT EXISTS agent_lifecycle (
    agent_id INTEGER PRIMARY KEY,
    pause_requested INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    prepared_sequence INTEGER
);
CREATE TABLE IF NOT EXISTS preparation_mentions (
    agent_id INTEGER NOT NULL,
    discussion_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    PRIMARY KEY (agent_id, discussion_id, message_id)
);
CREATE TABLE IF NOT EXISTS agent_windows (
    agent_id INTEGER PRIMARY KEY,
    number INTEGER NOT NULL,
    since_sequence INTEGER NOT NULL,
    reset_at TEXT,
    reason TEXT,
    overflow_context TEXT
);
CREATE TABLE IF NOT EXISTS agent_window_events (
    agent_id INTEGER NOT NULL,
    number INTEGER NOT NULL,
    since_sequence INTEGER NOT NULL,
    reset_at TEXT,
    reason TEXT,
    overflow_context TEXT,
    PRIMARY KEY (agent_id, number)
);
CREATE TABLE IF NOT EXISTS agent_sessions (
    agent_id INTEGER PRIMARY KEY,
    start_after INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    section TEXT PRIMARY KEY,
    values_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reminded (
    agent_id INTEGER NOT NULL,
    discussion_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    first_reminded_at TEXT NOT NULL,
    PRIMARY KEY (agent_id, discussion_id, message_id)
);
CREATE TABLE IF NOT EXISTS run_effects (
    agent_id INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    tool TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (agent_id, sequence, ordinal)
);
CREATE TABLE IF NOT EXISTS agent_model_requests (
    agent_id INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    request_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    window_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    streaming INTEGER NOT NULL,
    messages_json TEXT NOT NULL DEFAULT '',
    parameters_json TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    model_json TEXT NOT NULL,
    response_json TEXT,
    input_count INTEGER NOT NULL DEFAULT 0,
    response_count INTEGER NOT NULL DEFAULT 0,
    related_count INTEGER NOT NULL DEFAULT 0,
    input_length INTEGER NOT NULL DEFAULT 0,
    response_length INTEGER NOT NULL DEFAULT 0,
    parameters_length INTEGER NOT NULL DEFAULT 0,
    settings_length INTEGER NOT NULL DEFAULT 0,
    model_length INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    PRIMARY KEY (agent_id, sequence, ordinal)
);
CREATE INDEX IF NOT EXISTS agent_model_requests_summary
ON agent_model_requests (agent_id, sequence, ordinal, status, started_at, completed_at);
CREATE TABLE IF NOT EXISTS agent_history_message_blobs (
    content_hash TEXT PRIMARY KEY,
    content_json TEXT NOT NULL,
    byte_length INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_model_request_messages (
    agent_id INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    channel TEXT NOT NULL,
    position INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY (agent_id, sequence, ordinal, channel, position),
    FOREIGN KEY (content_hash) REFERENCES agent_history_message_blobs(content_hash)
);
CREATE INDEX IF NOT EXISTS agent_model_request_messages_lookup
ON agent_model_request_messages (agent_id, sequence, ordinal, channel, position, content_hash);
"""


class SqliteAgentStore:
    def __init__(self, db: LockedConnection) -> None:
        self._db = db
        self._db.executescript(SCHEMA)
        request_columns = {
            row["name"]
            for row in self._db.execute("PRAGMA table_info(agent_model_requests)")
        }
        request_migrations = {
            "input_count": "INTEGER NOT NULL DEFAULT 0",
            "response_count": "INTEGER NOT NULL DEFAULT 0",
            "related_count": "INTEGER NOT NULL DEFAULT 0",
            "input_length": "INTEGER NOT NULL DEFAULT 0",
            "response_length": "INTEGER NOT NULL DEFAULT 0",
            "parameters_length": "INTEGER NOT NULL DEFAULT 0",
            "settings_length": "INTEGER NOT NULL DEFAULT 0",
            "model_length": "INTEGER NOT NULL DEFAULT 0",
        }
        with self._db:
            for table in ("agent_windows", "agent_window_events"):
                columns = {
                    row["name"]
                    for row in self._db.execute(f"PRAGMA table_info({table})")
                }
                if "overflow_context" not in columns:
                    self._db.execute(
                        f"ALTER TABLE {table} ADD COLUMN overflow_context TEXT"
                    )
            for name, definition in request_migrations.items():
                if name not in request_columns:
                    self._db.execute(
                        f"ALTER TABLE agent_model_requests ADD COLUMN {name} {definition}"
                    )
            self._db.execute(
                "INSERT OR IGNORE INTO agent_window_events "
                "(agent_id, number, since_sequence, reset_at, reason, overflow_context) "
                "SELECT agent_id, number, since_sequence, reset_at, reason, overflow_context "
                "FROM agent_windows"
            )
        self.update_settings(
            "model", lambda values: ModelCatalog.restore(values).model_dump()
        )

    def transaction(self) -> LockedConnection:
        return self._db

    def pending_revision(self, agent_id: int) -> int:
        row = first(
            self._db.execute(
                "SELECT revision FROM pending_revisions WHERE member_id = ?",
                (agent_id,),
            )
        )
        return int(row["revision"]) if row is not None else 0

    def repeated_turns(self, agent_id: int, keys: Sequence[tuple[int, int]]) -> int:
        if not keys:
            return 0
        with self._db:
            row = first(
                self._db.execute(
                    "SELECT COUNT(*) AS total FROM (SELECT 1 FROM agent_runs"
                    " WHERE agent_id = ? AND pending_revision = ?"
                    " AND status = 'completed' AND reminded_json = ?"
                    " AND sequence > COALESCE((SELECT resumed_after FROM agent_safety"
                    " WHERE agent_id = ?), 0) ORDER BY sequence DESC LIMIT 3)",
                    (
                        agent_id,
                        self.pending_revision(agent_id),
                        json.dumps(sorted(set(keys))),
                        agent_id,
                    ),
                )
            )
            assert row is not None
            return int(row["total"])

    def lifecycle(self, agent_id: int) -> AgentLifecycle:
        row = first(
            self._db.execute(
                "SELECT pause_requested, error, prepared_sequence FROM agent_lifecycle WHERE agent_id = ?",
                (agent_id,),
            )
        )
        if row is not None:
            return AgentLifecycle(bool(row[0]), row[1], row[2])
        member = first(
            self._db.execute("SELECT state FROM members WHERE id = ?", (agent_id,))
        )
        return AgentLifecycle(
            pause_requested=member is not None and member[0] == "paused"
        )

    def set_lifecycle(self, agent_id: int, value: AgentLifecycle) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO agent_lifecycle (agent_id, pause_requested, error, prepared_sequence)"
                " VALUES (?, ?, ?, ?) ON CONFLICT (agent_id) DO UPDATE SET"
                " pause_requested = excluded.pause_requested, error = excluded.error,"
                " prepared_sequence = excluded.prepared_sequence",
                (agent_id, value.pause_requested, value.error, value.prepared_sequence),
            )

    def consume_preparation(
        self, agent_id: int, sequence: int, keys: Sequence[tuple[int, int]]
    ) -> None:
        with self._db:
            self._db.executemany(
                "INSERT OR IGNORE INTO preparation_mentions (agent_id, discussion_id, message_id, sequence)"
                " VALUES (?, ?, ?, ?)",
                [
                    (agent_id, discussion_id, message_id, sequence)
                    for discussion_id, message_id in keys
                ],
            )

    def new_mentions(
        self, agent_id: int, keys: Sequence[tuple[int, int]]
    ) -> frozenset[tuple[int, int]]:
        with self._db:
            consumed = {
                (int(row[0]), int(row[1]))
                for row in self._db.execute(
                    "SELECT discussion_id, message_id FROM reminded WHERE agent_id = ?"
                    " UNION SELECT discussion_id, message_id FROM preparation_mentions WHERE agent_id = ?",
                    (agent_id, agent_id),
                )
            }
            return frozenset(keys) - consumed

    def prepared_mentions(
        self, agent_id: int, sequence: int
    ) -> frozenset[tuple[int, int]]:
        return frozenset(
            (int(row[0]), int(row[1]))
            for row in self._db.execute(
                "SELECT discussion_id, message_id FROM preparation_mentions WHERE agent_id = ? AND sequence = ?",
                (agent_id, sequence),
            )
        )

    def _now(self) -> str:
        from fora.adapters.sqlite.store import now

        return now()

    def _run(self, row: sqlite3.Row) -> AgentRun:
        return AgentRun(
            agent_id=int(row["agent_id"]),
            sequence=int(row["sequence"]),
            run_id=str(row["run_id"]),
            status=str(row["status"]),
            started_at=str(row["started_at"]),
            completed_at=row["completed_at"],
            messages_json=str(row["messages_json"]),
            usage_json=row["usage_json"],
            error=row["error"],
            window_number=(
                int(row["window_number"]) if row["window_number"] is not None else None
            ),
            pending_revision=int(row["pending_revision"]),
        )

    def previously_reminded(
        self, agent_id: int, keys: Sequence[tuple[int, int]]
    ) -> frozenset[tuple[int, int]]:
        if not keys:
            return frozenset()
        found: set[tuple[int, int]] = set()
        for discussion_id, message_id in keys:
            row = first(
                self._db.execute(
                    "SELECT 1 FROM reminded WHERE agent_id = ? AND discussion_id = ?"
                    " AND message_id = ?",
                    (agent_id, discussion_id, message_id),
                )
            )
            if row is not None:
                found.add((discussion_id, message_id))
        return frozenset(found)

    def last_reminder(self, agent_id: int) -> frozenset[tuple[int, int]]:
        row = first(
            self._db.execute(
                "SELECT reminded_json FROM agent_runs WHERE agent_id = ?"
                " AND sequence > COALESCE((SELECT resumed_after FROM agent_safety"
                " WHERE agent_id = ?), 0)"
                " AND sequence > COALESCE((SELECT start_after FROM agent_sessions"
                " WHERE agent_id = ?), 0)"
                " AND reminded_json != '[]' ORDER BY sequence DESC LIMIT 1",
                (agent_id, agent_id, agent_id),
            )
        )
        return (
            frozenset((int(d), int(m)) for d, m in json.loads(row["reminded_json"]))
            if row is not None
            else frozenset()
        )

    def no_tool_streak(self, agent_id: int) -> int:
        rows = self._db.execute(
            "SELECT status, usage_json FROM agent_runs WHERE agent_id = ?"
            " AND sequence > COALESCE((SELECT resumed_after FROM agent_safety"
            " WHERE agent_id = ?), 0) ORDER BY sequence DESC",
            (agent_id, agent_id),
        )
        streak = 0
        for row in rows:
            calls = json.loads(row["usage_json"] or "{}").get("tool_calls")
            if type(calls) is int and calls > 0:
                break
            if row["status"] != "completed":
                continue
            if type(calls) is not int or calls != 0:
                break
            streak += 1
        return streak

    def pause_reason(self, agent_id: int) -> str | None:
        row = first(
            self._db.execute(
                "SELECT pause_reason FROM agent_safety WHERE agent_id = ?", (agent_id,)
            )
        )
        return row["pause_reason"] if row is not None else None

    def pause_for_safety(self, agent_id: int, reason: str) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO agent_safety (agent_id, pause_reason) VALUES (?, ?)"
                " ON CONFLICT (agent_id) DO UPDATE SET pause_reason = excluded.pause_reason",
                (agent_id, reason),
            )

    def reset_safety(self, agent_id: int) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO agent_safety (agent_id, resumed_after) VALUES"
                " (?, (SELECT COALESCE(MAX(sequence), 0) FROM agent_runs WHERE agent_id = ?))"
                " ON CONFLICT (agent_id) DO UPDATE SET"
                " resumed_after = excluded.resumed_after, pause_reason = NULL",
                (agent_id, agent_id),
            )

    def start_run(
        self,
        agent_id: int,
        run_id: str | None = None,
        reminded: Sequence[tuple[int, int]] = (),
    ) -> AgentRun:
        row = first(
            self._db.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS v FROM agent_runs WHERE agent_id = ?",
                (agent_id,),
            )
        )
        assert row is not None
        sequence = int(row["v"])
        identifier = run_id or uuid.uuid4().hex
        started = self._now()
        current_window = self.window(agent_id)
        with self._db:
            revision = self.pending_revision(agent_id)
            self._db.execute(
                "INSERT OR IGNORE INTO agent_window_events"
                " (agent_id, number, since_sequence, reset_at, reason)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    agent_id,
                    current_window.number,
                    current_window.since_sequence,
                    current_window.reset_at,
                    current_window.reason,
                ),
            )
            self._db.execute(
                "INSERT INTO agent_runs (agent_id, sequence, run_id, status, started_at,"
                " window_number, last_saved_at, messages_json, reminded_json, pending_revision)"
                " VALUES (?, ?, ?, 'running', ?, ?, ?, '[]', ?, ?)",
                (
                    agent_id,
                    sequence,
                    identifier,
                    started,
                    current_window.number,
                    started,
                    json.dumps(sorted(set(reminded))),
                    revision,
                ),
            )
            self._db.executemany(
                "INSERT OR IGNORE INTO reminded (agent_id, discussion_id, message_id,"
                " first_reminded_at) VALUES (?, ?, ?, ?)",
                [
                    (agent_id, discussion_id, message_id, started)
                    for discussion_id, message_id in reminded
                ],
            )
            self._db.executemany(
                "DELETE FROM preparation_mentions WHERE agent_id = ? AND discussion_id = ? AND message_id = ?",
                [
                    (agent_id, discussion_id, message_id)
                    for discussion_id, message_id in reminded
                ],
            )
        return AgentRun(
            agent_id=agent_id,
            sequence=sequence,
            run_id=identifier,
            status="running",
            started_at=started,
            completed_at=None,
            messages_json="[]",
            usage_json=None,
            error=None,
            window_number=current_window.number,
            pending_revision=revision,
        )

    def save_progress(self, agent_id: int, sequence: int, messages_json: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE agent_runs SET messages_json = ?, last_saved_at = ?"
                " WHERE agent_id = ? AND sequence = ?",
                (messages_json, self._now(), agent_id, sequence),
            )

    def finish_run(
        self,
        agent_id: int,
        sequence: int,
        *,
        status: str,
        messages_json: str,
        usage_json: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._db:
            self._db.execute(
                "UPDATE agent_runs SET status = ?, completed_at = ?, last_saved_at = ?,"
                " messages_json = ?, usage_json = ?, error = ?"
                " WHERE agent_id = ? AND sequence = ?",
                (
                    status,
                    self._now(),
                    self._now(),
                    messages_json,
                    usage_json,
                    error,
                    agent_id,
                    sequence,
                ),
            )
            if status == "interrupted":
                self._db.execute(
                    "UPDATE agent_model_requests SET status = 'interrupted',"
                    " completed_at = ? WHERE agent_id = ? AND sequence = ?"
                    " AND status = 'pending'",
                    (self._now(), agent_id, sequence),
                )

    def _store_message_refs(
        self,
        agent_id: int,
        sequence: int,
        ordinal: int,
        channel: str,
        messages_json: Sequence[str],
    ) -> tuple[int, int]:
        total_bytes = 0
        count = 0
        values = (
            ()
            if messages_json == "[]"
            else (messages_json,)
            if isinstance(messages_json, str)
            else messages_json
        )
        for position, content_json in enumerate(values):
            encoded = content_json.encode("utf-8")
            content_hash = hashlib.sha256(encoded).hexdigest()
            byte_length = len(encoded)
            self._db.execute(
                "INSERT OR IGNORE INTO agent_history_message_blobs"
                " (content_hash, content_json, byte_length) VALUES (?, ?, ?)",
                (content_hash, content_json, byte_length),
            )
            self._db.execute(
                "INSERT INTO agent_model_request_messages"
                " (agent_id, sequence, ordinal, channel, position, content_hash)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (agent_id, sequence, ordinal, channel, position, content_hash),
            )
            total_bytes += byte_length
            count += 1
        return count, total_bytes

    def start_model_request(
        self,
        agent_id: int,
        sequence: int,
        run_id: str,
        window_number: int,
        messages_json: Sequence[str],
        parameters_json: str,
        settings_json: str,
        model_json: str,
        streaming: bool,
    ) -> ModelRequestHandle:
        row = first(
            self._db.execute(
                "SELECT COALESCE(MAX(ordinal), 0) + 1 AS v FROM agent_model_requests"
                " WHERE agent_id = ? AND sequence = ?",
                (agent_id, sequence),
            )
        )
        assert row is not None
        ordinal = int(row["v"])
        request_id = uuid.uuid4().hex
        started = self._now()
        with self._db:
            input_count, input_length = self._store_message_refs(
                agent_id, sequence, ordinal, "input", messages_json
            )
            self._db.execute(
                "INSERT INTO agent_model_requests (agent_id, sequence, ordinal, request_id,"
                " run_id, window_number, status, started_at, streaming, messages_json,"
                " parameters_json, settings_json, model_json, input_count, input_length,"
                " parameters_length, settings_length, model_length)"
                " VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    agent_id,
                    sequence,
                    ordinal,
                    request_id,
                    run_id,
                    window_number,
                    started,
                    int(streaming),
                    parameters_json,
                    settings_json,
                    model_json,
                    input_count,
                    input_length,
                    len(parameters_json.encode("utf-8")),
                    len(settings_json.encode("utf-8")),
                    len(model_json.encode("utf-8")),
                ),
            )
        return ModelRequestHandle(ordinal, request_id)

    def finish_model_request(
        self,
        agent_id: int,
        sequence: int,
        handle: ModelRequestHandle,
        response_json: Sequence[str],
    ) -> None:
        with self._db:
            response_count, response_length = self._store_message_refs(
                agent_id, sequence, handle.ordinal, "response", response_json
            )
            cursor = self._db.execute_cursor(
                "UPDATE agent_model_requests SET status = 'responded', completed_at = ?,"
                " response_json = NULL, response_count = ?, response_length = ?,"
                " error = NULL WHERE agent_id = ? AND sequence = ?"
                " AND ordinal = ? AND request_id = ?",
                (
                    self._now(),
                    response_count,
                    response_length,
                    agent_id,
                    sequence,
                    handle.ordinal,
                    handle.request_id,
                ),
            )
            if cursor.rowcount != 1:
                raise DomainError("not_found", "Model request does not exist")

    def link_model_request(
        self,
        agent_id: int,
        sequence: int,
        handle: ModelRequestHandle,
        messages_json: Sequence[str],
    ) -> None:
        with self._db:
            related_count, _ = self._store_message_refs(
                agent_id, sequence, handle.ordinal, "related", messages_json
            )
            cursor = self._db.execute_cursor(
                "UPDATE agent_model_requests SET related_count = related_count + ?"
                " WHERE agent_id = ? AND sequence = ? AND ordinal = ?"
                " AND request_id = ?",
                (
                    related_count,
                    agent_id,
                    sequence,
                    handle.ordinal,
                    handle.request_id,
                ),
            )
            if cursor.rowcount != 1:
                raise DomainError("not_found", "Model request does not exist")

    def fail_model_request(
        self,
        agent_id: int,
        sequence: int,
        handle: ModelRequestHandle,
        error: str,
    ) -> None:
        with self._db:
            cursor = self._db.execute_cursor(
                "UPDATE agent_model_requests SET status = 'failed', completed_at = ?,"
                " error = ? WHERE agent_id = ? AND sequence = ? AND ordinal = ?"
                " AND request_id = ?",
                (
                    self._now(),
                    error,
                    agent_id,
                    sequence,
                    handle.ordinal,
                    handle.request_id,
                ),
            )
            if cursor.rowcount != 1:
                raise DomainError("not_found", "Model request does not exist")

    def _model_request_summary(self, row: sqlite3.Row) -> AgentModelRequestSummary:
        return AgentModelRequestSummary(
            ordinal=int(row["ordinal"]),
            request_id=str(row["request_id"]),
            run_id=str(row["run_id"]),
            window_number=(
                int(row["window_number"]) if row["window_number"] is not None else None
            ),
            status=str(row["status"]),
            started_at=str(row["started_at"]),
            completed_at=row["completed_at"],
            streaming=bool(row["streaming"]),
            input_length=int(row["input_length"]),
            parameters_length=int(row["parameters_length"]),
            settings_length=int(row["settings_length"]),
            model_length=int(row["model_length"]),
            response_length=int(row["response_length"]),
            input_count=int(row["input_count"]),
            response_count=int(row["response_count"]),
            related_count=int(row["related_count"]),
            error=row["error"],
        )

    def _history_run(self, row: sqlite3.Row) -> AgentHistoryRun:
        return AgentHistoryRun(
            sequence=int(row["sequence"]),
            run_id=str(row["run_id"]),
            status=str(row["status"]),
            started_at=str(row["started_at"]),
            completed_at=row["completed_at"],
            usage_json=row["usage_json"],
            error=row["error"],
            window_number=(
                int(row["window_number"]) if row["window_number"] is not None else None
            ),
            window_reset_at=row["window_reset_at"],
            window_reason=row["window_reason"],
            request_count=int(row["request_count"]),
            last_saved_at=row["last_saved_at"],
        )

    def run_messages(self, agent_id: int, sequence: int) -> str | None:
        row = first(
            self._db.execute(
                "SELECT messages_json FROM agent_runs WHERE agent_id = ? AND sequence = ?",
                (agent_id, sequence),
            )
        )
        return str(row["messages_json"]) if row is not None else None

    def history_run(self, agent_id: int, sequence: int) -> AgentHistoryRun | None:
        row = first(
            self._db.execute(
                "SELECT r.sequence, r.run_id, r.status, r.started_at, r.completed_at,"
                " r.usage_json, r.error, r.last_saved_at, r.window_number,"
                " (SELECT reset_at FROM agent_window_events w"
                "  WHERE w.agent_id = r.agent_id AND w.number = r.window_number"
                "  LIMIT 1) AS window_reset_at,"
                " (SELECT reason FROM agent_window_events w"
                "  WHERE w.agent_id = r.agent_id AND w.number = r.window_number"
                "  LIMIT 1) AS window_reason,"
                " (SELECT COUNT(*) FROM agent_model_requests m"
                "  WHERE m.agent_id = r.agent_id AND m.sequence = r.sequence) AS request_count"
                " FROM agent_runs r WHERE r.agent_id = ? AND r.sequence = ?",
                (agent_id, sequence),
            )
        )
        return self._history_run(row) if row is not None else None

    def history_runs(
        self,
        agent_id: int,
        *,
        before: int | None = None,
        limit: int = 30,
    ) -> tuple[AgentHistoryRun, ...]:
        clause = "" if before is None else " AND r.sequence < ?"
        parameters: tuple[object, ...] = (
            (agent_id,) if before is None else (agent_id, before)
        )
        rows = self._db.execute(
            "SELECT r.sequence, r.run_id, r.status, r.started_at, r.completed_at,"
            " r.usage_json, r.error, r.last_saved_at, r.window_number,"
            " (SELECT reset_at FROM agent_window_events w"
            "  WHERE w.agent_id = r.agent_id AND w.number = r.window_number"
            "  LIMIT 1) AS window_reset_at,"
            " (SELECT reason FROM agent_window_events w"
            "  WHERE w.agent_id = r.agent_id AND w.number = r.window_number"
            "  LIMIT 1) AS window_reason,"
            " (SELECT COUNT(*) FROM agent_model_requests m"
            "  WHERE m.agent_id = r.agent_id AND m.sequence = r.sequence) AS request_count"
            " FROM agent_runs r WHERE r.agent_id = ?"
            f"{clause} ORDER BY r.sequence DESC LIMIT ?",
            (*parameters, limit),
        )
        return tuple(self._history_run(row) for row in rows)

    def model_request_summaries(
        self,
        agent_id: int,
        sequence: int,
        *,
        after: int | None = None,
        limit: int = 30,
    ) -> tuple[AgentModelRequestSummary, ...]:
        clause = "" if after is None else " AND ordinal > ?"
        parameters: tuple[object, ...] = (
            (agent_id, sequence) if after is None else (agent_id, sequence, after)
        )
        rows = self._db.execute(
            "SELECT ordinal, request_id, run_id, window_number, status, started_at,"
            " completed_at, streaming, input_length, parameters_length, settings_length,"
            " model_length, response_length, input_count, response_count, related_count, error"
            " FROM agent_model_requests WHERE agent_id = ? AND sequence = ?"
            f"{clause} ORDER BY ordinal LIMIT ?",
            (*parameters, limit),
        )
        return tuple(self._model_request_summary(row) for row in rows)

    def model_request_summary(
        self, agent_id: int, sequence: int, ordinal: int
    ) -> AgentModelRequestSummary | None:
        row = first(
            self._db.execute(
                "SELECT ordinal, request_id, run_id, window_number, status, started_at,"
                " completed_at, streaming, input_length, parameters_length, settings_length,"
                " model_length, response_length, input_count, response_count, related_count, error"
                " FROM agent_model_requests WHERE agent_id = ? AND sequence = ? AND ordinal = ?",
                (agent_id, sequence, ordinal),
            )
        )
        return self._model_request_summary(row) if row is not None else None

    def model_request(
        self, agent_id: int, sequence: int, ordinal: int
    ) -> AgentModelRequest | None:
        summary = self.model_request_summary(agent_id, sequence, ordinal)
        return AgentModelRequest(summary) if summary is not None else None

    def model_request_field(
        self,
        agent_id: int,
        sequence: int,
        ordinal: int,
        field: str,
        offset: int,
        limit: int,
    ) -> AgentTextPage:
        columns = {
            "parameters": "parameters_json",
            "settings": "settings_json",
            "model": "model_json",
        }
        column = columns.get(field)
        if column is None:
            raise DomainError("invalid_params", "Model request field is invalid")
        if offset < 0:
            raise DomainError("invalid_offset", "Model request field offset is invalid")
        if limit <= 0 or limit > 16 * 1024:
            raise DomainError("invalid_params", "Model request field limit is invalid")
        row = first(
            self._db.execute(
                f"SELECT length(CAST({column} AS BLOB)) AS total_bytes FROM agent_model_requests"
                " WHERE agent_id = ? AND sequence = ? AND ordinal = ?",
                (agent_id, sequence, ordinal),
            )
        )
        if row is None:
            raise DomainError("not_found", "Model request does not exist")
        total_bytes = int(row["total_bytes"])
        if offset > total_bytes:
            raise DomainError(
                "invalid_offset", "Model request field offset exceeds its length"
            )
        start = offset
        value_row = first(
            self._db.execute(
                f"SELECT substr(CAST({column} AS BLOB), ?, ?) AS value"
                " FROM agent_model_requests WHERE agent_id = ? AND sequence = ? AND ordinal = ?",
                (start + 1, limit + 4, agent_id, sequence, ordinal),
            )
        )
        assert value_row is not None
        raw = bytes(value_row["value"] or b"")
        while raw and raw[0] & 0xC0 == 0x80:
            raw = raw[1:]
            start += 1
        value = raw[:limit].decode("utf-8", errors="ignore")
        chunk = value.encode("utf-8")
        end = start + len(chunk)
        return AgentTextPage(field, start, total_bytes, value, end < total_bytes)

    def model_request_messages(
        self,
        agent_id: int,
        sequence: int,
        ordinal: int,
        channel: str,
        offset: int,
        limit: int,
    ) -> AgentMessagePage:
        total_row = first(
            self._db.execute(
                "SELECT COUNT(*) AS total, COALESCE(SUM(b.byte_length), 0) AS total_bytes"
                " FROM agent_model_request_messages r"
                " JOIN agent_history_message_blobs b ON b.content_hash = r.content_hash"
                " WHERE r.agent_id = ? AND r.sequence = ? AND r.ordinal = ? AND r.channel = ?",
                (agent_id, sequence, ordinal, channel),
            )
        )
        assert total_row is not None
        rows = self._db.execute(
            "SELECT b.content_json FROM agent_model_request_messages r"
            " JOIN agent_history_message_blobs b ON b.content_hash = r.content_hash"
            " WHERE r.agent_id = ? AND r.sequence = ? AND r.ordinal = ? AND r.channel = ?"
            " ORDER BY r.position LIMIT ? OFFSET ?",
            (agent_id, sequence, ordinal, channel, limit, offset),
        )
        total = int(total_row["total"])
        return AgentMessagePage(
            channel=channel,
            offset=offset,
            total=total,
            total_bytes=int(total_row["total_bytes"]),
            messages=tuple(str(row["content_json"]) for row in rows),
            has_more=offset + len(rows) < total,
        )

    def model_request_message(
        self,
        agent_id: int,
        sequence: int,
        ordinal: int,
        channel: str,
        position: int,
    ) -> str | None:
        row = first(
            self._db.execute(
                "SELECT b.content_json FROM agent_model_request_messages r"
                " JOIN agent_history_message_blobs b ON b.content_hash = r.content_hash"
                " WHERE r.agent_id = ? AND r.sequence = ? AND r.ordinal = ?"
                " AND r.channel = ? AND r.position = ?",
                (agent_id, sequence, ordinal, channel, position),
            )
        )
        return str(row["content_json"]) if row is not None else None

    def window(self, agent_id: int) -> WindowState:
        row = first(
            self._db.execute(
                "SELECT number, since_sequence, reset_at, reason, overflow_context FROM agent_windows"
                " WHERE agent_id = ?",
                (agent_id,),
            )
        )
        if row is None:
            return WindowState(1, 1, None, None)
        return WindowState(
            int(row["number"]),
            int(row["since_sequence"]),
            row["reset_at"],
            row["reason"],
            row["overflow_context"],
        )

    def reset_window(
        self, agent_id: int, reason: str, overflow_context: str | None = None
    ) -> WindowState:
        if reason != "overflow":
            overflow_context = None
        if (
            overflow_context is not None
            and len(overflow_context.encode("utf-8")) > OVERFLOW_CONTEXT_BYTES
        ):
            raise ValueError("Overflow context exceeds its UTF-8 byte budget")
        current = self.window(agent_id)
        number = current.number + 1
        since_sequence_row = first(
            self._db.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS v FROM agent_runs"
                " WHERE agent_id = ?",
                (agent_id,),
            )
        )
        assert since_sequence_row is not None
        since_sequence = int(since_sequence_row["v"])
        reset_at = self._now()
        with self._db:
            self._db.execute(
                "INSERT INTO agent_window_events"
                " (agent_id, number, since_sequence, reset_at, reason, overflow_context)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (agent_id, number, since_sequence, reset_at, reason, overflow_context),
            )
            self._db.execute(
                "INSERT INTO agent_windows (agent_id, number, since_sequence, reset_at, reason, overflow_context)"
                " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (agent_id) DO UPDATE SET"
                " number = excluded.number, since_sequence = excluded.since_sequence,"
                " reset_at = excluded.reset_at, reason = excluded.reason,"
                " overflow_context = excluded.overflow_context",
                (agent_id, number, since_sequence, reset_at, reason, overflow_context),
            )
        return WindowState(number, since_sequence, reset_at, reason, overflow_context)

    def window_events(
        self,
        agent_id: int,
        *,
        after: int | None = None,
        limit: int = 30,
    ) -> tuple[WindowEvent, ...]:
        clause = "" if after is None else " AND number > ?"
        parameters: tuple[object, ...] = (
            (agent_id,) if after is None else (agent_id, after)
        )
        rows = self._db.execute(
            "SELECT number, since_sequence, reset_at, reason, overflow_context FROM agent_window_events"
            f" WHERE agent_id = ?{clause} ORDER BY number LIMIT ?",
            (*parameters, limit),
        )
        return tuple(
            WindowEvent(
                number=int(row["number"]),
                since_sequence=int(row["since_sequence"]),
                reset_at=row["reset_at"],
                reason=row["reason"],
                overflow_context=row["overflow_context"],
            )
            for row in rows
        )

    def latest_messages(self, agent_id: int) -> str:
        row = first(
            self._db.execute(
                "SELECT messages_json FROM agent_runs WHERE agent_id = ?"
                " AND sequence >= COALESCE((SELECT since_sequence FROM agent_windows"
                " WHERE agent_id = ?), 1)"
                " AND messages_json NOT IN ('[]', '') ORDER BY sequence DESC LIMIT 1",
                (agent_id, agent_id),
            )
        )
        return str(row["messages_json"]) if row else "[]"

    def read_run_slice(
        self, agent_id: int, sequence: int, offset: int | None, limit: int
    ) -> HistorySlice | None:
        row = first(
            self._db.execute(
                "SELECT sequence, status, started_at, last_saved_at,"
                " length(messages_json) AS total_length "
                "FROM agent_runs WHERE agent_id = ? AND sequence = ?",
                (agent_id, sequence),
            )
        )
        if row is None:
            return None
        total_length = int(row["total_length"])
        start = max(total_length - limit, 0) if offset is None else offset
        if start > total_length:
            raise DomainError("invalid_offset", "History offset exceeds its length")
        content = first(
            self._db.execute(
                "SELECT substr(messages_json, ?, ?) AS messages FROM agent_runs "
                "WHERE agent_id = ? AND sequence = ?",
                (start + 1, limit, agent_id, sequence),
            )
        )
        assert content is not None
        return HistorySlice(
            sequence=int(row["sequence"]),
            status=str(row["status"]),
            started_at=str(row["started_at"]),
            messages=str(content["messages"]),
            offset=start,
            total_length=total_length,
            last_saved_at=row["last_saved_at"],
        )

    def runs(self, agent_id: int, *, limit: int = 50) -> tuple[AgentRun, ...]:
        rows = self._db.execute(
            "SELECT * FROM agent_runs WHERE agent_id = ? ORDER BY sequence DESC LIMIT ?",
            (agent_id, limit),
        )
        return tuple(self._run(row) for row in rows)

    def run_summaries(
        self, agent_id: int, *, limit: int = 50
    ) -> tuple[RunSummary, ...]:
        rows = self._db.execute(
            "SELECT sequence, status, started_at, completed_at, usage_json, error"
            " FROM agent_runs WHERE agent_id = ? ORDER BY sequence DESC LIMIT ?",
            (agent_id, limit),
        )
        return tuple(
            RunSummary(
                sequence=int(row["sequence"]),
                status=str(row["status"]),
                started_at=str(row["started_at"]),
                completed_at=row["completed_at"],
                usage_json=row["usage_json"],
                error=row["error"],
            )
            for row in rows
        )

    def record_effect(
        self, agent_id: int, sequence: int, tool: str, summary: str
    ) -> None:
        with self._db:
            row = first(
                self._db.execute(
                    "SELECT COALESCE(MAX(ordinal), 0) AS last FROM run_effects"
                    " WHERE agent_id = ? AND sequence = ?",
                    (agent_id, sequence),
                )
            )
            ordinal = (int(row["last"]) if row is not None else 0) + 1
            self._db.execute(
                "INSERT INTO run_effects (agent_id, sequence, ordinal, tool, summary,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (agent_id, sequence, ordinal, tool, summary, self._now()),
            )

    def effects(
        self, agent_id: int, *, sequences: Sequence[int] = ()
    ) -> tuple[TurnEffect, ...]:
        if sequences:
            placeholders = ",".join("?" for _ in sequences)
            rows = self._db.execute(
                "SELECT sequence, ordinal, tool, summary, created_at FROM run_effects"
                f" WHERE agent_id = ? AND sequence IN ({placeholders})"
                " ORDER BY sequence DESC, ordinal",
                (agent_id, *sequences),
            )
        else:
            rows = self._db.execute(
                "SELECT sequence, ordinal, tool, summary, created_at FROM run_effects"
                " WHERE agent_id = ? ORDER BY sequence DESC, ordinal",
                (agent_id,),
            )
        return tuple(
            TurnEffect(
                int(row["sequence"]),
                int(row["ordinal"]),
                str(row["tool"]),
                str(row["summary"]),
                str(row["created_at"]),
            )
            for row in rows
        )

    def usage_total(self, agent_id: int) -> dict[str, int]:
        row = first(
            self._db.execute(
                "SELECT"
                " COALESCE(SUM(json_extract(usage_json, '$.input_tokens')), 0) AS input,"
                " COALESCE(SUM(json_extract(usage_json, '$.output_tokens')), 0) AS output,"
                " COALESCE(SUM(json_extract(usage_json, '$.cache_read_tokens')), 0) AS cached,"
                " COALESCE(SUM(json_extract(usage_json, '$.requests')), 0) AS requests"
                " FROM agent_runs WHERE agent_id = ? AND usage_json IS NOT NULL",
                (agent_id,),
            )
        )
        if row is None:
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "requests": 0,
                "total_tokens": 0,
            }
        stored = {
            "input_tokens": int(row["input"]),
            "output_tokens": int(row["output"]),
            "cache_read_tokens": int(row["cached"]),
            "requests": int(row["requests"]),
        }
        stored["total_tokens"] = stored["input_tokens"] + stored["output_tokens"]
        return stored

    def mark_interrupted(self) -> int:
        completed = self._now()
        with self._db:
            cursor = self._db.execute_cursor(
                "UPDATE agent_runs SET status = 'interrupted', completed_at = ?,"
                " last_saved_at = COALESCE(last_saved_at, ?) WHERE status = 'running'",
                (completed, completed),
            )
            self._db.execute(
                "UPDATE agent_model_requests SET status = 'interrupted', completed_at = ?"
                " WHERE status = 'pending'",
                (completed,),
            )
        return cursor.rowcount

    def mark_session_start(self) -> int:
        rows = self._db.execute(
            "SELECT agent_id, MAX(sequence) FROM agent_runs GROUP BY agent_id"
        )
        with self._db:
            self._db.executemany(
                "INSERT INTO agent_sessions (agent_id, start_after) VALUES (?, ?)"
                " ON CONFLICT (agent_id) DO UPDATE SET start_after = excluded.start_after",
                [(int(row[0]), int(row[1])) for row in rows],
            )
        return len(rows)

    def search_runs(
        self, agent_id: int, query: str, *, limit: int = 20
    ) -> tuple[AgentRun, ...]:
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self._db.execute(
            "SELECT * FROM agent_runs WHERE agent_id = ? AND messages_json LIKE ?"
            " ESCAPE '\\' ORDER BY sequence DESC LIMIT ?",
            (agent_id, f"%{escaped}%", limit),
        )
        return tuple(self._run(row) for row in rows)

    def get_settings(self, section: str) -> dict[str, object] | None:
        row = first(
            self._db.execute(
                "SELECT values_json FROM settings WHERE section = ?", (section,)
            )
        )
        if row is None:
            return None
        try:
            loaded = json.loads(str(row["values_json"]))
        except json.JSONDecodeError as error:
            raise DomainError(
                "invalid_setting", f"Settings section {section} contains invalid JSON"
            ) from error
        if not isinstance(loaded, dict):
            raise DomainError(
                "invalid_setting", f"Settings section {section} must be a JSON object"
            )
        return loaded

    def create_agent_with_model(
        self,
        name: str,
        model_config: object | None,
        create_member: Callable[[str], dict[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}

        def update(stored: dict[str, object] | None) -> dict[str, object]:
            nonlocal result
            catalog = ModelCatalog.restore(stored)
            try:
                selection = AgentModelConfig.model_validate(
                    {} if model_config is None else model_config
                )
            except ValidationError:
                raise DomainError(
                    "invalid_model_config", "Invalid Agent model configuration"
                ) from None
            catalog.validate_selection(selection, "New Agent")
            result = create_member(name)
            catalog.agent_configs[str(result["id"])] = selection
            return catalog.model_dump()

        self.update_settings("model", update)
        return result

    def model_catalog(self) -> dict[str, object]:
        catalog = ModelCatalog.restore(self.get_settings("model"))
        return {
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "enabled": model.enabled
                    and catalog.provider(model.provider_id).enabled,
                    "thinking_options": thinking_options(
                        catalog.provider(model.provider_id).api_type,
                        model.model,
                        model.thinking_budget_tokens,
                    ),
                    "thinking_budget_tokens": model.thinking_budget_tokens,
                }
                for model in catalog.models
            ],
            "default_model_id": catalog.default_model_id,
            "default_thinking": catalog.default_thinking,
        }

    def agent_model_selection(self, agent_id: int) -> dict[str, object]:
        catalog = ModelCatalog.restore(self.get_settings("model"))
        saved = catalog.agent_configs.get(str(agent_id), AgentModelConfig())
        model_id, thinking = catalog.selection(agent_id)
        return {
            "agent_id": agent_id,
            "model_config": saved.model_dump(),
            "effective": {"model_id": model_id, "thinking": thinking},
        }

    def update_settings(
        self,
        section: str,
        update: Callable[[dict[str, object] | None], dict[str, object]],
    ) -> dict[str, object]:
        with self._db:
            values = update(self.get_settings(section))
            self._db.execute(
                "INSERT INTO settings (section, values_json) VALUES (?, ?)"
                " ON CONFLICT (section) DO UPDATE SET values_json = excluded.values_json",
                (section, json.dumps(values, ensure_ascii=False, sort_keys=True)),
            )
            return values

    def set_settings(self, section: str, values: dict[str, object]) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO settings (section, values_json) VALUES (?, ?)"
                " ON CONFLICT (section) DO UPDATE SET values_json = excluded.values_json",
                (section, json.dumps(values, ensure_ascii=False, sort_keys=True)),
            )
