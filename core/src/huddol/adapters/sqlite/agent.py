from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable, Sequence

from huddol.adapters.model.config import ModelCatalog
from huddol.adapters.sqlite.store import LockedConnection, first
from huddol.core.errors import DomainError
from huddol.ports.agent import (
    AgentLifecycle,
    AgentRun,
    HistorySlice,
    RunSummary,
    TurnEffect,
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
    reason TEXT
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
"""


class SqliteAgentStore:
    def __init__(self, db: LockedConnection) -> None:
        self._db = db
        self._db.executescript(SCHEMA)
        self._db.commit()
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
        from huddol.adapters.sqlite.store import now

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
        with self._db:
            revision = self.pending_revision(agent_id)
            self._db.execute(
                "INSERT INTO agent_runs (agent_id, sequence, run_id, status, started_at,"
                " messages_json, reminded_json, pending_revision)"
                " VALUES (?, ?, ?, 'running', ?, '[]', ?, ?)",
                (
                    agent_id,
                    sequence,
                    identifier,
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
            agent_id,
            sequence,
            identifier,
            "running",
            started,
            None,
            "[]",
            None,
            None,
            revision,
        )

    def save_progress(self, agent_id: int, sequence: int, messages_json: str) -> None:
        with self._db:
            self._db.execute(
                "UPDATE agent_runs SET messages_json = ? WHERE agent_id = ? AND sequence = ?",
                (messages_json, agent_id, sequence),
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
                "UPDATE agent_runs SET status = ?, completed_at = ?, messages_json = ?,"
                " usage_json = ?, error = ? WHERE agent_id = ? AND sequence = ?",
                (
                    status,
                    self._now(),
                    messages_json,
                    usage_json,
                    error,
                    agent_id,
                    sequence,
                ),
            )

    def window(self, agent_id: int) -> WindowState:
        row = first(
            self._db.execute(
                "SELECT number, since_sequence, reset_at, reason FROM agent_windows"
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
        )

    def reset_window(self, agent_id: int, reason: str) -> WindowState:
        with self._db:
            self._db.execute(
                "INSERT INTO agent_windows (agent_id, number, since_sequence, reset_at, reason)"
                " VALUES (?, 2, (SELECT COALESCE(MAX(sequence), 0) + 1 FROM agent_runs"
                " WHERE agent_id = ?), ?, ?) ON CONFLICT (agent_id) DO UPDATE SET"
                " number = agent_windows.number + 1, since_sequence = excluded.since_sequence,"
                " reset_at = excluded.reset_at, reason = excluded.reason",
                (agent_id, agent_id, self._now(), reason),
            )
            return self.window(agent_id)

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
                "SELECT sequence, status, started_at, length(messages_json) AS total_length "
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
        with self._db:
            cursor = self._db.execute_cursor(
                "UPDATE agent_runs SET status = 'interrupted', completed_at = ?"
                " WHERE status = 'running'",
                (self._now(),),
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
