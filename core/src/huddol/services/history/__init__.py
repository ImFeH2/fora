from __future__ import annotations

from huddol.core.errors import DomainError
from huddol.ports.agent import AgentRun, HistorySlice, HistoryStore

READ_LIMIT = 2048
MAX_SQLITE_INTEGER = 2**63 - 1


class History:
    def __init__(self, store: HistoryStore, agent_id: int) -> None:
        self._store = store
        self._agent_id = agent_id

    def runs(self, *, limit: int = 50) -> tuple[AgentRun, ...]:
        return self._store.runs(self._agent_id, limit=limit)

    def search(self, query: str, *, limit: int = 20) -> tuple[AgentRun, ...]:
        return self._store.search_runs(self._agent_id, query, limit=limit)

    def read(self, sequence: int, offset: int = 0) -> HistorySlice | None:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise DomainError(
                "invalid_offset", "History offset must be a non-negative integer"
            )
        if offset > MAX_SQLITE_INTEGER:
            raise DomainError(
                "invalid_offset", "History offset exceeds the supported range"
            )
        return self._store.read_run_slice(self._agent_id, sequence, offset, READ_LIMIT)
