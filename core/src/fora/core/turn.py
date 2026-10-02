from __future__ import annotations

from collections.abc import Iterable, Sequence

PRODUCTIVE_TOOLS = frozenset({"send", "edit", "run"})


def is_productive(tools: Iterable[str]) -> bool:
    return any(tool in PRODUCTIVE_TOOLS for tool in tools)


def idle_streak(runs: Sequence[tuple[str, Sequence[str]]]) -> int:
    streak = 0
    for status, tools in runs:
        if status != "completed":
            break
        if is_productive(tools):
            break
        streak += 1
    return streak
