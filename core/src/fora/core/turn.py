from __future__ import annotations

from collections.abc import Iterable, Sequence

PRODUCTIVE_TOOLS = frozenset({"send", "edit", "run"})
OVERFLOW_CONTEXT_BYTES = 4096
OVERFLOW_NOTICE_BYTES = 6144


def bounded_text(value: str, limit: int) -> str:
    marker = "\n[truncated fragment]"
    if limit < len(marker.encode("utf-8")):
        raise ValueError("Text budget must accommodate the truncation marker")
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    return (
        encoded[: limit - len(marker.encode("utf-8"))].decode("utf-8", errors="ignore")
        + marker
    )


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
