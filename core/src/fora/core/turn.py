from __future__ import annotations

import re
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


def model_error_signature(error: str) -> str:
    protected: list[str] = []

    def preserve(match: re.Match[str]) -> str:
        protected.append(match.group())
        return f"\x00{chr(65 + len(protected) - 1)}\x00"

    value = re.sub(
        r"(?i)\b(?:status_code|model_name|model_id|code|type)['\"]?\s*[:=]\s*"
        r"(?:'[^']*'|\"[^\"]*\"|[\w./:+-]+)",
        preserve,
        error,
    )
    value = re.sub(
        r"(?i)\b(?:request[_ -]?id|trace[_ -]?id|diagnostic(?:[_ -]?id)?)"
        r"['\"]?\s*[:=]?\s*['\"]?[\w.-]+",
        "<id>",
        value,
    )
    value = re.sub(
        r"(?i)(?<![\w-])(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
        r"[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{24,})(?![\w-])",
        "<id>",
        value,
    )
    value = re.sub(
        r"(?i)(?<![\w./-])\d+(?:\.\d+)?\s*"
        r"(milliseconds?|seconds?|minutes?|hours?|ms|s|m|h)\b",
        "<duration>",
        value,
    )
    value = re.sub(
        r"(?<![\w./-])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w./-]|\d)",
        "<number>",
        value,
    )
    for index, field in enumerate(protected):
        value = value.replace(f"\x00{chr(65 + index)}\x00", field)
    return " ".join(value.split())


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
