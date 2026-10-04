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
    uuid = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    patterns = (
        (
            "fixed",
            (
                r"\b(?:status_code|model_name|model_id|code|type)['\"]?\s*[:=]\s*"
                r"(?:'[^']*'|\"[^\"]*\"|[\w./:+-]+)"
            ),
        ),
        (
            "identifier",
            (
                r"\b(?:request[_ -]?id|trace[_ -]?id|diagnostic[_ -]?id)"
                r"['\"]?\s*[:=]\s*['\"]?[\w.-]+"
                rf"|\bdiagnostic\s+(?:{uuid}|[0-9a-f]{{6,}})(?![\w-])"
            ),
        ),
        ("random", rf"(?<![\w-])(?:{uuid}|[0-9a-f]{{24,}})(?![\w-])"),
        (
            "duration",
            (
                r"(?<![\w./-])\d+(?:\.\d+)?\s*"
                r"(?:milliseconds?|seconds?|minutes?|hours?|ms|s|m|h)\b"
            ),
        ),
        (
            "number",
            r"(?<![\w./-])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w./-]|\d)",
        ),
    )
    replacements = {
        "identifier": "<id>",
        "random": "<id>",
        "duration": "<duration>",
        "number": "<number>",
    }

    def normalize(match: re.Match[str]) -> str:
        if match.lastgroup == "fixed":
            return match.group()
        assert match.lastgroup is not None
        return replacements[match.lastgroup]

    value = re.sub(
        "|".join(f"(?P<{name}>{pattern})" for name, pattern in patterns),
        normalize,
        error,
        flags=re.IGNORECASE,
    )
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
