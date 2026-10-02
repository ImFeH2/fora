from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from fora.core.errors import DomainError

MemberType = Literal["human", "agent"]
AgentState = Literal["idle", "running", "paused", "blocked", "error"]

MAX_NAME_LENGTH = 64
_WHITESPACE = re.compile(r"\s+")
_FORBIDDEN = re.compile(r"[@\x00-\x1f\x7f]")


@dataclass(frozen=True)
class Member:
    id: int
    type: MemberType
    name: str
    deleted: bool = False
    state: AgentState = "idle"

    @property
    def is_agent(self) -> bool:
        return self.type == "agent"

    @property
    def active(self) -> bool:
        return not self.deleted


def normalize_name(value: str) -> str:
    collapsed = _WHITESPACE.sub(" ", unicodedata.normalize("NFC", value)).strip()
    return collapsed


def name_key(value: str) -> str:
    return normalize_name(value).casefold()


def validate_name(value: object) -> str:
    if not isinstance(value, str):
        raise DomainError("invalid_name", "Member name must be a string")
    normalized = normalize_name(value)
    if not normalized:
        raise DomainError("invalid_name", "Member name must not be empty")
    if len(normalized) > MAX_NAME_LENGTH:
        raise DomainError(
            "invalid_name", f"Member name must be at most {MAX_NAME_LENGTH} characters"
        )
    if _FORBIDDEN.search(normalized):
        raise DomainError(
            "invalid_name", "Member name must not contain @ or control characters"
        )
    return normalized
