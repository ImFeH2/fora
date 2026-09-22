from __future__ import annotations

from dataclasses import dataclass, field

from huddol.core.attachment import Attachment
from huddol.core.errors import DomainError

MAX_TOPIC_LENGTH = 200
MAX_BODY_LENGTH = 100_000


@dataclass(frozen=True)
class MessageMention:
    member_id: int
    position: int
    length: int


@dataclass(frozen=True)
class Message:
    discussion_id: int
    id: int
    sender_id: int
    sender_name: str
    body: str
    created_at: str
    mentions: tuple[MessageMention, ...] = ()
    attachments: tuple[Attachment, ...] = ()


@dataclass(frozen=True)
class Discussion:
    id: int
    topic: str
    member_ids: frozenset[int] = field(default_factory=frozenset)
    archived: bool = False

    def has_member(self, member_id: int) -> bool:
        return member_id in self.member_ids


def validate_topic(value: object) -> str:
    if not isinstance(value, str):
        raise DomainError("invalid_topic", "Topic must be a string")
    topic = " ".join(value.split())
    if not topic:
        raise DomainError("invalid_topic", "Topic must not be empty")
    if len(topic) > MAX_TOPIC_LENGTH:
        raise DomainError(
            "invalid_topic", f"Topic must be at most {MAX_TOPIC_LENGTH} characters"
        )
    return topic


def validate_body(value: object, *, has_attachments: bool = False) -> str:
    if not isinstance(value, str):
        raise DomainError("invalid_body", "Message body must be a string")
    if not value.strip() and not has_attachments:
        raise DomainError("invalid_body", "Message body must not be empty")
    if len(value) > MAX_BODY_LENGTH:
        raise DomainError(
            "invalid_body", f"Message body must be at most {MAX_BODY_LENGTH} characters"
        )
    return value
