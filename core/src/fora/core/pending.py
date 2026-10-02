from __future__ import annotations

from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from fora.core.discussion import Discussion
from fora.core.mention import Mention

AckKey = tuple[int, int, int]


@dataclass(frozen=True)
class Ack:
    discussion_id: int
    message_id: int
    member_id: int

    @property
    def key(self) -> AckKey:
        return (self.discussion_id, self.message_id, self.member_id)


def ack_keys(acks: Iterable[Ack]) -> frozenset[AckKey]:
    return frozenset(ack.key for ack in acks)


def is_pending(
    mention: Mention,
    member_id: int,
    acks: AbstractSet[AckKey],
    discussions: Mapping[int, Discussion],
) -> bool:
    if mention.member_id != member_id:
        return False
    if (mention.discussion_id, mention.message_id, member_id) in acks:
        return False
    discussion = discussions.get(mention.discussion_id)
    if discussion is None or discussion.archived:
        return False
    return discussion.has_member(member_id)


def pending_for(
    member_id: int,
    mentions: Iterable[Mention],
    acks: AbstractSet[AckKey],
    discussions: Mapping[int, Discussion],
) -> tuple[Mention, ...]:
    return tuple(
        mention
        for mention in mentions
        if is_pending(mention, member_id, acks, discussions)
    )
