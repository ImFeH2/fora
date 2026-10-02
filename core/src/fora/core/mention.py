from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from fora.core.member import Member, name_key

_TRAILING_WORD = re.compile(r"[^\W_]", re.UNICODE)


@dataclass(frozen=True)
class Mention:
    discussion_id: int
    message_id: int
    member_id: int
    position: int
    length: int


def _candidates(members: Iterable[Member]) -> list[tuple[str, int]]:
    entries = [
        (name_key(member.name), member.id) for member in members if member.active
    ]
    entries.sort(key=lambda item: len(item[0]), reverse=True)
    return entries


def find_mention_ids(
    body: str, members: Iterable[Member]
) -> tuple[tuple[int, int, int], ...]:
    entries = _candidates(members)
    found: list[tuple[int, int, int]] = []
    index = 0
    length = len(body)
    while index < length:
        at = body.find("@", index)
        if at < 0:
            break
        if at > 0 and _TRAILING_WORD.match(body[at - 1]):
            index = at + 1
            continue
        start = at + 1
        matched = False
        for key, member_id in entries:
            end = start + len(key)
            if end > length or body[start:end].casefold() != key:
                continue
            if end < length and _TRAILING_WORD.match(body[end]):
                continue
            found.append((member_id, at, end - at))
            index = end
            matched = True
            break
        if not matched:
            index = at + 1
    return tuple(found)


def build_mentions(
    discussion_id: int,
    message_id: int,
    body: str,
    members: Iterable[Member],
    *,
    sender_id: int | None = None,
) -> tuple[Mention, ...]:
    seen: set[int] = set() if sender_id is None else {sender_id}
    mentions: list[Mention] = []
    for member_id, position, length in find_mention_ids(body, members):
        if member_id in seen:
            continue
        seen.add(member_id)
        mentions.append(Mention(discussion_id, message_id, member_id, position, length))
    return tuple(mentions)


def render_names(
    mentions: Iterable[Mention], members: Mapping[int, Member]
) -> tuple[str, ...]:
    return tuple(
        members[mention.member_id].name
        for mention in mentions
        if mention.member_id in members
    )
