from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet

from fora.core.discussion import Message


def context_range(
    messages: Sequence[Message],
    anchor_id: int,
    member_id: int,
    mentions_by_message: Mapping[int, AbstractSet[int]],
    read_watermark: int,
) -> tuple[int, int]:
    anchor = next(
        (index for index, item in enumerate(messages) if item.id == anchor_id), None
    )
    if anchor is None:
        raise ValueError("anchor message is not in the provided sequence")

    low = anchor
    while low > 0:
        previous = messages[low - 1]
        mentioned = mentions_by_message.get(previous.id, frozenset())
        unread = previous.id > read_watermark
        if mentioned and member_id not in mentioned and unread:
            break
        low -= 1

    sender = messages[anchor].sender_id
    high = anchor
    while high < len(messages) - 1 and messages[high + 1].sender_id == sender:
        high += 1

    return low, high


def context_window(
    messages: Sequence[Message],
    anchor_id: int,
    member_id: int,
    mentions_by_message: Mapping[int, AbstractSet[int]],
    read_watermark: int,
) -> tuple[Message, ...]:
    low, high = context_range(
        messages, anchor_id, member_id, mentions_by_message, read_watermark
    )
    return tuple(messages[low : high + 1])


def advance_watermark(current: int, seen: Sequence[Message]) -> int:
    return max([current, *(message.id for message in seen)]) if seen else current
