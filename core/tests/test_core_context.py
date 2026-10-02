from __future__ import annotations

from fora.core.context import advance_watermark, context_window
from fora.core.discussion import Message

AGENT = 13
ALICE = 1
BOB = 36


def msg(message_id: int, sender: int) -> Message:
    return Message(3, message_id, sender, f"m{sender}", "body", "2026-08-30T00:00:00Z")


def window(
    messages: list[Message],
    anchor: int,
    mentions: dict[int, set[int]],
    watermark: int = 0,
) -> tuple[int, ...]:
    found = context_window(messages, anchor, AGENT, mentions, watermark)
    return tuple(item.id for item in found)


def test_backward_stops_at_an_unread_message_mentioning_someone_else() -> None:
    messages = [msg(1, ALICE), msg(2, ALICE), msg(3, ALICE), msg(4, ALICE)]
    mentions = {2: {BOB}, 4: {AGENT}}
    assert window(messages, 4, mentions) == (3, 4)


def test_backward_passes_through_a_read_message_mentioning_someone_else() -> None:
    messages = [msg(1, ALICE), msg(2, ALICE), msg(3, ALICE), msg(4, ALICE)]
    mentions = {2: {BOB}, 4: {AGENT}}
    assert window(messages, 4, mentions, watermark=2) == (1, 2, 3, 4)


def test_backward_passes_through_messages_without_mentions() -> None:
    messages = [msg(1, ALICE), msg(2, ALICE), msg(3, ALICE)]
    assert window(messages, 3, {3: {AGENT}}) == (1, 2, 3)


def test_backward_passes_through_messages_mentioning_me() -> None:
    messages = [msg(1, ALICE), msg(2, ALICE), msg(3, ALICE)]
    mentions = {1: {AGENT}, 2: {AGENT}, 3: {AGENT}}
    assert window(messages, 3, mentions) == (1, 2, 3)


def test_forward_absorbs_a_consecutive_run_from_the_same_sender() -> None:
    messages = [msg(1, ALICE), msg(2, ALICE), msg(3, ALICE)]
    assert window(messages, 1, {1: {AGENT}}) == (1, 2, 3)


def test_forward_stops_at_a_different_sender() -> None:
    messages = [msg(1, ALICE), msg(2, ALICE), msg(3, BOB), msg(4, ALICE)]
    assert window(messages, 1, {1: {AGENT}}) == (1, 2)


def test_forward_does_not_swallow_a_later_exchange() -> None:
    messages = [msg(1, ALICE), msg(2, BOB), msg(3, ALICE), msg(4, BOB)]
    assert window(messages, 1, {1: {AGENT}}) == (1,)


def test_anchor_alone_when_both_sides_stop_immediately() -> None:
    messages = [msg(1, BOB), msg(2, ALICE), msg(3, BOB)]
    mentions = {1: {BOB}, 2: {AGENT}}
    assert window(messages, 2, mentions) == (2,)


def test_watermark_advances_to_the_highest_message_seen() -> None:
    assert advance_watermark(2, [msg(3, ALICE), msg(5, ALICE)]) == 5
    assert advance_watermark(9, [msg(3, ALICE)]) == 9
    assert advance_watermark(4, []) == 4
