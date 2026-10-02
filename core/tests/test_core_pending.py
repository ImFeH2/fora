from __future__ import annotations

from fora.core.discussion import Discussion
from fora.core.mention import Mention
from fora.core.pending import Ack, ack_keys, pending_for

AGENT = 13
OTHER = 36


def discussion(*member_ids: int, archived: bool = False) -> Discussion:
    return Discussion(3, "topic", frozenset(member_ids), archived)


def mention(message_id: int, member_id: int = AGENT) -> Mention:
    return Mention(3, message_id, member_id, 0, 5)


def pending(
    mentions: tuple[Mention, ...],
    acks: tuple[Ack, ...],
    room: Discussion,
) -> tuple[int, ...]:
    result = pending_for(AGENT, mentions, ack_keys(acks), {room.id: room})
    return tuple(item.message_id for item in result)


def test_unacked_mention_in_a_joined_discussion_is_pending() -> None:
    assert pending((mention(1),), (), discussion(AGENT)) == (1,)


def test_ack_removes_the_mention() -> None:
    assert pending((mention(1),), (Ack(3, 1, AGENT),), discussion(AGENT)) == ()


def test_ack_by_another_member_does_not_remove_it() -> None:
    assert pending((mention(1),), (Ack(3, 1, OTHER),), discussion(AGENT)) == (1,)


def test_mentions_of_other_members_are_not_pending() -> None:
    assert pending((mention(1, OTHER),), (), discussion(AGENT, OTHER)) == ()


def test_leaving_the_discussion_clears_pending_without_any_cleanup() -> None:
    mentions = (mention(1), mention(2))
    assert pending(mentions, (), discussion(AGENT)) == (1, 2)
    assert pending(mentions, (), discussion(OTHER)) == ()


def test_rejoining_restores_only_unacked_mentions() -> None:
    mentions = (mention(1), mention(2))
    acks = (Ack(3, 1, AGENT),)
    assert pending(mentions, acks, discussion(OTHER)) == ()
    assert pending(mentions, acks, discussion(AGENT)) == (2,)


def test_archived_discussions_stop_producing_pending() -> None:
    assert pending((mention(1),), (), discussion(AGENT, archived=True)) == ()


def test_unknown_discussions_have_no_pending() -> None:
    result = pending_for(AGENT, (mention(1),), frozenset(), {})
    assert result == ()


def test_revoking_an_ack_restores_pending() -> None:
    mentions = (mention(1),)
    room = discussion(AGENT)
    assert pending(mentions, (Ack(3, 1, AGENT),), room) == ()
    assert pending(mentions, (), room) == (1,)
