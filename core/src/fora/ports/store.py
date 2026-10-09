from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from fora.core.discussion import Discussion, Message
from fora.core.member import AgentState, Member, MemberType
from fora.core.mention import Mention


@dataclass(frozen=True)
class DiscussionPage:
    messages: tuple[Message, ...]
    read_through: int
    first_unread_id: int | None
    latest_id: int
    has_before: bool
    has_after: bool
    previous_sender_id: int | None
    awaiting_ack: tuple[int, ...]
    acknowledged: tuple[int, ...]
    pending_count: int
    discussion: Discussion
    members: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class PendingAcknowledgement:
    acked: int
    read_through: int
    pending_count: int


@dataclass(frozen=True)
class ReadSummary:
    id: int
    topic: str
    archived: bool
    read_through: int
    latest_id: int
    total_messages: int
    members_count: int
    awaiting_ack_count: int
    acknowledged_count: int


class OrganizationStore(Protocol):
    def read_summary(self, discussion_id: int, member_id: int) -> ReadSummary: ...

    def read_context_bounds(
        self, discussion_id: int, message_id: int, member_id: int, watermark: int
    ) -> tuple[int, int]: ...

    def read_message_ids(
        self, discussion_id: int, low: int, high: int, *, limit: int, reverse: bool
    ) -> tuple[int, ...]: ...

    def read_message(self, discussion_id: int, message_id: int) -> Message: ...

    def read_metadata(
        self,
        discussion_id: int,
        member_id: int,
        section: str,
        *,
        message_id: int | None = None,
        offset: int = 0,
        limit: int,
    ) -> tuple[int, tuple[object, ...]]: ...

    def read_ack_flags(
        self, discussion_id: int, member_id: int, message_ids: Sequence[int]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]: ...

    def finish_read(
        self, discussion_id: int, member_id: int, message_id: int | None
    ) -> None: ...

    def organization_uuid(self) -> str: ...

    def submit_message(
        self,
        discussion_id: int,
        sender_id: int,
        body: str,
        *,
        attachment_ids: Sequence[str] = (),
        client_message_id: str | None = None,
        mark_read: bool = False,
    ) -> tuple[Message, tuple[Mention, ...], bool]: ...

    def message_receipt(
        self,
        discussion_id: int,
        owner_id: int,
        client_message_id: str,
    ) -> Message | None: ...

    def send_outcome(
        self, discussion_id: int, owner_id: int, client_message_id: str
    ) -> tuple[str, Message | None]: ...

    def cancel_send(
        self, discussion_id: int, owner_id: int, client_message_id: str
    ) -> tuple[str, Message | None]: ...

    def list_members(self, *, include_deleted: bool = False) -> tuple[Member, ...]: ...

    def get_member(self, member_id: int) -> Member | None: ...

    def name_taken(self, name: str) -> bool: ...

    def create_member(self, member_type: MemberType, name: str) -> Member: ...

    def rename_member(self, member_id: int, name: str) -> Member: ...

    def set_agent_state(self, agent_id: int, state: AgentState) -> None: ...

    def delete_member(self, member_id: int) -> None: ...

    def list_discussions(
        self,
        *,
        member_id: int | None = None,
        include_archived: bool = False,
        limit: int | None = None,
    ) -> tuple[Discussion, ...]: ...

    def get_discussion(self, discussion_id: int) -> Discussion | None: ...

    def create_discussion(
        self, topic: str, member_ids: Sequence[int]
    ) -> Discussion: ...

    def set_discussion_members(
        self, discussion_id: int, member_ids: Sequence[int]
    ) -> Discussion: ...

    def change_discussion_members(
        self, discussion_id: int, member_ids: Sequence[int], *, remove: bool = False
    ) -> Discussion: ...

    def set_archived(self, discussion_id: int, archived: bool) -> None: ...

    def append_message(
        self, discussion_id: int, sender_id: int, body: str
    ) -> tuple[Message, tuple[Mention, ...]]: ...

    def messages(
        self,
        discussion_id: int,
        *,
        after: int | None = None,
        before: int | None = None,
        limit: int | None = None,
        latest: bool = False,
    ) -> tuple[Message, ...]: ...

    def discussion_page(
        self,
        discussion_id: int,
        member_id: int,
        *,
        limit: int,
        entry: bool = False,
        before: int | None = None,
        after: int | None = None,
    ) -> DiscussionPage: ...

    def mark_read(self, discussion_id: int, member_id: int, message_id: int) -> int: ...

    def ack_pending(
        self, discussion_id: int, member_id: int, through_message_id: int
    ) -> PendingAcknowledgement: ...

    def message_count(self, discussion_id: int) -> int: ...

    def mentions_by_message(
        self, discussion_id: int
    ) -> Mapping[int, frozenset[int]]: ...

    def search_messages(
        self,
        query: str,
        *,
        sender_id: int | None = None,
        discussion_id: int | None = None,
        limit: int = 50,
    ) -> tuple[Message, ...]: ...

    def pending(self, member_id: int) -> tuple[Mention, ...]: ...

    def ack(
        self, discussion_id: int, message_ids: Sequence[int], member_id: int
    ) -> int: ...

    def acknowledged(self, discussion_id: int, member_id: int) -> tuple[int, ...]: ...

    def revoke_ack(
        self, discussion_id: int, message_ids: Sequence[int], member_id: int
    ) -> int: ...

    def watermark(self, discussion_id: int, member_id: int) -> int: ...

    def set_watermark(
        self, discussion_id: int, member_id: int, message_id: int
    ) -> None: ...

    def unread_counts(self, member_id: int) -> Mapping[int, int]: ...
