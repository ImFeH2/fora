from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fora.core.attachment import (
    MAX_IMAGE_BYTES,
    ImageData,
    ViewedAttachment,
    ViewedImage,
    identifier,
)
from fora.core.context import advance_watermark, context_window
from fora.core.discussion import Discussion, Message, validate_body, validate_topic
from fora.core.errors import DomainError
from fora.core.member import validate_name
from fora.ports.agent import HistoryStore, SettingsStore
from fora.ports.execution import ExecutionControl, ExecutionEnvironment
from fora.ports.files import ConflictError, FileTree
from fora.ports.store import OrganizationStore
from fora.services.history import History
from fora.services.library import Library
from fora.services.uploads import Uploads
from fora.services.workspace import Workspace
from fora.tools.authorize import Actor, Authorizer


@dataclass
class Dependencies:
    store: OrganizationStore
    history: HistoryStore
    settings: SettingsStore
    execution: ExecutionControl
    library_tree: FileTree
    workspace_tree_for: Callable[[int], FileTree]
    agent_directory_for: Callable[[int], Path]
    decode_image: Callable[[bytes], ImageData]
    uploads: Uploads | None = None
    reading: Callable[[], AbstractContextManager[None]] = nullcontext


@dataclass(frozen=True)
class TurnBinding:
    agent_id: int
    sequence: int


class AgentTools:
    def __init__(
        self,
        deps: Dependencies,
        actor: Actor,
        authorizer: Authorizer | None = None,
        turn: TurnBinding | None = None,
        *,
        on_change: Callable[[str, dict[str, Any]], None] | None = None,
        agent_status: Callable[[int], dict[str, Any]] | None = None,
        pause: Callable[[int], dict[str, Any]] | None = None,
        resume: Callable[[int, bool], dict[str, Any]] | None = None,
        member_guard: Callable[[int], AbstractContextManager[object]] | None = None,
        member_deleted: Callable[[int], None] | None = None,
    ) -> None:
        self._deps = deps
        self._actor = actor
        self._auth = authorizer or Authorizer()
        self._turn = turn
        self._on_change = on_change or (lambda name, payload: None)
        self._agent_status = agent_status
        self._pause = pause
        self._resume = resume
        self._member_guard = member_guard
        self._member_deleted = member_deleted

    def _changed(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._on_change(name, payload)
        return payload

    def _record(self, tool: str, summary: str) -> None:
        if self._turn is None:
            return
        self._deps.history.record_effect(
            self._turn.agent_id, self._turn.sequence, tool, summary
        )

    def _check(self, capability: str, target: object = None) -> None:
        self._auth.check(self._actor, capability, target)

    def _discussion(self, discussion_id: int) -> Discussion:
        discussion = self._deps.store.get_discussion(discussion_id)
        if discussion is None:
            raise DomainError("not_found", f"Discussion {discussion_id} does not exist")
        return discussion

    def _require_membership(self, discussion_id: int) -> None:
        if not self._discussion(discussion_id).has_member(self._actor.member_id):
            raise DomainError(
                "not_a_member", f"You do not belong to Discussion {discussion_id}"
            )

    def authorize_agent_history(self, agent_id: int) -> None:
        self._check("agent.history", agent_id)
        if self._actor.is_agent:
            raise DomainError("not_permitted", "Agent history is restricted to Humans")
        member = self._deps.store.get_member(agent_id)
        if member is None or not member.is_agent or member.deleted:
            raise DomainError("not_found", f"Agent {agent_id} does not exist")

    def list_member_records(
        self, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        self._check("organization.list_members")
        return [
            {
                "id": item.id,
                "type": item.type,
                "name": item.name,
                "state": item.state,
            }
            for item in self._deps.store.list_members(include_deleted=include_deleted)
        ]

    def list_members(self, include_deleted: bool = False) -> list[dict[str, Any]]:
        members = self.list_member_records(include_deleted)
        if self._agent_status is not None:
            for member in members:
                if member["type"] == "agent":
                    member.update(self._agent_status(member["id"]))
        return members

    def create_agent(
        self, name: str, model_config: object | None = None
    ) -> dict[str, Any]:
        self._check("organization.create_agent")
        validated = validate_name(name)
        result = self._deps.settings.create_agent_with_model(validated, model_config)
        return self._changed("member.created", result)

    def list_models(self) -> dict[str, object]:
        self._check("organization.list_models")
        return self._deps.settings.model_catalog()

    def get_agent_model(self, agent_id: int) -> dict[str, object]:
        self._check("organization.get_model", agent_id)
        member = self._deps.store.get_member(agent_id)
        if member is None or not member.is_agent or member.deleted:
            raise DomainError("not_found", f"Agent {agent_id} does not exist")
        return self._deps.settings.agent_model_selection(agent_id)

    def rename_member(self, member_id: int, name: str) -> dict[str, Any]:
        self._check("organization.rename_member", member_id)
        validated = validate_name(name)
        existing = self._deps.store.get_member(member_id)
        if existing is None:
            raise DomainError("not_found", f"Member {member_id} does not exist")
        if existing.name != validated and self._deps.store.name_taken(validated):
            raise DomainError("duplicate_name", "Member names must be unique")
        member = self._deps.store.rename_member(member_id, validated)
        return {"id": member.id, "name": member.name}

    def pause_agent(self, agent_id: int) -> dict[str, Any]:
        self._check("organization.pause_agent", agent_id)
        if self._pause is not None:
            return self._pause(agent_id)
        self._deps.store.set_agent_state(agent_id, "paused")
        return {"id": agent_id, "state": "paused"}

    def resume_agent(self, agent_id: int) -> dict[str, Any]:
        self._check("organization.resume_agent", agent_id)
        if self._resume is not None:
            return self._resume(agent_id, self._actor.is_agent)
        member = self._deps.store.get_member(agent_id)
        if member is None or not member.is_agent:
            raise DomainError("not_found", f"Agent {agent_id} does not exist")
        if (
            self._deps.history.pause_reason(agent_id) is not None
            and self._actor.is_agent
        ):
            raise DomainError(
                "not_permitted", "Only a Human can resume a safety-paused Agent"
            )
        runs = self._deps.history.run_summaries(agent_id, limit=1)
        if runs and runs[0].status == "running":
            raise DomainError(
                "agent_running", "Wait for the active Turn to finish before resuming"
            )
        if member.state == "paused":
            self._deps.history.reset_safety(agent_id)
        self._deps.store.set_agent_state(agent_id, "idle")
        return {"id": agent_id, "state": "idle"}

    def delete_agent(self, agent_id: int) -> dict[str, Any]:
        self._check("organization.delete_agent", agent_id)
        with (
            self._member_guard(agent_id)
            if self._member_guard is not None
            else nullcontext()
        ):
            return self._delete_agent(agent_id)

    def _delete_agent(self, agent_id: int) -> dict[str, Any]:
        member = self._deps.store.get_member(agent_id)
        if member is None or not member.is_agent:
            raise DomainError("not_found", f"Agent {agent_id} does not exist")
        state = (
            self._agent_status(agent_id)["state"]
            if self._agent_status is not None
            else member.state
        )
        if state == "running":
            raise DomainError(
                "agent_running",
                "Pause the Agent and let its Turn finish before deleting",
            )
        self._deps.settings.delete_agent_with_model(agent_id)
        if self._member_deleted is not None:
            self._member_deleted(agent_id)
        return {"id": agent_id, "deleted": True}

    def create_discussion(
        self, topic: str, member_ids: Sequence[int]
    ) -> dict[str, Any]:
        self._check("discussion.create")
        validated = validate_topic(topic)
        others = [
            item for item in dict.fromkeys(member_ids) if item != self._actor.member_id
        ]
        if not others:
            raise DomainError(
                "needs_members", "A Discussion needs at least one other Member"
            )
        known = {item.id for item in self._deps.store.list_members()}
        unknown = [item for item in others if item not in known]
        if unknown:
            raise DomainError("not_found", f"Unknown Members: {unknown}")
        discussion = self._deps.store.create_discussion(
            validated, [self._actor.member_id, *others]
        )
        return self._changed(
            "discussion.created",
            {
                "id": discussion.id,
                "topic": discussion.topic,
                "member_ids": sorted(discussion.member_ids),
            },
        )

    def list_discussions(
        self, include_archived: bool = False, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        self._check("discussion.list")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise DomainError("invalid_pagination", "limit must be an integer >= 1")
        with self._deps.reading():
            unread = self._deps.store.unread_counts(self._actor.member_id)
            discussions = self._deps.store.list_discussions(
                member_id=self._actor.member_id,
                include_archived=include_archived,
                limit=limit,
            )
        return [
            {
                "id": item.id,
                "topic": item.topic,
                "member_ids": sorted(item.member_ids),
                "archived": item.archived,
                "unread": unread.get(item.id, 0),
            }
            for item in discussions
        ]

    def read_discussion(
        self,
        discussion_id: int,
        message_id: int | None = None,
        limit: int | None = None,
        *,
        before: int | None = None,
        after: int | None = None,
    ) -> dict[str, Any]:
        self._check("discussion.read", discussion_id)
        self._require_membership(discussion_id)
        for name, value, minimum in (
            ("before", before, 0),
            ("after", after, 0),
            ("limit", limit, 1),
        ):
            if value is not None and (type(value) is not int or value < minimum):
                raise DomainError(
                    "invalid_pagination", f"{name} must be an integer >= {minimum}"
                )
        if message_id is not None and (before is not None or after is not None):
            raise DomainError(
                "invalid_pagination",
                "message_id cannot be combined with before or after",
            )
        store = self._deps.store
        discussion = self._discussion(discussion_id)
        read_before = store.watermark(discussion_id, self._actor.member_id)

        if message_id is None:
            selected = store.messages(
                discussion_id,
                before=before,
                after=after,
                limit=limit,
                latest=after is None and limit is not None,
            )
        else:
            everything = store.messages(discussion_id)
            mentions = store.mentions_by_message(discussion_id)
            try:
                selected = context_window(
                    everything,
                    message_id,
                    self._actor.member_id,
                    mentions,
                    read_before,
                )
            except ValueError as error:
                raise DomainError(
                    "not_found", f"Message {message_id} is not in this Discussion"
                ) from error

        if selected:
            store.set_watermark(
                discussion_id,
                self._actor.member_id,
                advance_watermark(
                    store.watermark(discussion_id, self._actor.member_id), selected
                ),
            )

        members = {
            item.id: item.name for item in store.list_members(include_deleted=True)
        }
        awaiting = tuple(
            item.message_id
            for item in store.pending(self._actor.member_id)
            if item.discussion_id == discussion_id
        )
        return {
            "id": discussion.id,
            "topic": discussion.topic,
            "read_through": read_before,
            "awaiting_ack": list(awaiting),
            "acknowledged": list(
                store.acknowledged(discussion_id, self._actor.member_id)
            ),
            "members": [
                {"id": item, "name": members.get(item, f"Member {item}")}
                for item in sorted(discussion.member_ids)
            ],
            "total_messages": store.message_count(discussion_id),
            "archived": discussion.archived,
            "messages": [self._message(item) for item in selected],
        }

    def _message(self, item: Message) -> dict[str, Any]:
        return {
            "id": item.id,
            "sender_id": item.sender_id,
            "sender_name": item.sender_name,
            "body": item.body,
            "created_at": item.created_at,
            "mentions": [asdict(mention) for mention in item.mentions],
            "attachments": [
                asdict(attachment)
                | (
                    {"path": self._uploads().files.path(attachment.id)}
                    if self._actor.is_agent
                    else {}
                )
                for attachment in item.attachments
            ],
        }

    def _human_discussion(self, capability: str, discussion_id: int) -> None:
        self._check(capability, discussion_id)
        if self._actor.is_agent:
            raise DomainError(
                "not_permitted", "This operation is for the Human interface"
            )
        if type(discussion_id) is not int or discussion_id < 1:
            raise DomainError(
                "invalid_discussion", "discussion_id must be a positive integer"
            )

    @staticmethod
    def _message_id(value: int) -> None:
        if type(value) is not int or value < 1:
            raise DomainError(
                "invalid_message", "message_id must be a positive integer"
            )

    def discussion_page(
        self,
        discussion_id: int,
        *,
        limit: int = 50,
        entry: bool = False,
        before: int | None = None,
        after: int | None = None,
        metadata: bool = False,
    ) -> dict[str, Any]:
        self._human_discussion("discussion.page", discussion_id)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise DomainError(
                "invalid_pagination", "limit must be an integer from 1 to 100"
            )
        if type(entry) is not bool or type(metadata) is not bool:
            raise DomainError(
                "invalid_pagination", "entry and metadata must be boolean"
            )
        for bound in (before, after):
            if bound is not None and (type(bound) is not int or bound < 0):
                raise DomainError(
                    "invalid_pagination", "bounds must be nonnegative integers"
                )
        if entry and (before is not None or after is not None):
            raise DomainError(
                "invalid_pagination", "entry cannot be combined with bounds"
            )
        if before is not None and after is not None and before <= after:
            raise DomainError("invalid_pagination", "before must exceed after")
        page = self._deps.store.discussion_page(
            discussion_id,
            self._actor.member_id,
            limit=limit,
            entry=entry,
            before=before,
            after=after,
        )
        result: dict[str, Any] = {
            "id": discussion_id,
            "messages": [self._message(item) for item in page.messages],
            "read_through": page.read_through,
            "latest_id": page.latest_id,
            "has_before": page.has_before,
            "has_after": page.has_after,
            "previous_sender_id": page.previous_sender_id,
            "awaiting_ack": list(page.awaiting_ack),
            "acknowledged": list(page.acknowledged),
            "pending_count": page.pending_count,
        }
        if entry:
            result["first_unread_id"] = page.first_unread_id
        if metadata or entry:
            result["metadata"] = {
                "topic": page.discussion.topic,
                "archived": page.discussion.archived,
                "members": [{"id": key, "name": name} for key, name in page.members],
            }
        return result

    def mark_read(self, discussion_id: int, message_id: int) -> dict[str, Any]:
        self._human_discussion("discussion.mark_read", discussion_id)
        self._message_id(message_id)
        watermark = self._deps.store.mark_read(
            discussion_id, self._actor.member_id, message_id
        )
        return {
            "discussion_id": discussion_id,
            "member_id": self._actor.member_id,
            "read_through": watermark,
        }

    def ack_pending(
        self, discussion_id: int, through_message_id: int
    ) -> dict[str, Any]:
        self._human_discussion("discussion.ack_pending", discussion_id)
        self._message_id(through_message_id)
        result = self._deps.store.ack_pending(
            discussion_id, self._actor.member_id, through_message_id
        )
        self._changed(
            "mention.acked", {"discussion_id": discussion_id, "acked": result.acked}
        )
        return asdict(result)

    def view_attachment(
        self, discussion_id: int, message_id: int, attachment_id: str
    ) -> ViewedAttachment:
        self._check("discussion.read", discussion_id)
        self._check("view_attachment", discussion_id)
        self._require_membership(discussion_id)
        result = self._uploads().view(
            discussion_id, message_id, attachment_id, self._actor.member_id
        )
        self._record(
            "view_attachment",
            f"Discussion {discussion_id}: message {message_id}, attachment {attachment_id}",
        )
        return result

    def view_image(self, path: str) -> ViewedImage:
        self._check("view_image", path)
        execution = self._deps.execution.snapshot()
        image = self._deps.decode_image(execution.read_file(path, MAX_IMAGE_BYTES))
        self._record("view_image", f"Image path: {path}")
        return ViewedImage(path, image)

    def organization_uuid(self) -> str:
        self._check("organization.get")
        return self._deps.store.organization_uuid()

    def _uploads(self) -> Uploads:
        assert self._deps.uploads is not None, "Uploads service is required"
        return self._deps.uploads

    def create_upload(
        self,
        discussion_id: int,
        client_upload_id: str,
        name: str,
        size: int,
        media_type: str,
    ) -> dict[str, Any]:
        self._human_discussion("upload.create", discussion_id)
        self._require_membership(discussion_id)
        return asdict(
            self._uploads().create(
                discussion_id,
                self._actor.member_id,
                client_upload_id,
                name,
                size,
                media_type,
            )
        )

    def upload_status(self, upload_ids: Sequence[str]) -> list[dict[str, Any]]:
        self._check("upload.status")
        if len(upload_ids) > 40:
            raise DomainError("invalid_uploads", "Query at most 40 uploads")
        result = []
        for upload_id in upload_ids:
            record = self._uploads().store.get_upload(
                identifier(upload_id), self._actor.member_id
            )
            self._require_membership(record.discussion_id)
            result.append(
                asdict(self._uploads().status(record.id, self._actor.member_id))
            )
        return result

    def cancel_uploads(self, upload_ids: Sequence[str]) -> dict[str, int]:
        self._check("upload.cancel")
        self.upload_status(upload_ids)
        for upload_id in upload_ids:
            self._uploads().cancel(upload_id, self._actor.member_id)
        return {"cancelled": len(upload_ids)}

    def send_status(self, discussion_id: int, client_message_id: str) -> dict[str, Any]:
        self._check("discussion.send_status", discussion_id)
        self._require_membership(discussion_id)
        state, result = self._deps.store.send_outcome(
            discussion_id, self._actor.member_id, identifier(client_message_id)
        )
        return {
            "state": state,
            "message": self._message(result) if result is not None else None,
        }

    def cancel_send(self, discussion_id: int, client_message_id: str) -> dict[str, Any]:
        self._check("discussion.cancel_send", discussion_id)
        self._require_membership(discussion_id)
        state, result = self._deps.store.cancel_send(
            discussion_id, self._actor.member_id, identifier(client_message_id)
        )
        return {
            "state": state,
            "message": self._message(result) if result is not None else None,
        }

    def send_message(
        self,
        discussion_id: int,
        body: str,
        *,
        mark_read: bool = True,
        attachment_ids: Sequence[str] = (),
        client_message_id: str | None = None,
    ) -> dict[str, Any]:
        self._check("discussion.send", discussion_id)
        self._require_membership(discussion_id)
        if type(mark_read) is not bool or (self._actor.is_agent and not mark_read):
            raise DomainError(
                "not_permitted",
                "Only the Human interface can send without marking read",
            )
        validated = validate_body(body, has_attachments=bool(attachment_ids))
        if len(attachment_ids) > 10:
            raise DomainError("invalid_attachments", "Choose up to 10 distinct files")
        state = "unknown"
        if client_message_id is not None:
            state, _ = self._deps.store.send_outcome(
                discussion_id, self._actor.member_id, identifier(client_message_id)
            )
        if state == "cancelled":
            raise DomainError("send_cancelled", "This send attempt was cancelled")
        if state == "unknown":
            try:
                for upload_id in attachment_ids:
                    self._uploads().status(upload_id, self._actor.member_id)
            except (DomainError, OSError):
                if client_message_id is not None:
                    state, _ = self._deps.store.send_outcome(
                        discussion_id, self._actor.member_id, client_message_id
                    )
                    if state == "cancelled":
                        raise DomainError(
                            "send_cancelled", "This send attempt was cancelled"
                        ) from None
                raise
        message, mentions, created = self._deps.store.submit_message(
            discussion_id,
            self._actor.member_id,
            validated,
            attachment_ids=attachment_ids,
            client_message_id=client_message_id,
            mark_read=mark_read,
        )
        if not created:
            return {
                "discussion_id": discussion_id,
                "id": message.id,
                "created_at": message.created_at,
                "mentioned": [],
            }
        self._record(
            "send",
            f"Discussion {discussion_id}: message {message.id}"
            f" ({len(validated)} characters)",
        )
        return self._changed(
            "message.created",
            {
                "discussion_id": message.discussion_id,
                "id": message.id,
                "created_at": message.created_at,
                "mentioned": [item.member_id for item in mentions],
            },
        )

    def ack(self, discussion_id: int, message_ids: Sequence[int]) -> dict[str, Any]:
        self._check("discussion.ack", discussion_id)
        self._require_membership(discussion_id)
        watermark = self._deps.store.watermark(discussion_id, self._actor.member_id)
        unread = [item for item in message_ids if item > watermark]
        if unread:
            raise DomainError(
                "not_read", f"Read messages {unread} before acknowledging them"
            )
        acked = self._deps.store.ack(
            discussion_id, list(message_ids), self._actor.member_id
        )
        self._record("ack", f"Discussion {discussion_id}: {acked} acknowledged")
        return self._changed(
            "mention.acked", {"discussion_id": discussion_id, "acked": acked}
        )

    def revoke_ack(
        self, discussion_id: int, message_ids: Sequence[int]
    ) -> dict[str, Any]:
        self._check("discussion.revoke_ack", discussion_id)
        self._require_membership(discussion_id)
        revoked = self._deps.store.revoke_ack(
            discussion_id, message_ids, self._actor.member_id
        )
        return self._changed(
            "mention.revoked", {"discussion_id": discussion_id, "revoked": revoked}
        )

    def _member_ids(self, member_ids: Sequence[int]) -> None:
        if any(type(item) is not int or item <= 0 for item in member_ids):
            raise DomainError("invalid_members", "Member IDs must be positive integers")

    def set_discussion_members(
        self, discussion_id: int, member_ids: Sequence[int]
    ) -> dict[str, Any]:
        self._check("discussion.set_members", discussion_id)
        self._discussion(discussion_id)
        self._member_ids(member_ids)
        updated = self._deps.store.set_discussion_members(discussion_id, member_ids)
        return self._changed(
            "discussion.updated",
            {"id": updated.id, "member_ids": sorted(updated.member_ids)},
        )

    def add_members(
        self, discussion_id: int, member_ids: Sequence[int]
    ) -> dict[str, Any]:
        self._check("discussion.add_members", discussion_id)
        self._member_ids(member_ids)
        updated = self._deps.store.change_discussion_members(discussion_id, member_ids)
        return self._changed(
            "discussion.updated",
            {"id": updated.id, "member_ids": sorted(updated.member_ids)},
        )

    def remove_members(
        self, discussion_id: int, member_ids: Sequence[int]
    ) -> dict[str, Any]:
        self._check("discussion.remove_members", discussion_id)
        self._member_ids(member_ids)
        updated = self._deps.store.change_discussion_members(
            discussion_id, member_ids, remove=True
        )
        return self._changed(
            "discussion.updated",
            {"id": updated.id, "member_ids": sorted(updated.member_ids)},
        )

    def archive_discussion(
        self, discussion_id: int, archived: bool = True
    ) -> dict[str, Any]:
        self._check(
            "discussion.archive" if archived else "discussion.unarchive", discussion_id
        )
        self._discussion(discussion_id)
        self._deps.store.set_archived(discussion_id, archived)
        return self._changed(
            "discussion.updated", {"id": discussion_id, "archived": archived}
        )

    def search_messages(
        self,
        query: str,
        sender_id: int | None = None,
        discussion_id: int | None = None,
    ) -> list[dict[str, Any]]:
        self._check("discussion.search", discussion_id)
        mine = {
            item.id
            for item in self._deps.store.list_discussions(
                member_id=self._actor.member_id, include_archived=True
            )
        }
        found = self._deps.store.search_messages(
            query, sender_id=sender_id, discussion_id=discussion_id
        )
        return [
            {
                "discussion_id": item.discussion_id,
                "id": item.id,
                "sender_name": item.sender_name,
                "body": item.body,
                "mentions": [asdict(mention) for mention in item.mentions],
                "attachments": self._message(item)["attachments"],
            }
            for item in found
            if item.discussion_id in mine
        ]

    def _write_directories(self, execution: ExecutionEnvironment) -> list[str]:
        roots = (
            self._deps.workspace_tree_for(self._actor.member_id).root,
            self._deps.library_tree.root,
        )
        for root in roots:
            root.mkdir(parents=True, exist_ok=True)
        return [*execution.write_directories, *(str(root) for root in roots)]

    @contextmanager
    def _library_updates(self) -> Iterator[None]:
        library = self._library()
        before = library.snapshot()
        try:
            yield
        finally:
            after = library.snapshot()
            for path in library.changes(before, after):
                if path not in after:
                    self._changed("library.updated", {"path": path, "deleted": True})
                    continue
                self._changed(
                    "library.updated", {"path": path, "hash": self._library_hash(path)}
                )

    def run(
        self,
        argv: Sequence[str],
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        self._check("run")
        execution = self._deps.execution.snapshot()
        base = str(self._deps.agent_directory_for(self._actor.member_id))
        cwd = base if cwd is None else execution.resolve_path(cwd, base=base)
        directories = self._write_directories(execution)
        with self._library_updates():
            result = execution.run(
                list(argv), cwd=cwd, timeout=timeout, write_directories=directories
            )
        self._record("run", f"{' '.join(argv)} exited {result.exit_code}")
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "truncated": result.truncated,
        }

    def edit(
        self,
        path: str,
        edits: Sequence[Mapping[str, object]],
        create: bool = False,
    ) -> dict[str, Any]:
        self._check("edit", path)
        execution = self._deps.execution.snapshot()
        path = execution.resolve_path(
            path, base=str(self._deps.agent_directory_for(self._actor.member_id))
        )
        directories = self._write_directories(execution)
        with self._library_updates():
            result = execution.edit(
                path,
                edits,
                write_directories=directories,
                create=create,
            )
        summary = (
            f"{result.path} (created)"
            if create
            else f"{result.path} ({result.replacements} replaced)"
        )
        self._record("edit", summary)
        return {
            "path": result.path,
            "diff": result.diff,
            "replacements": result.replacements,
            "created": create,
        }

    def _workspace(self, agent_id: int | None = None) -> Workspace:
        if agent_id is None:
            agent_id = self._actor.member_id
        if self._actor.is_agent and agent_id != self._actor.member_id:
            raise DomainError("not_allowed", "Workspace is private")
        return Workspace(self._deps.workspace_tree_for(agent_id))

    def list_workspace(
        self, path: str | None = None, *, agent_id: int | None = None
    ) -> list[dict[str, Any]]:
        self._check("workspace.list")
        return [
            {
                "path": item.path,
                "kind": item.kind,
                "size": item.size,
                "modified_at": item.modified_at,
            }
            for item in self._workspace(agent_id).list(path)
        ]

    def read_workspace(
        self, path: str, *, agent_id: int | None = None
    ) -> dict[str, Any]:
        self._check("workspace.read", path)
        content, digest = self._workspace(agent_id).read(path)
        return {"path": path, "content": content, "hash": digest}

    @property
    def library_root(self) -> Path:
        return self._deps.library_tree.root

    def _library(self) -> Library:
        return Library(self._deps.library_tree)

    def _library_hash(self, path: str) -> str | None:
        try:
            return self._library().read(path).content_hash
        except DomainError:
            return None

    def list_library(self, path: str | None = None) -> list[dict[str, Any]]:
        self._check("library.list")
        return [
            {
                "path": item.path,
                "kind": item.kind,
                "size": item.size,
                "modified_at": item.modified_at,
            }
            for item in self._library().list(path)
        ]

    def read_library(self, path: str) -> dict[str, Any]:
        self._check("library.read", path)
        document = self._library().read(path)
        return {
            "path": document.path,
            "content": document.content,
            "hash": document.content_hash,
        }

    def write_library(
        self, path: str, content: str, expected_hash: str | None = None
    ) -> dict[str, Any]:
        self._check("library.write", path)
        try:
            entry = self._library().write(path, content, expected_hash=expected_hash)
        except ConflictError as conflict:
            current, digest = self._library()._tree.read(path)
            return {
                "conflict": True,
                "path": conflict.path,
                "current_hash": digest,
                "current_content": current,
            }
        self._record("library.write", entry.path)
        return self._changed(
            "library.updated",
            {"path": entry.path, "hash": self._library_hash(entry.path)},
        )

    def edit_library(
        self, path: str, old_text: str, new_text: str, replace_all: bool = False
    ) -> dict[str, Any]:
        self._check("library.edit", path)
        entry, diff = self._library().edit(
            path, old_text, new_text, replace_all=replace_all
        )
        self._record("library.edit", entry.path)
        updated = self._changed(
            "library.updated",
            {"path": entry.path, "hash": self._library_hash(entry.path)},
        )
        return {**updated, "diff": diff}

    def mkdir_library(self, path: str) -> dict[str, Any]:
        self._check("library.mkdir", path)
        entry = self._library().mkdir(path)
        self._record("library.mkdir", entry.path)
        return self._changed("library.updated", {"path": entry.path, "hash": None})

    def delete_library(self, path: str) -> dict[str, Any]:
        self._check("library.delete", path)
        self._library().delete(path)
        self._record("library.delete", path)
        return self._changed("library.updated", {"path": path, "deleted": True})

    def move_library(self, source: str, destination: str) -> dict[str, Any]:
        self._check("library.move", source)
        entry = self._library().move(source, destination)
        self._record("library.move", f"{source} to {entry.path}")
        self._changed("library.updated", {"path": source, "deleted": True})
        return self._changed(
            "library.updated",
            {"path": entry.path, "hash": self._library_hash(entry.path)},
        )

    def _history(self) -> History:
        return History(self._deps.history, self._actor.member_id)

    def search_history(self, query: str) -> list[dict[str, Any]]:
        self._check("history.search")
        return [
            {
                "sequence": item.sequence,
                "status": item.status,
                "started_at": item.started_at,
            }
            for item in self._history().search(query)
        ]

    def read_history(self, sequence: int, offset: int | None = None) -> dict[str, Any]:
        self._check("history.read", sequence)
        run = self._history().read(sequence, offset)
        if run is None:
            raise DomainError("not_found", f"Run {sequence} does not exist")
        next_offset = (
            run.offset + len(run.messages)
            if run.offset + len(run.messages) < run.total_length
            else None
        )
        return {
            "sequence": run.sequence,
            "status": run.status,
            "started_at": run.started_at,
            "messages": run.messages,
            "offset": run.offset,
            "total_length": run.total_length,
            "next_offset": next_offset,
        }
