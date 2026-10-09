from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

from fora.core.discussion import Message
from fora.core.errors import DomainError
from fora.ports.store import OrganizationStore, ReadSummary

SECTIONS = (
    "messages",
    "members",
    "awaiting_ack",
    "acknowledged",
    "mentions",
    "attachments",
)


def result_length(result: dict[str, Any]) -> int:
    return len(
        json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    )


def integer(name: str, value: object, low: int, high: int = 2**63 - 1) -> int:
    if type(value) is not int or not low <= value <= high:
        raise DomainError(
            "invalid_pagination", f"{name} must be an integer from {low} to {high}"
        )
    return cast(int, value)


@dataclass(frozen=True)
class Position:
    d: int
    s: str
    l: int
    h: int
    r: bool
    a: int
    m: int
    o: int
    w: int
    t: int
    v: int = 1

    def encode(self) -> str:
        return base64.urlsafe_b64encode(
            json.dumps(asdict(self), separators=(",", ":")).encode()
        ).decode()

    @classmethod
    def decode(cls, cursor: str, discussion_id: int, section: str) -> Position:
        if type(cursor) is not str or not 1 <= len(cursor) <= 2048:
            raise DomainError(
                "invalid_cursor", "cursor must contain 1 to 2048 characters"
            )
        try:
            data = json.loads(
                base64.b64decode(cursor.encode("ascii"), altchars=b"-_", validate=True)
            )
        except (ValueError, UnicodeError, binascii.Error) as error:
            raise DomainError(
                "invalid_cursor", "cursor is not valid encoded JSON"
            ) from error
        if type(data) is not dict or set(data) != {
            "d",
            "s",
            "l",
            "h",
            "r",
            "a",
            "m",
            "o",
            "w",
            "t",
            "v",
        }:
            raise DomainError("invalid_cursor", "cursor has invalid fields")
        for key in ("d", "l", "h", "a", "m", "o", "w", "t", "v"):
            if type(data[key]) is not int or not 0 <= data[key] <= 2**63 - 1:
                raise DomainError(
                    "invalid_cursor", "cursor positions must be nonnegative integers"
                )
        if (
            data["v"] != 1
            or data["d"] != discussion_id
            or data["s"] != section
            or type(data["r"]) is not bool
        ):
            raise DomainError(
                "invalid_cursor", "cursor version, Discussion or section does not match"
            )
        if data["l"] > data["h"] + 1 or data["h"] > data["t"]:
            raise DomainError("invalid_cursor", "cursor range is invalid")
        if section == "messages" and (
            data["m"] and not data["l"] <= data["m"] <= data["h"]
        ):
            raise DomainError("invalid_cursor", "cursor message is outside its range")
        if data["a"] or (section == "messages" and data["o"] and not data["m"]):
            raise DomainError("invalid_cursor", "cursor message position is invalid")
        if section in ("mentions", "attachments") and not data["m"]:
            raise DomainError("invalid_cursor", "metadata cursor requires a message")
        if section in ("members", "awaiting_ack", "acknowledged") and data["m"]:
            raise DomainError(
                "invalid_cursor", "metadata cursor cannot select a message"
            )
        return cls(**data)


class DiscussionReader:
    def __init__(
        self,
        store: OrganizationStore,
        summary: ReadSummary,
        member_id: int,
        limit: int,
        max_chars: int,
        attachment_path: Callable[[str], str] | None,
        measure: Callable[[dict[str, Any]], int] | None,
    ) -> None:
        self.store = store
        self.summary = summary
        self.member_id = member_id
        self.limit = limit
        self.max_chars = max_chars
        self.attachment_path = attachment_path
        self.measure = measure

    def size(self, result: dict[str, Any]) -> int:
        return max(result_length(result), self.measure(result) if self.measure else 0)

    def require_budget(self, result: dict[str, Any]) -> None:
        size = self.size(result)
        if size > self.max_chars:
            raise DomainError(
                "read_budget", f"Necessary result requires max_chars >= {size}"
            )

    def request(self, position: Position) -> dict[str, Any]:
        return {
            "action": "read",
            "discussion_id": position.d,
            "section": position.s,
            "cursor": position.encode(),
            "limit": self.limit,
            "max_chars": self.max_chars,
        }

    def base(self, section: str) -> dict[str, Any]:
        item = self.summary
        return {
            "id": item.id,
            "topic": item.topic,
            "archived": item.archived,
            "section": section,
            "read_through": item.read_through,
            "total_messages": item.total_messages,
            "continuations": [],
        }

    def metadata_items(
        self, section: str, message_id: int | None, offset: int, limit: int
    ) -> tuple[int, tuple[object, ...]]:
        total, items = self.store.read_metadata(
            self.summary.id,
            self.member_id,
            section,
            message_id=message_id,
            offset=offset,
            limit=limit,
        )
        if section == "attachments" and self.attachment_path is not None:
            items = tuple(
                cast(dict[str, Any], item)
                | {"path": self.attachment_path(cast(dict[str, Any], item)["id"])}
                for item in items
            )
        return total, items

    def metadata(self, position: Position) -> dict[str, Any]:
        total, candidates = self.metadata_items(
            position.s, position.m or None, position.o, self.limit
        )
        if position.o > total:
            raise DomainError("invalid_cursor", "metadata offset is out of range")
        result = self.base(position.s)
        if position.m:
            message = self.store.read_message(position.d, position.m)
            result.update(
                message_id=message.id,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
            )
        result["total"] = total
        result["offset"] = position.o
        result["realtime"] = position.s in ("members", "awaiting_ack", "acknowledged")
        accepted: list[object] = []
        for candidate in candidates:
            trial = [*accepted, candidate]
            end = position.o + len(trial)
            result[position.s] = trial
            result["complete"] = end == total
            result["continuations"] = (
                [self.request(replace(position, o=end))] if end < total else []
            )
            if self.size(result) > self.max_chars:
                if not accepted:
                    self.require_budget(result)
                break
            accepted = trial
        end = position.o + len(accepted)
        result[position.s] = accepted
        result["complete"] = end == total
        result["continuations"] = (
            [self.request(replace(position, o=end))] if end < total else []
        )
        self.require_budget(result)
        return result

    def array(
        self, section: str, position: Position, message_id: int | None, count: int = 0
    ) -> tuple[list[object], dict[str, Any]]:
        total, candidates = self.metadata_items(section, message_id, 0, count)
        state = replace(position, s=section, m=message_id or 0, o=0, a=0)
        return list(candidates), {
            "total": total,
            "complete": len(candidates) == total,
            "continuation": self.request(replace(state, o=len(candidates)))
            if len(candidates) < total
            else None,
        }

    def bare(
        self, message: Message, position: Position, offset: int, end: int
    ) -> dict[str, Any]:
        item: dict[str, Any] = {
            "id": message.id,
            "sender_id": message.sender_id,
            "sender_name": message.sender_name,
            "created_at": message.created_at,
            "body": message.body[offset:end],
            "body_offset": offset,
            "body_end": end,
            "body_length": len(message.body),
            "body_complete": end == len(message.body),
        }
        for section in ("mentions", "attachments"):
            item[section], item[f"{section}_info"] = self.array(
                section, position, message.id
            )
        item["mention_position_scope"] = "complete_body"
        return item

    def context_ids(self, position: Position) -> tuple[int, ...]:
        left = self.store.read_message_ids(
            position.d, position.l, position.a - 1, limit=self.limit - 1, reverse=True
        )
        right = self.store.read_message_ids(
            position.d, position.a + 1, position.h, limit=self.limit - 1, reverse=False
        )
        before = min(len(left), self.limit // 2)
        after = min(len(right), (self.limit - 1) // 2)
        remaining = self.limit - 1 - before - after
        extra = min(remaining, len(left) - before)
        before += extra
        after += min(remaining - extra, len(right) - after)
        return (*reversed(left[:before]), position.a, *right[:after])

    def continuations(
        self, position: Position, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        first, last = messages[0]["id"], messages[-1]["id"]
        partial = next((item for item in messages if not item["body_complete"]), None)
        requests: list[dict[str, Any]] = []
        if position.a:
            if self.store.read_message_ids(
                position.d, position.l, first - 1, limit=1, reverse=True
            ):
                requests.append(
                    self.request(replace(position, a=0, h=first - 1, r=True, m=0, o=0))
                )
            if self.store.read_message_ids(
                position.d, last + 1, position.h, limit=1, reverse=False
            ):
                requests.append(
                    self.request(replace(position, a=0, l=last + 1, r=False, m=0, o=0))
                )
            if partial:
                requests.append(
                    self.request(
                        replace(
                            position,
                            a=0,
                            l=partial["id"],
                            h=partial["id"],
                            r=False,
                            m=partial["id"],
                            o=partial["body_end"],
                        )
                    )
                )
        elif partial:
            next_position = replace(position, m=partial["id"], o=partial["body_end"])
            next_position = (
                replace(next_position, h=partial["id"])
                if position.r
                else replace(next_position, l=partial["id"])
            )
            requests.append(self.request(next_position))
        else:
            next_position = (
                replace(position, m=0, o=0, h=first - 1)
                if position.r
                else replace(position, m=0, o=0, l=last + 1)
            )
            if self.store.read_message_ids(
                position.d,
                next_position.l,
                next_position.h,
                limit=1,
                reverse=position.r,
            ):
                requests.append(self.request(next_position))
        return requests

    def messages_result(
        self, position: Position, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        result = self.base("messages")
        awaiting, acked = self.store.read_ack_flags(
            position.d, self.member_id, [item["id"] for item in messages]
        )
        result.update(
            messages=messages,
            awaiting_ack=list(awaiting),
            acknowledged=list(acked),
            ack_scope="returned_messages",
            awaiting_ack_count=self.summary.awaiting_ack_count,
            acknowledged_count=self.summary.acknowledged_count,
            direction="before" if position.r else "after",
            snapshot_max_id=position.t,
            context_target=position.a or None,
            continuations=self.continuations(position, messages),
        )
        result["members"], result["members_info"] = self.array(
            "members", position, None
        )
        for section in ("awaiting_ack", "acknowledged"):
            total = getattr(self.summary, f"{section}_count")
            result[f"{section}_info"] = {
                "total": total,
                "complete": len(result[section]) == total,
                "realtime": True,
                "continuation": self.request(
                    replace(position, s=section, a=0, m=0, o=0)
                )
                if len(result[section]) != total
                else None,
            }
        return result

    def add_metadata(self, result: dict[str, Any], position: Position) -> None:
        locations = [
            (message, section, message["id"])
            for message in result["messages"]
            for section in ("mentions", "attachments")
        ]
        locations.append((result, "members", None))
        for container, section, message_id in locations:
            total, candidates = self.metadata_items(section, message_id, 0, self.limit)
            related = {message["sender_id"] for message in result["messages"]}
            related.update(
                mention["member_id"]
                for message in result["messages"]
                for mention in message["mentions"]
            )
            accepted: list[object] = []
            for candidate in candidates:
                if (
                    section == "members"
                    and cast(dict[str, Any], candidate)["id"] not in related
                ):
                    break
                info = container[f"{section}_info"]
                old_info = dict(info)
                container[section] = [*accepted, candidate]
                n = len(accepted) + 1
                info.update(
                    complete=n == total,
                    continuation=self.request(
                        replace(position, s=section, a=0, m=message_id or 0, o=n)
                    )
                    if n < total
                    else None,
                )
                if self.size(result) > self.max_chars:
                    container[section] = accepted
                    container[f"{section}_info"] = old_info
                    break
                accepted.append(candidate)

    def messages(self, position: Position) -> dict[str, Any]:
        if position.t > self.summary.latest_id:
            raise DomainError(
                "invalid_cursor", "cursor snapshot is beyond this Discussion"
            )
        if position.a:
            ids = self.context_ids(position)
            anchor = ids.index(position.a)
            order = sorted(
                ids, key=lambda item: (abs(ids.index(item) - anchor), item > position.a)
            )
        elif position.m:
            order = [position.m]
        else:
            order = list(
                self.store.read_message_ids(
                    position.d,
                    position.l,
                    position.h,
                    limit=self.limit,
                    reverse=position.r,
                )
            )
        chosen: list[dict[str, Any]] = []
        for message_id in order:
            message = self.store.read_message(position.d, message_id)
            offset = position.o if message_id == position.m else 0
            if offset > len(message.body) or (
                offset == len(message.body) and offset != 0
            ):
                raise DomainError("invalid_cursor", "body offset is out of range")
            full = self.bare(message, position, offset, len(message.body))
            trial = sorted([*chosen, full], key=lambda item: item["id"])
            result = self.messages_result(position, trial)
            if self.size(result) <= self.max_chars:
                chosen = trial
                if position.m:
                    break
                continue
            if chosen:
                break
            minimum = self.bare(
                message, position, offset, min(offset + 1, len(message.body))
            )
            self.require_budget(self.messages_result(position, [minimum]))
            low, high = minimum["body_end"], len(message.body)
            best = minimum
            while low <= high:
                end = (low + high) // 2
                part = self.bare(message, position, offset, end)
                if self.size(self.messages_result(position, [part])) <= self.max_chars:
                    best = part
                    low = end + 1
                else:
                    high = end - 1
            chosen = [best]
            break
        result = self.messages_result(position, chosen)
        self.require_budget(result)
        self.add_metadata(result, position)
        self.require_budget(result)
        return result


def read_position(
    summary: ReadSummary,
    member_id: int,
    store: OrganizationStore,
    section: str,
    message_id: int | None,
    before: int | None,
    after: int | None,
    cursor: str | None,
) -> Position:
    if type(section) is not str or section not in SECTIONS:
        raise DomainError("invalid_pagination", "Unknown read section")
    for name, value, low in (
        ("message_id", message_id, 1),
        ("before", before, 0),
        ("after", after, 0),
    ):
        if value is not None:
            integer(name, value, low)
    if cursor is not None:
        if message_id is not None or before is not None or after is not None:
            raise DomainError(
                "invalid_cursor", "cursor cannot be combined with message_id or bounds"
            )
        position = Position.decode(cursor, summary.id, section)
        if position.t > summary.latest_id:
            raise DomainError(
                "invalid_cursor", "cursor snapshot is beyond this Discussion"
            )
        return position
    if message_id is not None and (before is not None or after is not None):
        raise DomainError(
            "invalid_pagination", "message_id cannot be combined with bounds"
        )
    if before is not None and after is not None and before <= after:
        raise DomainError("invalid_pagination", "before must exceed after")
    if section != "messages":
        if before is not None or after is not None:
            raise DomainError("invalid_pagination", "metadata uses cursor pagination")
        if section in ("mentions", "attachments"):
            if message_id is None:
                raise DomainError(
                    "invalid_pagination", "This section requires message_id"
                )
            store.read_message(summary.id, message_id)
        elif message_id is not None:
            raise DomainError(
                "invalid_pagination",
                "message_id only selects messages, mentions or attachments",
            )
        return Position(
            summary.id,
            section,
            0,
            summary.latest_id,
            False,
            0,
            message_id or 0,
            0,
            summary.read_through,
            summary.latest_id,
        )
    if message_id is not None:
        low, high = store.read_context_bounds(
            summary.id, message_id, member_id, summary.read_through
        )
    else:
        high = max(
            0,
            min(
                summary.latest_id,
                before - 1 if before is not None else summary.latest_id,
            ),
        )
        low = min((after or 0) + 1, high + 1)
    return Position(
        summary.id,
        section,
        low,
        high,
        after is None,
        message_id or 0,
        0,
        0,
        summary.read_through,
        summary.latest_id,
    )
