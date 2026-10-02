from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, TypeAdapter
from pydantic_ai import BinaryContent, ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage

from fora.core.errors import DomainError

PathPart = str | int
TEXT_PAGE_BYTES = 16 * 1024


def decode_history(raw: str) -> list[ModelMessage]:
    try:
        return ModelMessagesTypeAdapter.validate_json(raw)
    except ValueError as error:
        raise DomainError(
            "history_invalid", "Stored model history is invalid"
        ) from error


def render_history(
    raw: str,
    *,
    source: str,
    request_ordinal: int | None = None,
    offset: int = 0,
    limit: int = 100,
    message_index_base: int = 0,
) -> dict[str, Any]:
    messages = decode_history(raw)
    total = len(messages)
    if offset > total:
        raise DomainError("invalid_offset", "History message offset exceeds its length")
    selected = messages[offset : offset + limit]
    return {
        "messages": [
            _render_value(
                message,
                source=source,
                request_ordinal=request_ordinal,
                message_index=index + message_index_base,
                path=(),
            )
            for index, message in enumerate(selected, start=offset)
        ],
        "offset": offset,
        "total": total,
        "has_more": offset + len(selected) < total,
    }


def binary_from_history(
    raw: str, message_index: int, path: list[PathPart]
) -> BinaryContent:
    messages = decode_history(raw)
    if message_index < 0 or message_index >= len(messages):
        raise DomainError("not_found", "History message does not exist")
    value: Any = messages[message_index]
    for part in path:
        if isinstance(value, BaseModel):
            if not isinstance(part, str) or part not in value.model_fields:
                raise DomainError("not_found", "History binary content does not exist")
            value = getattr(value, part)
        elif is_dataclass(value) and not isinstance(value, type):
            if not isinstance(part, str) or part not in {
                field.name for field in fields(value)
            }:
                raise DomainError("not_found", "History binary content does not exist")
            value = getattr(value, part)
        elif isinstance(value, dict):
            if part not in value:
                raise DomainError("not_found", "History binary content does not exist")
            value = value[part]
        elif isinstance(value, list):
            if not isinstance(part, int) or part < 0 or part >= len(value):
                raise DomainError("not_found", "History binary content does not exist")
            value = value[part]
        else:
            raise DomainError("not_found", "History binary content does not exist")
    if not isinstance(value, BinaryContent):
        raise DomainError("not_found", "History binary content does not exist")
    return value


def _text_page(
    value: str, offset: int = 0, limit: int = TEXT_PAGE_BYTES
) -> dict[str, Any] | str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit and offset == 0:
        return value
    start = min(max(offset, 0), len(encoded))
    while start < len(encoded) and encoded[start] & 0xC0 == 0x80:
        start += 1
    end = min(start + limit, len(encoded))
    while end < len(encoded) and end > start and encoded[end] & 0xC0 == 0x80:
        end -= 1
    return {
        "kind": "text",
        "source": "history",
        "offset": start,
        "next_offset": end,
        "total_bytes": len(encoded),
        "value": encoded[start:end].decode("utf-8"),
        "has_more": end < len(encoded),
    }


def text_from_history(
    raw: str,
    message_index: int,
    path: list[PathPart],
    offset: int,
    limit: int,
) -> dict[str, Any]:
    messages = decode_history(raw)
    if message_index < 0 or message_index >= len(messages):
        raise DomainError("not_found", "History message does not exist")
    value: Any = messages[message_index]
    for part in path:
        if isinstance(value, BaseModel):
            if not isinstance(part, str) or part not in value.model_fields:
                raise DomainError("not_found", "History text does not exist")
            value = getattr(value, part)
        elif is_dataclass(value) and not isinstance(value, type):
            if not isinstance(part, str) or part not in {
                field.name for field in fields(value)
            }:
                raise DomainError("not_found", "History text does not exist")
            value = getattr(value, part)
        elif isinstance(value, dict):
            if part not in value:
                raise DomainError("not_found", "History text does not exist")
            value = value[part]
        elif isinstance(value, list):
            if not isinstance(part, int) or part < 0 or part >= len(value):
                raise DomainError("not_found", "History text does not exist")
            value = value[part]
        else:
            raise DomainError("not_found", "History text does not exist")
    if not isinstance(value, str):
        raise DomainError("not_found", "History text does not exist")
    page = _text_page(value, offset, limit)
    if isinstance(page, str):
        return {
            "offset": 0,
            "next_offset": len(value.encode("utf-8")),
            "total_bytes": len(value.encode("utf-8")),
            "value": page,
            "has_more": False,
        }
    return page


def _render_value(
    value: Any,
    *,
    source: str,
    request_ordinal: int | None,
    message_index: int,
    path: tuple[PathPart, ...],
) -> Any:
    if isinstance(value, BinaryContent):
        result: dict[str, Any] = {
            "kind": "binary",
            "source": source,
            "message_index": message_index,
            "path": list(path),
            "media_type": value.media_type,
            "size": len(value.data),
            "identifier": value.identifier,
        }
        if request_ordinal is not None:
            result["request_ordinal"] = request_ordinal
        return result
    if isinstance(value, BaseModel):
        return {
            name: _render_value(
                getattr(value, name),
                source=source,
                request_ordinal=request_ordinal,
                message_index=message_index,
                path=(*path, name),
            )
            for name in value.model_fields
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _render_value(
                getattr(value, field.name),
                source=source,
                request_ordinal=request_ordinal,
                message_index=message_index,
                path=(*path, field.name),
            )
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(name): _render_value(
                child,
                source=source,
                request_ordinal=request_ordinal,
                message_index=message_index,
                path=(*path, str(name)),
            )
            for name, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _render_value(
                child,
                source=source,
                request_ordinal=request_ordinal,
                message_index=message_index,
                path=(*path, index),
            )
            for index, child in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        return [
            _render_value(
                child,
                source=source,
                request_ordinal=request_ordinal,
                message_index=message_index,
                path=(*path, index),
            )
            for index, child in enumerate(sorted(value, key=str))
        ]
    if isinstance(value, Enum):
        return _render_value(
            value.value,
            source=source,
            request_ordinal=request_ordinal,
            message_index=message_index,
            path=path,
        )
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        page = _text_page(value)
        if isinstance(page, dict):
            page.update(
                {
                    "source": source,
                    "message_index": message_index,
                    "path": list(path),
                }
            )
            if request_ordinal is not None:
                page["request_ordinal"] = request_ordinal
        return page
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return TypeAdapter(Any).dump_python(value, mode="json")
