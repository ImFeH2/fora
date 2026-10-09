from __future__ import annotations

import base64
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict
from functools import partial
from typing import Any

from fora.adapters.files.tree import relative_path
from fora.adapters.jsonl.protocol import Dispatcher, RequestScope, Resource
from fora.adapters.model.config import ModelCatalog
from fora.adapters.model.history import (
    binary_from_history,
    render_history,
    text_from_history,
)
from fora.adapters.voice.config import VoiceConfig
from fora.core.errors import DomainError
from fora.core.parameters import agent_parameters
from fora.core.turn import idle_streak
from fora.runtime.scheduler import Scheduler
from fora.runtime_info import RuntimeInfo
from fora.tools import AgentTools
from fora.tools.authorize import Actor

HUMAN_ID = 1

ModelProbe = Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]]


def _integer_param(
    params: dict[str, Any],
    name: str,
    *,
    default: int | None = None,
    maximum: int | None = None,
) -> int | None:
    value = params.get(name, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DomainError("invalid_params", f"{name} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise DomainError("invalid_params", f"{name} exceeds its maximum")
    return value


def _required_integer(params: dict[str, Any], name: str) -> int:
    value = _integer_param(params, name)
    if value is None:
        raise DomainError("invalid_params", f"{name} is required")
    return value


def _history_run_payload(run: Any) -> dict[str, Any]:
    return {
        "sequence": run.sequence,
        "run_id": run.run_id,
        "status": run.status,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "usage": run.usage_json,
        "error": run.error,
        "window_number": run.window_number,
        "window_reset_at": run.window_reset_at,
        "window_reason": run.window_reason,
        "request_count": run.request_count,
        "last_saved_at": run.last_saved_at,
        "legacy": run.request_count == 0,
    }


def _window_event_payload(event: Any) -> dict[str, Any]:
    return {
        "number": event.number,
        "since_sequence": event.since_sequence,
        "reset_at": event.reset_at,
        "reason": event.reason,
    }


def _request_summary_payload(summary: Any) -> dict[str, Any]:
    return {
        "ordinal": summary.ordinal,
        "request_id": summary.request_id,
        "run_id": summary.run_id,
        "window_number": summary.window_number,
        "status": summary.status,
        "started_at": summary.started_at,
        "completed_at": summary.completed_at,
        "streaming": summary.streaming,
        "input_length": summary.input_length,
        "parameters_length": summary.parameters_length,
        "settings_length": summary.settings_length,
        "model_length": summary.model_length,
        "response_length": summary.response_length,
        "input_count": summary.input_count,
        "response_count": summary.response_count,
        "related_count": summary.related_count,
        "error": summary.error,
    }


def _render_message_page(
    page: Any, source: str, request_ordinal: int
) -> dict[str, Any]:
    messages = [
        render_history(
            raw,
            source=source,
            request_ordinal=request_ordinal,
            limit=1,
            message_index_base=page.offset + index,
        )["messages"][0]
        for index, raw in enumerate(page.messages)
    ]
    return {
        "messages": messages,
        "offset": page.offset,
        "total": page.total,
        "total_bytes": page.total_bytes,
        "has_more": page.has_more,
    }


def _render_model_field_page(
    page: Any, source: str, request_ordinal: int
) -> dict[str, Any]:
    value_bytes = len(page.value.encode("utf-8"))
    return {
        "kind": "text",
        "source": source,
        "request_ordinal": request_ordinal,
        "message_index": 0,
        "path": [],
        "offset": page.offset,
        "next_offset": page.offset + value_bytes,
        "total_bytes": page.total_bytes,
        "value": page.value,
        "has_more": page.has_more,
    }


def _list_models(values: dict[str, Any], stored: dict[str, Any] | None) -> Any:
    from fora.adapters.model.probe import list_models

    return list_models(values, stored)


def _test_model(values: dict[str, Any], stored: dict[str, Any] | None) -> Any:
    from fora.adapters.model.probe import try_model

    return try_model(values, stored)


class Api:
    def __init__(
        self,
        scheduler: Scheduler,
        dispatcher: Dispatcher,
        *,
        list_models: ModelProbe = _list_models,
        test_model: ModelProbe = _test_model,
        runtime_info: RuntimeInfo | None = None,
        reading: Callable[[], AbstractContextManager[None]] = nullcontext,
    ) -> None:
        self._scheduler = scheduler
        self._dispatcher = dispatcher
        self._list_models = list_models
        self._test_model = test_model
        self._runtime_info = runtime_info or RuntimeInfo.capture()
        self._reading = reading
        self._register()

    def _human(self) -> AgentTools:
        return self._scheduler.tools_for_actor(Actor(HUMAN_ID, False))

    def _changed(self, event: str, payload: dict[str, Any]) -> None:
        self._dispatcher.emit(event, payload)
        self._scheduler.wake()

    def _register(self) -> None:
        register = self._dispatcher.register
        settings = self._scheduler.settings

        def info_get(params: dict[str, Any]) -> Any:
            del params
            return self._runtime_info.as_dict()

        def organization_get(params: dict[str, Any]) -> Any:
            del params
            with self._reading():
                members = self._human().list_member_records()
                for member in members:
                    if member["type"] == "agent":
                        usage = self._scheduler.history.usage_total(int(member["id"]))
                        member["tokens"] = usage["total_tokens"]
                organization_uuid = self._human().organization_uuid()
                statistics = self._scheduler.statistics()
            for member in members:
                if member["type"] == "agent":
                    member.update(self._scheduler.agent_status(int(member["id"])))
            return {
                "id": 1,
                "uuid": organization_uuid,
                "members": members,
                "human_id": HUMAN_ID,
                **statistics,
            }

        def create_agent(params: dict[str, Any]) -> Any:
            return self._human().create_agent(
                str(params.get("name", "")), params.get("model_config")
            )

        def rename_member(params: dict[str, Any]) -> Any:
            result = self._human().rename_member(
                int(params["member_id"]), str(params.get("name", ""))
            )
            self._changed("member.updated", result)
            return result

        def pause_agent(params: dict[str, Any]) -> Any:
            result = self._human().pause_agent(int(params["agent_id"]))
            self._changed("member.updated", result)
            return result

        def resume_agent(params: dict[str, Any]) -> Any:
            result = self._human().resume_agent(int(params["agent_id"]))
            self._changed("member.updated", result)
            return result

        def delete_agent(params: dict[str, Any]) -> Any:
            result = self._human().delete_agent(int(params["agent_id"]))
            self._changed("member.deleted", result)
            return result

        def discussion_create(params: dict[str, Any]) -> Any:
            return self._human().create_discussion(
                str(params.get("topic", "")), list(params.get("member_ids", []))
            )

        def discussion_list(params: dict[str, Any]) -> Any:
            return self._human().list_discussions(
                bool(params.get("include_archived", False)), limit=params.get("limit")
            )

        def discussion_read(params: dict[str, Any]) -> Any:
            return self._human().read_discussion(
                params["discussion_id"],
                params.get("message_id"),
                params.get("limit"),
                before=params.get("before"),
                after=params.get("after"),
                max_chars=params.get("max_chars"),
                section=params.get("section", "messages"),
                cursor=params.get("cursor"),
            )

        def discussion_page(params: dict[str, Any]) -> Any:
            return self._human().discussion_page(
                params["discussion_id"],
                limit=params.get("limit", 50),
                entry=params.get("entry", False),
                before=params.get("before"),
                after=params.get("after"),
                metadata=params.get("metadata", False),
            )

        def discussion_mark_read(params: dict[str, Any]) -> Any:
            result = self._human().mark_read(
                params["discussion_id"], params["message_id"]
            )
            self._dispatcher.emit("discussion.read_updated", result)
            return result

        def discussion_ack_pending(params: dict[str, Any]) -> Any:
            result = self._human().ack_pending(
                params["discussion_id"], params["through_message_id"]
            )
            self._dispatcher.emit(
                "discussion.read_updated",
                {
                    "discussion_id": params["discussion_id"],
                    "member_id": HUMAN_ID,
                    "read_through": result["read_through"],
                },
            )
            return result

        def discussion_send(params: dict[str, Any]) -> Any:
            return self._human().send_message(
                int(params["discussion_id"]),
                str(params.get("body", "")),
                mark_read=params.get("mark_read", True),
                attachment_ids=params.get("attachment_ids", []),
                client_message_id=params.get("client_message_id"),
            )

        def upload_create(params: dict[str, Any]) -> Any:
            return self._human().create_upload(
                params["discussion_id"],
                params["client_upload_id"],
                params["name"],
                params["size"],
                params.get("media_type", ""),
            )

        member_set = Resource("member")
        discussion_set = Resource("discussion")
        model_settings = Resource("settings", "model")
        attachments = Resource("attachments", HUMAN_ID)

        def independent(params: dict[str, Any]) -> RequestScope:
            return RequestScope()

        def discussion_resource(params: dict[str, Any]) -> Resource:
            return Resource("discussion", int(params["discussion_id"]))

        def discussion_query(params: dict[str, Any]) -> RequestScope:
            return RequestScope(reads=(discussion_resource(params), member_set))

        def discussion_change(params: dict[str, Any]) -> RequestScope:
            return RequestScope(
                reads=(member_set,), writes=(discussion_resource(params),)
            )

        def discussion_send_scope(params: dict[str, Any]) -> RequestScope:
            scope = discussion_change(params)
            if params.get("attachment_ids"):
                return RequestScope(
                    scope.reads + (attachments,), scope.writes + (attachments,)
                )
            return scope

        def agent_query(params: dict[str, Any]) -> RequestScope:
            identity = _required_integer(params, "agent_id")
            return RequestScope(
                reads=(Resource("member", identity), Resource("history", identity))
            )

        def agent_change(params: dict[str, Any]) -> RequestScope:
            return RequestScope(writes=(Resource("member", int(params["agent_id"])),))

        def workspace_scope(params: dict[str, Any]) -> RequestScope:
            identity = int(params["agent_id"])
            return RequestScope(
                reads=(Resource("member", identity),),
                writes=(Resource("workspace", identity),),
            )

        def detail_scope(params: dict[str, Any]) -> RequestScope:
            scope = workspace_scope(params)
            return RequestScope(
                scope.reads
                + (
                    Resource("history", int(params["agent_id"])),
                    model_settings,
                    Resource("settings", "agent"),
                ),
                scope.writes,
            )

        def setting_query(params: dict[str, Any]) -> RequestScope:
            section = str(params.get("section", "model"))
            resource = Resource("settings", section)
            return (
                RequestScope(writes=(resource,))
                if section == "voice"
                else RequestScope(reads=(resource,))
            )

        def setting_change(params: dict[str, Any]) -> RequestScope:
            section = str(params.get("section", "model"))
            return RequestScope(
                reads=(member_set,) if section == "model" else (),
                writes=(Resource("settings", section),),
            )

        def library_scope(
            params: dict[str, Any],
            *,
            writing: bool = False,
            directory: bool = False,
            moving: bool = False,
        ) -> RequestScope:
            path = (
                params.get("path") if directory and not writing else str(params["path"])
            )
            root = self._human().library_root

            def identities(value: str, descendants: bool) -> tuple[Resource, ...]:
                target = root / relative_path(value, allow_root=descendants)
                resolved = target.resolve()
                if not resolved.is_relative_to(root):
                    raise DomainError("invalid_path", "Path must stay inside the tree")
                return tuple(
                    dict.fromkeys(
                        (
                            Resource("library", target, descendants),
                            Resource("library", resolved, descendants),
                        )
                    )
                )

            resources = identities(path if path is not None else ".", directory)
            if moving:
                resources += identities(str(params["destination"]), True)
            return (
                RequestScope(writes=resources)
                if writing
                else RequestScope(reads=resources)
            )

        register(
            "upload.create",
            upload_create,
            scope=lambda params: RequestScope(
                writes=(discussion_resource(params), attachments)
            ),
        )
        register(
            "upload.status",
            lambda params: self._human().upload_status(params["upload_ids"]),
            scope=lambda params: RequestScope(writes=(attachments,)),
        )
        register(
            "upload.cancel",
            lambda params: self._human().cancel_uploads(params["upload_ids"]),
            scope=lambda params: RequestScope(writes=(attachments,)),
        )
        register(
            "discussion.send_status",
            lambda params: self._human().send_status(
                params["discussion_id"], params["client_message_id"]
            ),
            scope=discussion_query,
        )

        register(
            "discussion.cancel_send",
            lambda params: self._human().cancel_send(
                params["discussion_id"], params["client_message_id"]
            ),
            scope=discussion_change,
        )

        def discussion_ack(params: dict[str, Any]) -> Any:
            return self._human().ack(
                int(params["discussion_id"]), list(params.get("message_ids", []))
            )

        def discussion_revoke_ack(params: dict[str, Any]) -> Any:
            return self._human().revoke_ack(
                int(params["discussion_id"]), list(params.get("message_ids", []))
            )

        def discussion_members(params: dict[str, Any]) -> Any:
            return self._human().set_discussion_members(
                int(params["discussion_id"]), list(params.get("member_ids", []))
            )

        def discussion_add_members(params: dict[str, Any]) -> Any:
            return self._human().add_members(
                int(params["discussion_id"]), list(params["member_ids"])
            )

        def discussion_remove_members(params: dict[str, Any]) -> Any:
            return self._human().remove_members(
                int(params["discussion_id"]), list(params["member_ids"])
            )

        def discussion_archive(params: dict[str, Any]) -> Any:
            return self._human().archive_discussion(
                int(params["discussion_id"]), bool(params.get("archived", True))
            )

        def discussion_unarchive(params: dict[str, Any]) -> Any:
            return self._human().archive_discussion(int(params["discussion_id"]), False)

        def discussion_search(params: dict[str, Any]) -> Any:
            return self._human().search_messages(
                str(params.get("query", "")),
                sender_id=int(params["sender_id"])
                if params.get("sender_id") is not None
                else None,
                discussion_id=int(params["discussion_id"])
                if params.get("discussion_id") is not None
                else None,
            )

        def library_list(params: dict[str, Any]) -> Any:
            return self._human().list_library(params.get("path"))

        def library_read(params: dict[str, Any]) -> Any:
            return self._human().read_library(str(params["path"]))

        def library_write(params: dict[str, Any]) -> Any:
            return self._human().write_library(
                str(params["path"]),
                str(params.get("content", "")),
                params.get("expected_hash"),
            )

        def library_edit(params: dict[str, Any]) -> Any:
            return self._human().edit_library(
                str(params["path"]),
                str(params["old_text"]),
                str(params["new_text"]),
                bool(params.get("replace_all", False)),
            )

        def library_mkdir(params: dict[str, Any]) -> Any:
            return self._human().mkdir_library(str(params["path"]))

        def workspace_list(params: dict[str, Any]) -> Any:
            return self._human().list_workspace(
                params.get("path"), agent_id=int(params["agent_id"])
            )

        def workspace_read(params: dict[str, Any]) -> Any:
            return self._human().read_workspace(
                str(params["path"]), agent_id=int(params["agent_id"])
            )

        def library_delete(params: dict[str, Any]) -> Any:
            return self._human().delete_library(str(params["path"]))

        def library_move(params: dict[str, Any]) -> Any:
            return self._human().move_library(
                str(params["path"]), str(params["destination"])
            )

        def agent_history(params: dict[str, Any]) -> Any:
            agent_id = _required_integer(params, "agent_id")
            self._human().authorize_agent_history(agent_id)
            limit = _integer_param(params, "limit", default=30, maximum=50)
            windows_after = _integer_param(params, "windows_after", maximum=2**63 - 1)
            windows_limit = _integer_param(
                params, "windows_limit", default=30, maximum=30
            )
            assert limit is not None
            assert windows_limit is not None
            if limit == 0 or windows_limit == 0:
                raise DomainError("invalid_params", "history limits must be positive")
            before = _integer_param(params, "before", maximum=2**63 - 1)
            with self._reading():
                self._human().authorize_agent_history(agent_id)
                runs = self._scheduler.history.history_runs(
                    agent_id, before=before, limit=limit
                )
                window_page = self._scheduler.history.window_events(
                    agent_id, after=windows_after, limit=windows_limit + 1
                )
            windows = window_page[:windows_limit]
            has_more_windows = len(window_page) > windows_limit
            return {
                "agent_id": agent_id,
                "runs": [_history_run_payload(run) for run in runs],
                "windows": [_window_event_payload(event) for event in windows],
                "has_before": len(runs) == limit,
                "next_before": runs[-1].sequence if len(runs) == limit else None,
                "windows_has_more": has_more_windows,
                "windows_next_after": (
                    windows[-1].number if has_more_windows and windows else None
                ),
            }

        def agent_history_read_data(params: dict[str, Any]) -> Any:
            agent_id = _required_integer(params, "agent_id")
            self._human().authorize_agent_history(agent_id)
            sequence = _required_integer(params, "sequence")
            message_offset = _integer_param(
                params, "message_offset", default=0, maximum=2**63 - 1
            )
            message_limit = _integer_param(
                params, "message_limit", default=100, maximum=100
            )
            input_offset = _integer_param(
                params, "input_offset", default=0, maximum=2**63 - 1
            )
            input_limit = _integer_param(
                params, "input_limit", default=100, maximum=100
            )
            response_offset = _integer_param(
                params, "response_offset", default=0, maximum=2**63 - 1
            )
            response_limit = _integer_param(
                params, "response_limit", default=100, maximum=100
            )
            related_offset = _integer_param(
                params, "related_offset", default=0, maximum=2**63 - 1
            )
            related_limit = _integer_param(
                params, "related_limit", default=100, maximum=100
            )
            parameters_offset = _integer_param(
                params, "parameters_offset", default=0, maximum=2**63 - 1
            )
            settings_offset = _integer_param(
                params, "settings_offset", default=0, maximum=2**63 - 1
            )
            model_offset = _integer_param(
                params, "model_offset", default=0, maximum=2**63 - 1
            )
            field_limit = _integer_param(
                params, "field_limit", default=16 * 1024, maximum=16 * 1024
            )
            request_after = _integer_param(params, "request_after", maximum=2**63 - 1)
            request_limit = _integer_param(
                params, "request_limit", default=30, maximum=30
            )
            windows_after = _integer_param(params, "windows_after", maximum=2**63 - 1)
            windows_limit = _integer_param(
                params, "windows_limit", default=30, maximum=30
            )
            assert message_offset is not None
            assert message_limit is not None
            assert input_offset is not None
            assert input_limit is not None
            assert response_offset is not None
            assert response_limit is not None
            assert related_offset is not None
            assert related_limit is not None
            assert parameters_offset is not None
            assert settings_offset is not None
            assert model_offset is not None
            assert field_limit is not None
            assert request_limit is not None
            assert windows_limit is not None
            if 0 in (message_limit, input_limit, response_limit, related_limit):
                raise DomainError("invalid_params", "message limits must be positive")
            if 0 in (field_limit, request_limit, windows_limit):
                raise DomainError("invalid_params", "history limits must be positive")
            run = self._scheduler.history.history_run(agent_id, sequence)
            if run is None:
                raise DomainError("not_found", f"Turn {sequence} does not exist")
            request_page = self._scheduler.history.model_request_summaries(
                agent_id,
                sequence,
                after=request_after,
                limit=request_limit + 1,
            )
            requests = request_page[:request_limit]
            requests_has_more = len(request_page) > request_limit
            window_page = self._scheduler.history.window_events(
                agent_id, after=windows_after, limit=windows_limit + 1
            )
            windows = window_page[:windows_limit]
            windows_has_more = len(window_page) > windows_limit
            if requests:
                messages: dict[str, Any] = {
                    "messages": [],
                    "offset": 0,
                    "total": 0,
                    "next_offset": 0,
                    "has_more": False,
                }
            else:
                legacy = self._scheduler.history.read_run_slice(
                    agent_id, sequence, message_offset, 16 * 1024
                )
                if legacy is None:
                    raise DomainError("not_found", f"Turn {sequence} does not exist")
                next_offset = legacy.offset + len(legacy.messages)
                messages = {
                    "messages": [{"kind": "legacy", "content": legacy.messages}],
                    "offset": legacy.offset,
                    "total": legacy.total_length,
                    "next_offset": next_offset,
                    "has_more": next_offset < legacy.total_length,
                }
            result: dict[str, Any] = {
                "agent_id": agent_id,
                "run": _history_run_payload(run),
                "windows": [_window_event_payload(event) for event in windows],
                "windows_has_more": windows_has_more,
                "windows_next_after": (
                    windows[-1].number if windows_has_more and windows else None
                ),
                "requests": [_request_summary_payload(item) for item in requests],
                "requests_has_more": requests_has_more,
                "requests_next_after": (
                    requests[-1].ordinal if requests_has_more and requests else None
                ),
                "messages": messages,
                "missing": (
                    [
                        "provider_system_instructions",
                        "model_input",
                        "model_request_parameters",
                        "model_settings",
                        "model_identity",
                        "request_window",
                    ]
                    if not requests and run.status != "running"
                    else []
                ),
            }
            ordinal = _integer_param(params, "ordinal", maximum=2**63 - 1)
            if ordinal is None and requests:
                ordinal = requests[0].ordinal
            if ordinal is None:
                return result
            summary = self._scheduler.history.model_request_summary(
                agent_id, sequence, ordinal
            )
            if summary is None:
                raise DomainError(
                    "not_found", f"Model request {ordinal} does not exist"
                )
            input_page = self._scheduler.history.model_request_messages(
                agent_id, sequence, ordinal, "input", input_offset, input_limit
            )
            response_page = self._scheduler.history.model_request_messages(
                agent_id, sequence, ordinal, "response", response_offset, response_limit
            )
            related_page = self._scheduler.history.model_request_messages(
                agent_id, sequence, ordinal, "related", related_offset, related_limit
            )
            parameters_page = self._scheduler.history.model_request_field(
                agent_id,
                sequence,
                ordinal,
                "parameters",
                parameters_offset,
                field_limit,
            )
            settings_page = self._scheduler.history.model_request_field(
                agent_id, sequence, ordinal, "settings", settings_offset, field_limit
            )
            model_page = self._scheduler.history.model_request_field(
                agent_id, sequence, ordinal, "model", model_offset, field_limit
            )
            result["request"] = {
                "summary": _request_summary_payload(summary),
                "input": input_page,
                "parameters": parameters_page,
                "settings": settings_page,
                "model": model_page,
                "response": response_page if summary.response_count else None,
                "related": related_page,
            }
            if summary.status == "pending" and run.status != "running":
                result["missing"].append("model_response")
            return result

        def agent_history_read(params: dict[str, Any]) -> Any:
            with self._reading():
                result = agent_history_read_data(params)
            request = result.get("request")
            if request is not None:
                ordinal = request["summary"]["ordinal"]
                for source in ("input", "response", "related"):
                    if request[source] is not None:
                        request[source] = _render_message_page(
                            request[source], source, ordinal
                        )
                for source in ("parameters", "settings", "model"):
                    request[source] = _render_model_field_page(
                        request[source], source, ordinal
                    )
            return result

        def agent_history_text_data(params: dict[str, Any]) -> Callable[[], Any]:
            agent_id = _required_integer(params, "agent_id")
            self._human().authorize_agent_history(agent_id)
            sequence = _required_integer(params, "sequence")
            source = params.get("source")
            if source not in (
                "run",
                "input",
                "response",
                "related",
                "parameters",
                "settings",
                "model",
            ):
                raise DomainError("invalid_params", "source is invalid")
            message_index = _required_integer(params, "message_index")
            offset = _integer_param(params, "offset", default=0, maximum=2**63 - 1)
            limit = _integer_param(
                params, "limit", default=16 * 1024, maximum=16 * 1024
            )
            assert offset is not None
            assert limit is not None
            if limit == 0:
                raise DomainError("invalid_params", "limit must be positive")
            path = params.get("path")
            if not isinstance(path, list) or any(
                isinstance(item, bool) or not isinstance(item, (str, int))
                for item in path
            ):
                raise DomainError("invalid_params", "path must be a list")
            if source in ("parameters", "settings", "model"):
                ordinal = _integer_param(params, "ordinal")
                if ordinal is None:
                    raise DomainError("invalid_params", "ordinal is required")
                page = self._scheduler.history.model_request_field(
                    agent_id, sequence, ordinal, source, offset, limit
                )
                return partial(_render_model_field_page, page, source, ordinal)
            raw = (
                self._scheduler.history.run_messages(agent_id, sequence)
                if source == "run"
                else None
            )
            index = message_index
            if source in ("input", "response", "related"):
                ordinal = _integer_param(params, "ordinal")
                if ordinal is None:
                    raise DomainError("invalid_params", "ordinal is required")
                raw = self._scheduler.history.model_request_message(
                    agent_id, sequence, ordinal, source, message_index
                )
                index = 0
            if raw is None:
                raise DomainError("not_found", "History content does not exist")
            return partial(text_from_history, raw, index, path, offset, limit)

        def agent_history_text(params: dict[str, Any]) -> Any:
            with self._reading():
                render = agent_history_text_data(params)
            return render()

        def agent_history_image_data(params: dict[str, Any]) -> Any:
            agent_id = _required_integer(params, "agent_id")
            self._human().authorize_agent_history(agent_id)
            sequence = _required_integer(params, "sequence")
            source = params.get("source")
            if source not in ("run", "input", "response", "related"):
                raise DomainError("invalid_params", "source is invalid")
            message_index = _required_integer(params, "message_index")
            path = params.get("path")
            if not isinstance(path, list) or any(
                isinstance(item, bool) or not isinstance(item, (str, int))
                for item in path
            ):
                raise DomainError("invalid_params", "path must be a list")
            ordinal = _integer_param(params, "ordinal")
            raw = (
                self._scheduler.history.run_messages(agent_id, sequence)
                if source == "run"
                else None
            )
            if source in ("input", "response", "related"):
                if ordinal is None:
                    raise DomainError("invalid_params", "ordinal is required")
                raw = self._scheduler.history.model_request_message(
                    agent_id, sequence, ordinal, source, message_index
                )
                if raw is None:
                    raise DomainError("not_found", "History content does not exist")
            if raw is None:
                raise DomainError("not_found", "History content does not exist")
            return (
                raw,
                0 if source in ("input", "response", "related") else message_index,
                path,
            )

        def agent_history_image(params: dict[str, Any]) -> Any:
            with self._reading():
                data = agent_history_image_data(params)
            image = binary_from_history(*data)
            return {
                "media_type": image.media_type,
                "size": len(image.data),
                "identifier": image.identifier,
                "data": base64.b64encode(image.data).decode("ascii"),
            }

        def agent_detail(params: dict[str, Any]) -> Any:
            agent_id = int(params["agent_id"])
            status = self._scheduler.agent_status(agent_id)
            with self._reading():
                self._human().authorize_agent_history(agent_id)
                runs = self._scheduler.history.history_runs(agent_id, limit=30)
                effects = (
                    self._scheduler.history.effects(
                        agent_id, sequences=[run.sequence for run in runs]
                    )
                    if runs
                    else ()
                )
                window = self._scheduler.history.window(agent_id)
                usage = self._scheduler.history.usage_total(agent_id)
                pause_reason = self._scheduler.history.pause_reason(agent_id)
                no_tool_streak = self._scheduler.history.no_tool_streak(agent_id)
            produced: dict[int, list[dict[str, Any]]] = {}
            for effect in effects:
                produced.setdefault(effect.sequence, []).append(
                    {
                        "ordinal": effect.ordinal,
                        "tool": effect.tool,
                        "summary": effect.summary,
                    }
                )
            streak = idle_streak(
                [
                    (
                        run.status,
                        [item["tool"] for item in produced.get(run.sequence, [])],
                    )
                    for run in runs
                ]
            )
            statistics = self._scheduler.statistics(agent_id, idle_streak=streak)
            return {
                **status,
                "window": _window_event_payload(window),
                "workspace": self._human().list_workspace(agent_id=agent_id),
                "usage": usage,
                **statistics,
                "pause_reason": pause_reason,
                "no_tool_streak": no_tool_streak,
                "idle_streak": streak,
                "runs": [
                    {
                        "sequence": run.sequence,
                        "run_id": run.run_id,
                        "status": run.status,
                        "started_at": run.started_at,
                        "completed_at": run.completed_at,
                        "last_saved_at": run.last_saved_at,
                        "window_number": run.window_number,
                        "window_reset_at": run.window_reset_at,
                        "window_reason": run.window_reason,
                        "request_count": run.request_count,
                        "usage": run.usage_json,
                        "error": run.error,
                        "effects": produced.get(run.sequence, []),
                    }
                    for run in runs
                ],
            }

        def settings_get(params: dict[str, Any]) -> Any:
            section = str(params.get("section", "model"))
            values = settings.get_settings(section) or {}
            if section == "model":
                return ModelCatalog.restore(values).redacted()
            if section == "voice":
                if "mode" in values:
                    values = settings.update_settings("voice", {})
                return VoiceConfig.restore(values).public()
            if section == "agent":
                return asdict(agent_parameters(values))
            if section == "observability":
                return {
                    key: value
                    for key, value in values.items()
                    if key not in ("secret_key", "public_key")
                } | {"keys_set": bool(values.get("secret_key"))}
            if section == "execution":
                return self._scheduler.execution.status()
            return values

        def settings_update(params: dict[str, Any]) -> Any:
            section = str(params.get("section", "model"))
            values = params.get("values", {})
            if not isinstance(values, dict):
                raise DomainError(
                    "invalid_setting", "Settings values must be an object"
                )
            if section == "execution":
                result = self._scheduler.execution.configure(
                    values, lambda stored: settings.set_settings("execution", stored)
                )
                self._dispatcher.emit("settings.updated", {"section": section})
                return result
            updated = settings.update_settings(section, values)
            self._dispatcher.emit("settings.updated", {"section": section})
            if section == "model":
                return ModelCatalog.restore(updated).redacted()
            return settings_get({"section": section})

        def settings_list_models(params: dict[str, Any]) -> Any:
            return self._list_models(dict(params), settings.get_settings("model"))

        def settings_test_model(params: dict[str, Any]) -> Any:
            return self._test_model(dict(params), settings.get_settings("model"))

        register("info.get", info_get, scope=independent)
        register(
            "organization.get",
            organization_get,
            scope=lambda params: RequestScope(
                reads=(
                    member_set,
                    model_settings,
                    Resource("settings", "agent"),
                    Resource("settings", "execution"),
                )
            ),
        )
        register(
            "organization.create_agent",
            create_agent,
            scope=lambda params: RequestScope(writes=(member_set, model_settings)),
        )
        register(
            "organization.rename_member",
            rename_member,
            scope=lambda params: RequestScope(
                reads=(member_set,),
                writes=(Resource("member", int(params["member_id"])),),
            ),
        )
        register("organization.pause_agent", pause_agent, scope=agent_change)
        register("organization.resume_agent", resume_agent, scope=agent_change)
        register(
            "organization.delete_agent",
            delete_agent,
            scope=lambda params: RequestScope(
                writes=(
                    Resource("member", int(params["agent_id"])),
                    discussion_set,
                    model_settings,
                )
            ),
        )
        register(
            "discussion.create",
            discussion_create,
            scope=lambda params: RequestScope(
                reads=(member_set,), writes=(discussion_set,)
            ),
        )
        register(
            "discussion.list",
            discussion_list,
            scope=lambda params: RequestScope(reads=(discussion_set, member_set)),
        )
        register("discussion.read", discussion_read, scope=discussion_change)
        register("discussion.page", discussion_page, scope=discussion_query)
        register("discussion.mark_read", discussion_mark_read, scope=discussion_change)
        register(
            "discussion.ack_pending", discussion_ack_pending, scope=discussion_change
        )
        register("discussion.send", discussion_send, scope=discussion_send_scope)
        register("discussion.ack", discussion_ack, scope=discussion_change)
        register(
            "discussion.revoke_ack", discussion_revoke_ack, scope=discussion_change
        )
        register("discussion.set_members", discussion_members, scope=discussion_change)
        register(
            "discussion.add_members", discussion_add_members, scope=discussion_change
        )
        register(
            "discussion.remove_members",
            discussion_remove_members,
            scope=discussion_change,
        )
        register("discussion.archive", discussion_archive, scope=discussion_change)
        register("discussion.unarchive", discussion_unarchive, scope=discussion_change)
        register(
            "discussion.search",
            discussion_search,
            scope=lambda params: RequestScope(
                reads=(
                    Resource(
                        "discussion",
                        int(params["discussion_id"])
                        if params.get("discussion_id") is not None
                        else None,
                    ),
                    member_set,
                )
            ),
        )
        register(
            "library.list",
            library_list,
            scope=lambda params: library_scope(params, directory=True),
        )
        register("library.read", library_read, scope=library_scope)
        register(
            "library.write",
            library_write,
            scope=lambda params: library_scope(params, writing=True),
        )
        register(
            "library.edit",
            library_edit,
            scope=lambda params: library_scope(params, writing=True),
        )
        register(
            "library.mkdir",
            library_mkdir,
            scope=lambda params: library_scope(params, writing=True, directory=True),
        )
        register("workspace.list", workspace_list, scope=workspace_scope)
        register("workspace.read", workspace_read, scope=workspace_scope)
        register(
            "library.delete",
            library_delete,
            scope=lambda params: library_scope(params, writing=True, directory=True),
        )
        register(
            "library.move",
            library_move,
            scope=lambda params: library_scope(
                params, writing=True, directory=True, moving=True
            ),
        )
        register("agent.detail", agent_detail, scope=detail_scope)
        register("agent.history", agent_history, scope=agent_query)
        register("agent.history.read", agent_history_read, scope=agent_query)
        register("agent.history.text", agent_history_text, scope=agent_query)
        register("agent.history.image", agent_history_image, scope=agent_query)
        register("settings.get", settings_get, scope=setting_query)
        register("settings.update", settings_update, scope=setting_change)
        register(
            "settings.list_models",
            settings_list_models,
            scope=lambda params: RequestScope(reads=(model_settings,)),
        )
        register(
            "settings.test_model",
            settings_test_model,
            scope=lambda params: RequestScope(reads=(model_settings,)),
        )
