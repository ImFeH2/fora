from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from pydantic import ValidationError

from huddol.adapters.jsonl.protocol import Dispatcher
from huddol.adapters.model.config import AgentModelConfig, ModelCatalog
from huddol.core.errors import DomainError
from huddol.core.parameters import agent_parameters, validate_parameters
from huddol.core.turn import idle_streak
from huddol.runtime.scheduler import Scheduler
from huddol.tools import AgentTools
from huddol.tools.authorize import Actor

HUMAN_ID = 1

ModelProbe = Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]]


def _list_models(values: dict[str, Any], stored: dict[str, Any] | None) -> Any:
    from huddol.adapters.model.probe import list_models

    return list_models(values, stored)


def _test_model(values: dict[str, Any], stored: dict[str, Any] | None) -> Any:
    from huddol.adapters.model.probe import try_model

    return try_model(values, stored)


class Api:
    def __init__(
        self,
        scheduler: Scheduler,
        dispatcher: Dispatcher,
        *,
        list_models: ModelProbe = _list_models,
        test_model: ModelProbe = _test_model,
    ) -> None:
        self._scheduler = scheduler
        self._dispatcher = dispatcher
        self._list_models = list_models
        self._test_model = test_model
        self._register()

    def _human(self) -> AgentTools:
        return self._scheduler.tools_for_actor(Actor(HUMAN_ID, False))

    def _changed(self, event: str, payload: dict[str, Any]) -> None:
        self._dispatcher.emit(event, payload)
        self._scheduler.wake()

    def _register(self) -> None:
        register = self._dispatcher.register
        settings = self._scheduler.settings

        def organization_get(params: dict[str, Any]) -> Any:
            del params
            members = self._human().list_members()
            for member in members:
                if member["type"] != "agent":
                    continue
                usage = self._scheduler.history.usage_total(int(member["id"]))
                member["tokens"] = usage["total_tokens"]
            return {
                "id": 1,
                "uuid": self._human().organization_uuid(),
                "members": members,
                "human_id": HUMAN_ID,
                **self._scheduler.statistics(),
            }

        def create_agent(params: dict[str, Any]) -> Any:
            result: dict[str, Any] = {}

            def create(stored: dict[str, object] | None) -> dict[str, object]:
                nonlocal result
                catalog = ModelCatalog.restore(stored)
                try:
                    selection = AgentModelConfig.model_validate(
                        params.get("model_config", {})
                    )
                except ValidationError:
                    raise DomainError(
                        "invalid_model_config", "Invalid Agent model configuration"
                    ) from None
                catalog.validate_selection(selection, "New Agent")
                result = self._human().create_agent(str(params.get("name", "")))
                catalog.agent_configs[str(result["id"])] = selection
                return catalog.model_dump()

            settings.update_settings("model", create)
            self._changed("member.created", result)
            return result

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
            message_id = params.get("message_id")
            return self._human().read_discussion(
                int(params["discussion_id"]),
                int(message_id) if message_id is not None else None,
                params.get("limit"),
                before=params.get("before"),
                after=params.get("after"),
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

        register("upload.create", upload_create)
        register(
            "upload.status",
            lambda params: self._human().upload_status(params["upload_ids"]),
        )
        register(
            "upload.cancel",
            lambda params: self._human().cancel_uploads(params["upload_ids"]),
        )
        register(
            "discussion.send_status",
            lambda params: self._human().send_status(
                params["discussion_id"], params["client_message_id"]
            ),
        )

        register(
            "discussion.cancel_send",
            lambda params: self._human().cancel_send(
                params["discussion_id"], params["client_message_id"]
            ),
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

        def agent_detail(params: dict[str, Any]) -> Any:
            agent_id = int(params["agent_id"])
            runs = self._scheduler.history.run_summaries(agent_id, limit=30)
            effects = self._scheduler.history.effects(
                agent_id, sequences=[run.sequence for run in runs]
            )
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
                **self._scheduler.agent_status(agent_id),
                "window": asdict(self._scheduler.history.window(agent_id)),
                "workspace": self._human().list_workspace(agent_id=agent_id),
                "usage": self._scheduler.history.usage_total(agent_id),
                **statistics,
                "pause_reason": self._scheduler.history.pause_reason(agent_id),
                "no_tool_streak": self._scheduler.history.no_tool_streak(agent_id),
                "idle_streak": streak,
                "runs": [
                    {
                        "sequence": run.sequence,
                        "status": run.status,
                        "started_at": run.started_at,
                        "completed_at": run.completed_at,
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
            if section == "model":

                def update_model(stored: dict[str, object] | None) -> dict[str, object]:
                    agent_ids = {
                        int(member["id"])
                        for member in self._human().list_members()
                        if member["type"] == "agent"
                    }
                    return (
                        ModelCatalog.restore(stored)
                        .apply(values, agent_ids)
                        .model_dump()
                    )

                updated = settings.update_settings("model", update_model)
                self._dispatcher.emit("settings.updated", {"section": section})
                return ModelCatalog.restore(updated).redacted()
            if section == "agent":
                values = validate_parameters(values)
            if section == "execution":
                result = self._scheduler.execution.configure(
                    values, lambda stored: settings.set_settings("execution", stored)
                )
                self._dispatcher.emit("settings.updated", {"section": section})
                return result
            merged = {**(settings.get_settings(section) or {}), **values}
            if section == "agent":
                validate_parameters(merged)
            settings.set_settings(section, merged)
            self._dispatcher.emit("settings.updated", {"section": section})
            return settings_get({"section": section})

        def settings_list_models(params: dict[str, Any]) -> Any:
            return self._list_models(dict(params), settings.get_settings("model"))

        def settings_test_model(params: dict[str, Any]) -> Any:
            return self._test_model(dict(params), settings.get_settings("model"))

        register("organization.get", organization_get)
        register("organization.create_agent", create_agent)
        register("organization.rename_member", rename_member)
        register("organization.pause_agent", pause_agent)
        register("organization.resume_agent", resume_agent)
        register("organization.delete_agent", delete_agent)
        register("discussion.create", discussion_create)
        register("discussion.list", discussion_list)
        register("discussion.read", discussion_read)
        register("discussion.page", discussion_page)
        register("discussion.mark_read", discussion_mark_read)
        register("discussion.ack_pending", discussion_ack_pending)
        register("discussion.send", discussion_send)
        register("discussion.ack", discussion_ack)
        register("discussion.revoke_ack", discussion_revoke_ack)
        register("discussion.set_members", discussion_members)
        register("discussion.add_members", discussion_add_members)
        register("discussion.remove_members", discussion_remove_members)
        register("discussion.archive", discussion_archive)
        register("discussion.unarchive", discussion_unarchive)
        register("discussion.search", discussion_search)
        register("library.list", library_list)
        register("library.read", library_read)
        register("library.write", library_write)
        register("library.edit", library_edit)
        register("library.mkdir", library_mkdir)
        register("workspace.list", workspace_list)
        register("workspace.read", workspace_read)
        register("library.delete", library_delete)
        register("library.move", library_move)
        register("agent.detail", agent_detail)
        register("settings.get", settings_get)
        register("settings.update", settings_update)
        register("settings.list_models", settings_list_models)
        register("settings.test_model", settings_test_model)
