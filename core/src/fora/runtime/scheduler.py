from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from fora.core.errors import DomainError
from fora.core.parameters import AgentParameters, agent_parameters
from fora.ports.agent import (
    AgentLifecycle,
    AgentRun,
    HistoryStore,
    ModelRequestHandle,
    SettingsStore,
)
from fora.ports.execution import ExecutionControl
from fora.ports.store import OrganizationStore
from fora.runtime.reminder import (
    PREPARATION_PROMPT,
    HistoryValidationError,
    ModelRunner,
    Reminder,
    TurnRequest,
    build_reminder,
    exchange_nudge,
    read_agents_instructions,
    render_resident,
    reset_notice,
)
from fora.services.workspace import Workspace
from fora.tools import AgentTools, Dependencies, TurnBinding
from fora.tools.authorize import Actor, Authorizer

logger = logging.getLogger("fora.runtime")


@dataclass
class TurnRecord:
    agent_id: int
    sequence: int
    status: str
    error: str | None = None


@dataclass
class PreparedTurn:
    run: AgentRun
    name: str
    reminder: Reminder | None
    prompt: str
    history: str


@dataclass
class FailedFinalization:
    sequence: int
    saved_history: str
    messages: str
    usage: str | None
    error: str


class _TurnRequestRecorder:
    def __init__(
        self,
        history: HistoryStore,
        run: AgentRun,
        window_number: int,
        progress: Callable[[str], None],
    ) -> None:
        self._history = history
        self._run = run
        self._window_number = window_number
        self._progress = progress

    def start(
        self,
        messages_json: Sequence[str],
        parameters_json: str,
        settings_json: str,
        model_json: str,
        streaming: bool,
    ) -> ModelRequestHandle:
        handle = self._history.start_model_request(
            self._run.agent_id,
            self._run.sequence,
            self._run.run_id,
            self._window_number,
            messages_json,
            parameters_json,
            settings_json,
            model_json,
            streaming,
        )
        self._progress("request.started")
        return handle

    def response(
        self, handle: ModelRequestHandle, response_json: Sequence[str]
    ) -> None:
        self._history.finish_model_request(
            self._run.agent_id, self._run.sequence, handle, response_json
        )
        self._progress("request.response")

    def related(self, handle: ModelRequestHandle, messages_json: Sequence[str]) -> None:
        self._history.link_model_request(
            self._run.agent_id, self._run.sequence, handle, messages_json
        )
        self._progress("tool.result")

    def error(self, handle: ModelRequestHandle, error: str) -> None:
        self._history.fail_model_request(
            self._run.agent_id, self._run.sequence, handle, error
        )
        self._progress("request.error")

    def success(self, handle: ModelRequestHandle) -> None:
        self._history.complete_model_request(
            self._run.agent_id, self._run.sequence, handle
        )
        self._progress("request.completed")


class Scheduler:
    def __init__(
        self,
        deps: Dependencies,
        runner: ModelRunner,
        *,
        authorizer: Authorizer | None = None,
        on_event: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        self._deps = deps
        self._runner = runner
        self._authorizer = authorizer or Authorizer()
        self._threads: dict[int, threading.Thread] = {}
        self._lock = threading.RLock()
        self._member_locks: dict[int, Any] = {}
        self._active: dict[int, AgentRun] = {}
        self._reserved: set[int] = set()
        self._failures: dict[int, FailedFinalization] = {}
        self._blocked: dict[int, str] = {}
        self._pause_requested: set[int] = set()
        self._scheduler_stopped: str | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._loop: threading.Thread | None = None
        self._on_event = on_event or (lambda name, payload: None)

    @property
    def store(self) -> OrganizationStore:
        return self._deps.store

    @property
    def history(self) -> HistoryStore:
        return self._deps.history

    @property
    def settings(self) -> SettingsStore:
        return self._deps.settings

    def emit(self, name: str, payload: dict[str, object]) -> None:
        try:
            self._on_event(name, payload)
        except Exception:
            logger.exception("event listener failed for %s", name)

    def _changed(self, name: str, payload: dict[str, object]) -> None:
        self.emit(name, payload)
        self.wake()

    def wake(self) -> None:
        self._wake.set()

    def recover(self) -> None:
        with self.history.transaction():
            for member in self.store.list_members():
                if not member.is_agent:
                    continue
                lifecycle = self.history.lifecycle(member.id)
                runs = self.history.runs(member.id, limit=1)
                try:
                    self._runner.validate_history(
                        self.history.latest_messages(member.id)
                    )
                    if runs and runs[0].status == "running":
                        run = runs[0]
                        self._runner.validate_history(run.messages_json)
                        self.history.save_progress(
                            member.id, run.sequence, run.messages_json
                        )
                        self.history.finish_run(
                            member.id,
                            run.sequence,
                            status="interrupted",
                            messages_json=run.messages_json,
                            usage_json=run.usage_json,
                        )
                except HistoryValidationError:
                    self.history.pause_for_safety(member.id, "history_invalid")
                if member.state == "running":
                    self.store.set_agent_state(
                        member.id,
                        "paused"
                        if lifecycle.pause_requested
                        else "error"
                        if lifecycle.error
                        else "idle",
                    )
            self.history.mark_session_start()

    def start(self, poll_seconds: float = 0.2) -> None:
        if self._loop is not None:
            return
        self.parameters()
        self._loop = threading.Thread(
            target=self.serve,
            args=(poll_seconds,),
            name="fora-loop",
            daemon=True,
        )
        self._loop.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        loop = self._loop
        self._loop = None
        if loop is not None and loop.ident is not None:
            loop.join(timeout=5)
        with self._lock:
            running = list(self._threads.values())
        self._deps.execution.close()
        for thread in running:
            if thread.ident is None:
                continue
            thread.join(timeout=5)

    def member_lock(self, agent_id: int) -> Any:
        with self._lock:
            return self._member_locks.setdefault(agent_id, threading.RLock())

    def _require_agent(self, agent_id: int) -> None:
        member = self.store.get_member(agent_id)
        if member is None or not member.is_agent or member.deleted:
            raise DomainError("not_found", f"Agent {agent_id} does not exist")

    def _obstacles(self, agent_id: int) -> list[dict[str, str]]:
        found = []
        reason = self._blocked.get(agent_id) or self.history.pause_reason(agent_id)
        if reason is not None:
            descriptions = {
                "no_tool_calls": "Consecutive Turns completed without tool calls",
                "runtime_error": "A previous runtime failure requires Human recovery",
                "history_invalid": "Stored model history is invalid",
                "storage_unavailable": "Turn state could not be saved",
                "repeated_mentions": "Three completed Turns left the same mentions pending",
            }
            found.append(
                {
                    "code": reason,
                    "message": descriptions[reason],
                    "recovery": (
                        "A Human must review the Turns and Resume to start a new three-Turn cycle"
                        if reason == "repeated_mentions"
                        else "A Human must repair the condition and Resume"
                    ),
                }
            )
        if self._scheduler_stopped is not None:
            found.append(
                {
                    "code": "scheduler_stopped",
                    "message": self._scheduler_stopped,
                    "recovery": "Correct the configuration and restart the application",
                }
            )
        try:
            if self.over_token_limit(agent_id):
                found.append(
                    {
                        "code": "token_limit",
                        "message": "Token limit reached",
                        "recovery": "Increase the token limit in Settings",
                    }
                )
        except DomainError as error:
            found.append(
                {
                    "code": "invalid_setting",
                    "message": str(error),
                    "recovery": "Correct Agent Settings and restart the application",
                }
            )
        try:
            self._runner.check_available(agent_id)
        except DomainError as error:
            found.append(
                {
                    "code": error.code,
                    "message": str(error),
                    "recovery": "Correct Model Settings; if unavailable, stop the application, back up and repair the model section, then restart",
                }
            )
        execution_error = self.execution.status()["error"]
        if execution_error is not None:
            found.append(
                {
                    "code": "execution_unavailable",
                    "message": str(execution_error),
                    "recovery": "Correct execution settings",
                }
            )
        return found

    def agent_status(self, agent_id: int) -> dict[str, Any]:
        with self.member_lock(agent_id):
            lifecycle = self.history.lifecycle(agent_id)
            requested = lifecycle.pause_requested or agent_id in self._pause_requested
            obstacles = self._obstacles(agent_id)
            active = agent_id in self._active or agent_id in self._reserved
            if active:
                state = "running"
            elif obstacles:
                state = "blocked"
            elif requested:
                state = "paused"
            elif lifecycle.error is not None:
                state = "error"
            else:
                state = "idle"
            return {
                "id": agent_id,
                "state": state,
                "pause_requested": requested,
                "reasons": obstacles,
                "error": lifecycle.error,
                "scheduler_stopped": self._scheduler_stopped,
            }

    def pause(self, agent_id: int) -> dict[str, Any]:
        failure = None
        with self.member_lock(agent_id):
            self._require_agent(agent_id)
            self._pause_requested.add(agent_id)
            active = agent_id in self._active
            try:
                with self.history.transaction():
                    self.history.set_lifecycle(
                        agent_id,
                        replace(self.history.lifecycle(agent_id), pause_requested=True),
                    )
                    if not active:
                        self.store.set_agent_state(agent_id, "paused")
            except Exception as error:
                logger.exception("Saving pause intent failed for agent %s", agent_id)
                self._blocked[agent_id] = "storage_unavailable"
                failure = error
        self._changed("organization.changed", {"id": agent_id})
        if failure is not None:
            raise DomainError(
                "pause_save_failed",
                "The current Turn will finish; saving the pause intent failed",
            ) from failure
        return self.agent_status(agent_id)

    def resume(self, agent_id: int, is_agent: bool = False) -> dict[str, Any]:
        with self.member_lock(agent_id):
            self._require_agent(agent_id)
            if is_agent and (
                self.history.pause_reason(agent_id) is not None
                or agent_id in self._blocked
            ):
                raise DomainError(
                    "not_permitted", "Only a Human can recover this Agent"
                )
            active = agent_id in self._active
            with self.history.transaction():
                lifecycle = self.history.lifecycle(agent_id)
                if active:
                    self.history.set_lifecycle(
                        agent_id, replace(lifecycle, pause_requested=False)
                    )
                else:
                    current = self.history.latest_messages(agent_id)
                    self._runner.validate_history(current)
                    failure = self._failures.get(agent_id)
                    if failure is not None:
                        runs = self.history.runs(agent_id, limit=1)
                        if not runs or runs[0].sequence != failure.sequence:
                            raise DomainError(
                                "history_changed", "The failed Turn identity changed"
                            )
                        stored = runs[0].messages_json
                        messages = (
                            failure.messages
                            if stored == failure.saved_history
                            else stored
                        )
                        self._runner.validate_history(messages)
                        self.history.save_progress(agent_id, failure.sequence, messages)
                        self.history.finish_run(
                            agent_id,
                            failure.sequence,
                            status="interrupted",
                            messages_json=messages,
                            usage_json=failure.usage,
                            error=failure.error,
                        )
                    else:
                        runs = self.history.runs(agent_id, limit=1)
                        if runs and runs[0].status == "running":
                            run = runs[0]
                            self._runner.validate_history(run.messages_json)
                            self.history.save_progress(
                                agent_id, run.sequence, run.messages_json
                            )
                            self.history.finish_run(
                                agent_id,
                                run.sequence,
                                status="interrupted",
                                messages_json=run.messages_json,
                                usage_json=run.usage_json,
                                error=run.error,
                            )
                    self.history.reset_safety(agent_id)
                    self.history.set_lifecycle(agent_id, AgentLifecycle())
                    self.store.set_agent_state(agent_id, "idle")
            if not active:
                self._failures.pop(agent_id, None)
            self._blocked.pop(agent_id, None)
            self._pause_requested.discard(agent_id)
        self._changed("organization.changed", {"id": agent_id})
        return self.agent_status(agent_id)

    def statistics(
        self, agent_id: int | None = None, *, idle_streak: int = 0
    ) -> dict[str, Any]:
        try:
            parameters = self.parameters()
        except DomainError as error:
            return {
                "token_limit": None,
                "over_token_limit": None,
                "idle": None,
                "statistics_unavailable": str(error),
            }
        result: dict[str, Any] = {
            "token_limit": parameters.token_limit,
            "statistics_unavailable": None,
        }
        if agent_id is not None:
            usage = self.history.usage_total(agent_id)
            result["over_token_limit"] = (
                parameters.token_limit > 0
                and usage["total_tokens"] >= parameters.token_limit
            )
            result["idle"] = idle_streak >= parameters.idle_streak_after
        return result

    def token_limit(self) -> int:
        return self.parameters().token_limit

    def over_token_limit(self, agent_id: int) -> bool:
        limit = self.token_limit()
        if limit <= 0:
            return False
        return self.history.usage_total(agent_id)["total_tokens"] >= limit

    def parameters(self) -> AgentParameters:
        return agent_parameters(self.settings.get_settings("agent"))

    def preparation_due(self, agent_id: int) -> bool:
        runs = self.history.run_summaries(agent_id, limit=1)
        if not runs:
            return False
        run = runs[0]
        if run.sequence < self.history.window(
            agent_id
        ).since_sequence or run.status not in ("completed", "failed"):
            return False
        tokens = json.loads(run.usage_json or "{}").get("last_input_tokens")
        return type(tokens) is int and tokens >= self.parameters().context_window_tokens

    def pending_keys(self, agent_id: int) -> frozenset[tuple[int, int]]:
        return frozenset(
            (item.discussion_id, item.message_id)
            for item in self.store.pending(agent_id)
        )

    def _eligible(self, agent_id: int) -> bool:
        state = self.agent_status(agent_id)["state"]
        if state not in ("idle", "error"):
            return False
        return self._pending_eligible(
            agent_id, self.pending_keys(agent_id), self.history.lifecycle(agent_id)
        )

    def _pending_eligible(
        self,
        agent_id: int,
        keys: frozenset[tuple[int, int]],
        lifecycle: AgentLifecycle,
    ) -> bool:
        if lifecycle.error is not None:
            return bool(self.history.new_mentions(agent_id, tuple(keys)))
        if lifecycle.prepared_sequence is not None:
            return bool(
                keys
                & self.history.prepared_mentions(agent_id, lifecycle.prepared_sequence)
            ) or bool(self.history.new_mentions(agent_id, tuple(keys)))
        return self.preparation_due(agent_id) or bool(keys)

    def runnable_agents(self) -> tuple[int, ...]:
        return tuple(
            member.id
            for member in self.store.list_members()
            if member.is_agent and self._eligible(member.id)
        )

    def tools_for(self, agent_id: int) -> AgentTools:
        member = self.store.get_member(agent_id)
        if member is None:
            raise DomainError("not_found", f"Member {agent_id} does not exist")
        return self.tools_for_actor(Actor(agent_id, member.is_agent))

    def tools_for_actor(
        self,
        actor: Actor,
        turn: TurnBinding | None = None,
    ) -> AgentTools:
        return AgentTools(
            self._deps,
            actor,
            self._authorizer,
            turn,
            on_change=self._changed,
            agent_status=self.agent_status,
            pause=self.pause,
            resume=self.resume,
            member_guard=self.member_lock,
        )

    @property
    def execution(self) -> ExecutionControl:
        return self._deps.execution

    def resident_block(self, agent_id: int) -> str:
        return render_resident(
            Workspace(self._deps.workspace_tree_for(agent_id)).index(
                self.parameters().memory_index_bytes
            ),
            self.environment_facts(agent_id),
            reset_notice(self.history.window(agent_id)),
        )

    def environment_facts(self, agent_id: int) -> str | None:
        try:
            return self._deps.execution.snapshot().describe_environment(
                (
                    (
                        str(self._deps.workspace_tree_for(agent_id).root),
                        "your workspace, private",
                    ),
                    (
                        str(self._deps.library_tree.root),
                        "Library, shared with the whole organization",
                    ),
                )
            )
        except DomainError:
            return None

    def _prepare(self, agent_id: int) -> PreparedTurn | None:
        with self.member_lock(agent_id):
            member = self.store.get_member(agent_id)
            if (
                member is None
                or not member.is_agent
                or member.deleted
                or not self._eligible(agent_id)
            ):
                return None
            maximum = self.parameters().max_concurrent_turns
            with self._lock:
                if agent_id in self._reserved or len(self._reserved) >= maximum:
                    return None
                self._reserved.add(agent_id)
            try:
                with self.history.transaction():
                    lifecycle = self.history.lifecycle(agent_id)
                    pending = self.pending_keys(agent_id)
                    if not self._pending_eligible(agent_id, pending, lifecycle):
                        return None
                    prepared_from_error = (
                        lifecycle.error is not None and self.preparation_due(agent_id)
                    )
                    reminder = (
                        None
                        if self.preparation_due(agent_id)
                        else build_reminder(
                            self.store, self.history, agent_id, member.name
                        )
                    )
                    if reminder is None and not self.preparation_due(agent_id):
                        return None
                    previous = self.history.latest_messages(agent_id)
                    self._runner.validate_history(previous)
                    keys = (
                        [
                            (item.discussion_id, item.message_id)
                            for item in reminder.items
                        ]
                        if reminder is not None
                        else []
                    )
                    run = self.history.start_run(agent_id, reminded=keys)
                    if reminder is None:
                        self.history.consume_preparation(
                            agent_id, run.sequence, tuple(pending)
                        )
                    self.store.set_agent_state(agent_id, "running")
                    self.history.set_lifecycle(
                        agent_id,
                        replace(
                            lifecycle,
                            prepared_sequence=run.sequence
                            if prepared_from_error
                            else None,
                        ),
                    )
                self._active[agent_id] = run
                return PreparedTurn(
                    run,
                    member.name,
                    reminder,
                    reminder.render() if reminder is not None else PREPARATION_PROMPT,
                    previous,
                )
            except HistoryValidationError:
                self.history.pause_for_safety(agent_id, "history_invalid")
                return None
            except Exception:
                self._blocked[agent_id] = "storage_unavailable"
                raise
            finally:
                if agent_id not in self._active:
                    with self._lock:
                        self._reserved.discard(agent_id)

    def run_turn(self, agent_id: int) -> TurnRecord | None:
        prepared = self._prepare(agent_id)
        return self._execute(prepared) if prepared is not None else None

    def _execute(
        self, prepared: PreparedTurn, startup_error: Exception | None = None
    ) -> TurnRecord:
        run = prepared.run
        agent_id = run.agent_id
        agent_name = prepared.name
        reminder = prepared.reminder
        prompt = prepared.prompt
        items = reminder.items if reminder is not None else ()
        self.emit(
            "turn.started",
            {
                "agent_id": agent_id,
                "sequence": run.sequence,
                "items": len(items),
                "kind": "reminder" if reminder is not None else "preparation",
            },
        )
        discussion_ids = tuple(item.discussion_id for item in items)
        persisted_history = prepared.history

        def progress(phase: str) -> None:
            saved = self.history.history_run(agent_id, run.sequence)
            self.emit(
                "turn.progress",
                {
                    "agent_id": agent_id,
                    "sequence": run.sequence,
                    "phase": phase,
                    "saved_at": saved.last_saved_at if saved is not None else None,
                },
            )

        request_recorder = _TurnRequestRecorder(
            self.history, run, self.history.window(agent_id).number, progress
        )

        def persist(messages_json: str) -> None:
            nonlocal persisted_history
            self.history.save_progress(agent_id, run.sequence, messages_json)
            persisted_history = messages_json
            progress("history.saved")

        request = TurnRequest(
            agent_id=agent_id,
            sequence=run.sequence,
            agent_name=agent_name,
            prompt=prompt,
            reminder=reminder,
            history_json=persisted_history,
            resident="",
            environment=lambda: self.environment_facts(agent_id),
            persist=persist,
            request_recorder=request_recorder,
            ephemeral=lambda: exchange_nudge(
                self.store,
                agent_id,
                discussion_ids,
                after=self.parameters().exchange_nudge_after,
            ),
        )
        status = "completed"
        error: str | None = None
        usage_json: str | None = None
        context_exceeded = False
        overflow_context = None
        messages = persisted_history
        run_failure: Exception | None = None
        try:
            if startup_error is not None:
                raise startup_error
            request = replace(
                request,
                resident=self.resident_block(agent_id),
                agents_instructions=(
                    read_agents_instructions(
                        self._deps.library_tree, self._deps.workspace_tree_for(agent_id)
                    )
                    if not json.loads(request.history_json)
                    else None
                ),
            )
            outcome = self._runner.run(
                request,
                self.tools_for_actor(
                    Actor(agent_id, True), TurnBinding(agent_id, run.sequence)
                ),
            )
            usage_json = outcome.usage_json
            context_exceeded = outcome.context_exceeded
            overflow_context = outcome.overflow_context
            messages = outcome.messages_json
            if outcome.error:
                status = "failed"
                error = outcome.error
        except Exception as failure:
            run_failure = failure
            status = "failed"
            error = f"{type(failure).__name__}: {failure}"
            messages = persisted_history
            context_exceeded = False
            if isinstance(failure, HistoryValidationError):
                with self.member_lock(agent_id):
                    self._blocked[agent_id] = "history_invalid"
            logger.exception("turn failed for agent %s", agent_id)
        window = None
        threshold = None
        parameter_error = None
        if status == "completed":
            try:
                threshold = self.parameters().no_tool_turns_before_pause
            except DomainError as failure:
                parameter_error = failure
                context_exceeded = False
                logger.exception(
                    "Turn finalization configuration failed for agent %s", agent_id
                )
        try:
            with self.member_lock(agent_id):
                try:
                    if parameter_error is not None:
                        self.history.finish_run(
                            agent_id,
                            run.sequence,
                            status=status,
                            messages_json=messages,
                            usage_json=usage_json,
                            error=error,
                        )
                        persisted_history = messages
                        raise parameter_error
                    with self.history.transaction():
                        self.history.finish_run(
                            agent_id,
                            run.sequence,
                            status=status,
                            messages_json=messages,
                            usage_json=usage_json,
                            error=error,
                        )
                        keys = frozenset(
                            (item.discussion_id, item.message_id) for item in items
                        )
                        if (
                            status == "completed"
                            and keys
                            and self.pending_keys(agent_id) == keys
                            and self.history.pending_revision(agent_id)
                            == run.pending_revision
                            and self.history.repeated_turns(agent_id, tuple(keys)) >= 3
                        ):
                            self.history.pause_for_safety(agent_id, "repeated_mentions")
                        elif (
                            status == "completed"
                            and threshold is not None
                            and self.history.no_tool_streak(agent_id) >= threshold
                        ):
                            self.history.pause_for_safety(agent_id, "no_tool_calls")
                        reason = None
                        if status == "failed" and (
                            (
                                context_exceeded
                                and run.sequence
                                != self.history.window(agent_id).since_sequence
                            )
                            or (
                                run_failure is None
                                and reminder is None
                                and self.history.preparation_failure_streak(
                                    agent_id, run.sequence
                                )
                                >= 3
                            )
                        ):
                            reason = "overflow"
                        elif status == "completed" and reminder is None:
                            reason = "prepared"
                        if reason is not None:
                            window = self.history.reset_window(
                                agent_id,
                                reason,
                                overflow_context if reason == "overflow" else None,
                            )
                        lifecycle = self.history.lifecycle(agent_id)
                        requested = (
                            lifecycle.pause_requested
                            or agent_id in self._pause_requested
                        )
                        lifecycle_error = (
                            None
                            if window is not None and window.reason == "overflow"
                            else error
                        )
                        self.history.set_lifecycle(
                            agent_id,
                            AgentLifecycle(
                                pause_requested=requested,
                                error=lifecycle_error,
                                prepared_sequence=lifecycle.prepared_sequence
                                if reminder is None and status == "completed"
                                else None,
                            ),
                        )
                        self.store.set_agent_state(
                            agent_id,
                            "blocked"
                            if self.history.pause_reason(agent_id) is not None
                            else "paused"
                            if requested
                            else "error"
                            if lifecycle_error
                            else "idle",
                        )
                    persisted_history = messages
                except Exception as failure:
                    window = None
                    logger.exception("Turn finalization failed for agent %s", agent_id)
                    status = "failed"
                    error = f"{type(failure).__name__}: {failure}"
                    with self.history.transaction():
                        self.history.finish_run(
                            agent_id,
                            run.sequence,
                            status="failed",
                            messages_json=persisted_history,
                            usage_json=usage_json,
                            error=error,
                        )
                        lifecycle = self.history.lifecycle(agent_id)
                        requested = (
                            lifecycle.pause_requested
                            or agent_id in self._pause_requested
                        )
                        self.history.set_lifecycle(
                            agent_id,
                            AgentLifecycle(pause_requested=requested, error=error),
                        )
                        self.store.set_agent_state(
                            agent_id, "paused" if requested else "error"
                        )
                if self._blocked.get(agent_id) == "storage_unavailable":
                    self._blocked.pop(agent_id)
        except Exception as failure:
            with self.member_lock(agent_id):
                self._blocked[agent_id] = "storage_unavailable"
                saved = self.history.runs(agent_id, limit=1)[0].messages_json
                self._failures[agent_id] = FailedFinalization(
                    run.sequence,
                    saved,
                    persisted_history,
                    usage_json,
                    error or str(failure),
                )
            if run_failure is not None:
                raise failure from run_failure
            raise
        finally:
            with self.member_lock(agent_id):
                self._active.pop(agent_id, None)
                with self._lock:
                    self._reserved.discard(agent_id)
            if window is not None:
                self.emit(
                    "window.reset",
                    {
                        "agent_id": agent_id,
                        "number": window.number,
                        "reason": window.reason,
                    },
                )
            self._changed(
                "turn.finished",
                {"agent_id": agent_id, "sequence": run.sequence, "status": status},
            )
        return TurnRecord(agent_id, run.sequence, status, error)

    def tick(self) -> tuple[int, ...]:
        self.parameters()
        if self._scheduler_stopped is not None or self._stop.is_set():
            return ()
        started: list[int] = []
        for agent_id in self.runnable_agents():
            prepared = self._prepare(agent_id)
            if prepared is None:
                continue

            def work(turn: PreparedTurn = prepared) -> None:
                try:
                    self._execute(turn)
                except Exception:
                    logger.exception(
                        "Turn finalization failed for agent %s", turn.run.agent_id
                    )
                finally:
                    self.wake()

            try:
                thread = threading.Thread(
                    target=work, name=f"fora-agent-{agent_id}", daemon=True
                )
                with self._lock:
                    self._threads[agent_id] = thread
                thread.start()
            except Exception as error:
                self._execute(prepared, error)
                raise
            started.append(agent_id)
        return tuple(started)

    def serve(self, poll_seconds: float = 0.2) -> None:
        try:
            while not self._stop.is_set():
                self.tick()
                self._wake.wait(timeout=poll_seconds)
                self._wake.clear()
        except Exception as error:
            self._scheduler_stopped = f"{type(error).__name__}: {error}"
            self.emit(
                "organization.changed", {"scheduler_stopped": self._scheduler_stopped}
            )
            logger.exception("Scheduler stopped after a runtime failure")
            raise
