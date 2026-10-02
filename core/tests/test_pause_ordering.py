import threading

import pytest
from test_runtime import HELPER, HUMAN, MAIN, RecordingRunner, mention

pytest_plugins = ("test_runtime",)

from fora.runtime.reminder import TurnOutcome
from fora.runtime.scheduler import Scheduler


def test_pause_committed_before_start_preserves_pending(world):
    room = mention(world)
    runner = RecordingRunner()
    scheduler = Scheduler(world, runner)
    assert scheduler.pause(MAIN)["state"] == "paused"
    assert scheduler.tick() == ()
    assert runner.requests == []
    assert world.history.runs(MAIN) == ()
    assert world.history.new_mentions(MAIN, [(room, 1)]) == {(room, 1)}


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_pause_after_commit_before_thread_execution(world, monkeypatch, resume, failed):
    room = mention(world)
    helper = world.store.create_discussion("helper", [HUMAN, HELPER])
    world.store.append_message(helper.id, HUMAN, "@Helper ready")
    world.settings.set_settings("agent", {"max_concurrent_turns": 1})
    entered = threading.Event()
    release = threading.Event()
    model_release = threading.Event()
    calls = []
    starts = []
    original_start = threading.Thread.start

    def controlled_start(thread):
        if thread.name == f"fora-agent-{MAIN}":
            entered.set()
            assert release.wait(10)
        original_start(thread)

    def respond(request, tools):
        calls.append(request.agent_id)
        assert model_release.wait(10)
        tools.send_message(room, "finished current work")
        return TurnOutcome("[]", error="model failed" if failed else None)

    scheduler = Scheduler(world, RecordingRunner(respond))
    launcher = threading.Thread(target=lambda: starts.append(scheduler.tick()))
    monkeypatch.setattr(threading.Thread, "start", controlled_start)
    try:
        original_start(launcher)
        assert entered.wait(5)
        assert calls == []
        assert scheduler.pause(MAIN)["state"] == "running"
        assert scheduler.tick() == ()
        if resume:
            assert scheduler.resume(MAIN)["state"] == "running"
            assert not scheduler.agent_status(MAIN)["pause_requested"]
        release.set()
        launcher.join(5)
        model_release.set()
        scheduler._threads[MAIN].join(5)
        assert starts == [(MAIN,)]
        assert calls == [MAIN]
        assert world.store.messages(room)[-1].body == "finished current work"
        assert scheduler.agent_status(MAIN)["state"] == (
            "paused" if not resume else "error" if failed else "idle"
        )
        assert MAIN not in scheduler._reserved
    finally:
        release.set()
        model_release.set()
        launcher.join(5)
        scheduler.stop()


@pytest.mark.parametrize("resume", [False, True])
def test_operation_after_finalization_uses_committed_state(world, resume):
    room = mention(world)
    entered = threading.Event()
    release = threading.Event()

    def respond(request, tools):
        entered.set()
        assert release.wait(10)
        tools.send_message(room, "completed")
        return TurnOutcome("[]")

    scheduler = Scheduler(world, RecordingRunner(respond))
    try:
        scheduler.tick()
        assert entered.wait(5)
        if resume:
            scheduler.pause(MAIN)
        release.set()
        scheduler._threads[MAIN].join(5)
        assert MAIN not in scheduler._active
        result = scheduler.resume(MAIN) if resume else scheduler.pause(MAIN)
        assert result["state"] == ("idle" if resume else "paused")
        assert world.history.runs(MAIN)[0].status == "completed"
        assert scheduler._reserved == set()
    finally:
        release.set()
        scheduler.stop()
