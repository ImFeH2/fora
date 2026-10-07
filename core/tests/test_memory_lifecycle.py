import gc
import json
import threading
import weakref

import pytest
from test_runtime import HUMAN, MAIN, RecordingRunner, mention, turn_thread

from fora.core.errors import DomainError
from fora.runtime.reminder import TurnOutcome
from fora.runtime.scheduler import Scheduler
from fora.tools.authorize import Actor

pytest_plugins = ("test_runtime",)


def test_member_lock_identity_covers_holder_waiter_and_reentry(world):
    scheduler = Scheduler(world, RecordingRunner())
    lock = scheduler.member_lock(MAIN)
    reference = weakref.ref(lock)
    waiting = threading.Event()
    acquired = threading.Event()
    release = threading.Event()
    identities = []

    def waiter():
        candidate = scheduler.member_lock(MAIN)
        identities.append(id(candidate))
        waiting.set()
        with candidate:
            acquired.set()
            assert release.wait(5)
            with scheduler.member_lock(MAIN) as nested:
                identities.append(id(nested))

    worker = threading.Thread(target=waiter)
    with lock:
        with scheduler.member_lock(MAIN) as nested:
            assert nested is lock
        worker.start()
        assert waiting.wait(5)
        assert not acquired.is_set()
        assert scheduler.member_lock(MAIN) is lock
        gc.collect()
        assert reference() is lock
    assert acquired.wait(5)
    assert scheduler.member_lock(MAIN) is lock
    release.set()
    worker.join(5)
    assert not worker.is_alive()
    assert identities == [id(lock), id(lock)]
    del lock, nested
    gc.collect()
    assert reference() is None
    assert MAIN not in scheduler._member_locks


def test_finished_threads_are_reaped_by_idle_maintenance(world):
    mention(world)
    scheduler = Scheduler(
        world, RecordingRunner(lambda request, tools: TurnOutcome("[]", error="failed"))
    )
    assert scheduler.tick() == (MAIN,)
    thread = turn_thread(scheduler, MAIN)
    reference = weakref.ref(thread)
    thread.join(5)
    assert not thread.is_alive()
    assert thread in scheduler._threads
    assert scheduler.tick() == ()
    assert scheduler._threads == set()
    del thread
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("stage", ["construct", "start"])
def test_thread_start_failure_removes_registration(world, monkeypatch, stage):
    mention(world)
    scheduler = Scheduler(world, RecordingRunner())

    def fail(*args, **kwargs):
        raise RuntimeError("controlled Thread startup failure")

    if stage == "construct":
        monkeypatch.setattr(threading, "Thread", fail)
    else:
        monkeypatch.setattr(threading.Thread, "start", fail)
    with pytest.raises(RuntimeError, match="startup failure"):
        scheduler.tick()
    assert scheduler._threads == scheduler._starting_threads == set()
    assert scheduler._active == {}
    assert scheduler._reserved == set()
    assert world.history.runs(MAIN)[0].status == "failed"


def test_stop_waits_for_two_turn_threads_with_the_same_member(world):
    room = mention(world)
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    stopped = threading.Event()
    finished = []

    def on_event(name, payload):
        if name == "turn.finished":
            index = payload["sequence"] - 1
            entered[index].set()
            assert release[index].wait(5)
            finished.append(payload["sequence"])

    scheduler = Scheduler(world, RecordingRunner(), on_event=on_event)
    stopper = threading.Thread(target=lambda: (scheduler.stop(), stopped.set()))
    try:
        assert scheduler.tick() == (MAIN,)
        assert entered[0].wait(5)
        first = turn_thread(scheduler, MAIN)
        world.store.append_message(room, HUMAN, "@Main next")
        assert scheduler.tick() == (MAIN,)
        assert entered[1].wait(5)
        assert len(scheduler._threads) == 2
        second = next(thread for thread in scheduler._threads if thread is not first)
        stopper.start()
        assert not stopped.wait(0.05)
        release[0].set()
        first.join(5)
        assert not first.is_alive()
        assert second.is_alive()
        assert not stopped.wait(0.05)
        release[1].set()
        stopper.join(5)
        assert stopped.is_set()
        assert finished == [1, 2]
        assert scheduler._threads == set()
    finally:
        for event in release:
            event.set()
        if stopper.ident is not None:
            stopper.join(5)
        scheduler.stop()


def test_maintenance_and_stop_preserve_thread_start_window(world, monkeypatch):
    mention(world)
    entering = threading.Event()
    release_start = threading.Event()
    entering_model = threading.Event()
    release_model = threading.Event()
    stopped = threading.Event()
    original_start = threading.Thread.start

    def start(thread):
        if thread.name == f"fora-agent-{MAIN}":
            entering.set()
            assert release_start.wait(5)
        original_start(thread)

    def respond(request, tools):
        entering_model.set()
        assert release_model.wait(5)
        return TurnOutcome("[]")

    scheduler = Scheduler(world, RecordingRunner(respond))
    launcher = threading.Thread(target=scheduler.tick)
    stopper = threading.Thread(target=lambda: (scheduler.stop(), stopped.set()))
    monkeypatch.setattr(threading.Thread, "start", start)
    try:
        launcher.start()
        assert entering.wait(5)
        thread = turn_thread(scheduler, MAIN)
        assert thread.ident is None
        assert scheduler.tick() == ()
        assert thread in scheduler._threads
        assert scheduler.pause(MAIN)["state"] == "running"
        stopper.start()
        assert not stopped.wait(0.05)
        release_start.set()
        assert entering_model.wait(5)
        assert not stopped.wait(0.05)
        release_model.set()
        launcher.join(5)
        stopper.join(5)
        assert stopped.is_set()
        assert scheduler._threads == scheduler._starting_threads == set()
        assert world.history.runs(MAIN)[0].status == "completed"
        assert scheduler.agent_status(MAIN)["state"] == "paused"
    finally:
        release_start.set()
        release_model.set()
        launcher.join(5)
        if stopper.ident is not None:
            stopper.join(5)
        scheduler.stop()


def failed_scheduler(world):
    mention(world)
    history = json.dumps([{"kind": "response", "content": "h" * 2097152}])

    def respond(request, tools):
        request.persist(history)
        world.store._db.execute(
            "CREATE TEMP TRIGGER reject_finish BEFORE UPDATE OF completed_at ON agent_runs BEGIN SELECT RAISE(ABORT, 'finish rejected'); END"
        )
        return TurnOutcome(history)

    scheduler = Scheduler(world, RecordingRunner(respond))
    with pytest.raises(Exception, match="finish rejected"):
        scheduler.run_turn(MAIN)
    scheduler._pause_requested.add(MAIN)
    assert MAIN in scheduler._failures
    return scheduler


def test_successful_deletion_releases_failed_history_and_pause_intent(world):
    scheduler = failed_scheduler(world)
    failure = weakref.ref(scheduler._failures[MAIN])
    human = scheduler.tools_for_actor(Actor(HUMAN, False))
    assert human.delete_agent(MAIN) == {"id": MAIN, "deleted": True}
    gc.collect()
    assert failure() is None
    assert MAIN not in scheduler._failures
    assert MAIN not in scheduler._blocked
    assert MAIN not in scheduler._pause_requested
    assert any(item["id"] == MAIN for item in human.list_members(include_deleted=True))
    assert MAIN not in scheduler._member_locks
    with pytest.raises(DomainError, match="does not exist"):
        scheduler.pause(MAIN)
    with pytest.raises(DomainError, match="does not exist"):
        scheduler.resume(MAIN)
    assert scheduler.run_turn(MAIN) is None
    assert scheduler._failures == {}
    assert scheduler._blocked == {}
    assert scheduler._pause_requested == set()


def test_delete_transaction_failure_retains_recovery_history(world):
    scheduler = failed_scheduler(world)
    failure = scheduler._failures[MAIN]
    world.store._db.execute(
        "CREATE TEMP TRIGGER reject_delete BEFORE UPDATE OF deleted ON members BEGIN SELECT RAISE(ABORT, 'delete rejected'); END"
    )
    human = scheduler.tools_for_actor(Actor(HUMAN, False))
    with pytest.raises(Exception, match="delete rejected"):
        human.delete_agent(MAIN)
    assert not world.store.get_member(MAIN).deleted
    assert scheduler._failures[MAIN] is failure
    assert scheduler._blocked[MAIN] == "storage_unavailable"
    assert MAIN in scheduler._pause_requested
    world.store._db.execute("DROP TRIGGER reject_delete")
    world.store._db.execute("DROP TRIGGER reject_finish")
    assert scheduler.resume(MAIN)["state"] == "idle"
    assert len(world.history.latest_messages(MAIN)) >= 2097152
    assert scheduler._failures == {}


def test_pause_waiting_behind_successful_deletion_cannot_restore_state(world):
    scheduler = failed_scheduler(world)
    human = scheduler.tools_for_actor(Actor(HUMAN, False))
    waiting = threading.Event()
    errors = []

    def pause():
        waiting.set()
        try:
            scheduler.pause(MAIN)
        except DomainError as error:
            errors.append(error.code)

    worker = threading.Thread(target=pause)
    with scheduler.member_lock(MAIN):
        worker.start()
        assert waiting.wait(5)
        human.delete_agent(MAIN)
    worker.join(5)
    assert not worker.is_alive()
    assert errors == ["not_found"]
    assert scheduler._failures == {}
    assert scheduler._blocked == {}
    assert scheduler._pause_requested == set()
