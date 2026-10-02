from __future__ import annotations

from fora.core.turn import idle_streak, is_productive


def test_only_send_edit_and_run_count_as_output() -> None:
    assert is_productive(["send"])
    assert is_productive(["ack", "run"])
    assert not is_productive(["ack"])
    assert not is_productive(["ack", "workspace.write", "library.write"])
    assert not is_productive([])


def test_idle_streak_counts_leading_turns_without_output() -> None:
    runs = [
        ("completed", ["ack"]),
        ("completed", []),
        ("completed", ["send", "ack"]),
        ("completed", ["ack"]),
    ]
    assert idle_streak(runs) == 2


def test_idle_streak_is_zero_when_the_newest_turn_produced_something() -> None:
    assert idle_streak([("completed", ["edit"]), ("completed", ["ack"])]) == 0


def test_a_failed_turn_stops_the_streak_rather_than_extending_it() -> None:
    runs = [("failed", []), ("completed", ["ack"])]
    assert idle_streak(runs) == 0


def test_a_still_running_turn_is_not_counted() -> None:
    assert idle_streak([("running", []), ("completed", ["ack"])]) == 0
