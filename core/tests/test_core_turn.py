from __future__ import annotations

import pytest

from fora.core.turn import idle_streak, is_productive, model_error_signature


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


@pytest.mark.parametrize(
    "first, second",
    [
        ("request_id: req-abc123", "request_id: req-other999"),
        ("trace_id='trace-a'", "trace_id='trace-b'"),
        ("diagnostic abc123", "diagnostic def456"),
        (
            "123e4567-e89b-12d3-a456-426614174000",
            "123e4567-e89b-12d3-a456-426614174001",
        ),
        ("a" * 32, "b" * 32),
        ("tokens 4000 after 12.5ms", "tokens 6000 after 8ms"),
        ("duration 12 seconds", "duration 27 seconds"),
        ("duration 12 seconds", "duration 27ms"),
        ("  empty\n response  ", "empty response"),
    ],
)
def test_model_error_signature_normalizes_variable_details(first, second):
    assert model_error_signature(
        "UnexpectedModelBehavior: " + first
    ) == model_error_signature("UnexpectedModelBehavior: " + second)


@pytest.mark.parametrize(
    "first, second",
    [
        ("UnexpectedModelBehavior: empty", "ContentFilterError: empty"),
        ("UnexpectedModelBehavior: empty", "UnexpectedModelBehavior: filtered"),
        ("ModelHTTPError: status_code: 400", "ModelHTTPError: status_code: 429"),
        ("ModelHTTPError: model_name: gpt-6.1", "ModelHTTPError: model_name: gpt-6.2"),
        ("ModelHTTPError: {'code': 400}", "ModelHTTPError: {'code': 429}"),
        (
            "ModelHTTPError: {'code': 'server_error'}",
            "ModelHTTPError: {'code': 'rate_limit'}",
        ),
        (
            "ModelHTTPError: {'type': 'failure2'}",
            "ModelHTTPError: {'type': 'failure3'}",
        ),
        ("UnexpectedModelBehavior: tool2", "UnexpectedModelBehavior: tool3"),
        ("UnexpectedModelBehavior: model-6.1", "UnexpectedModelBehavior: model-6.2"),
        ("UnexpectedModelBehavior: response", "UnexpectedModelBehavior: request"),
        (
            "UnexpectedModelBehavior: diagnostic failed",
            "UnexpectedModelBehavior: diagnostic unavailable",
        ),
        (
            "UnexpectedModelBehavior: request id unavailable",
            "UnexpectedModelBehavior: request id invalid",
        ),
        (
            "UnexpectedModelBehavior: trace id unavailable",
            "UnexpectedModelBehavior: trace id invalid",
        ),
    ],
)
def test_model_error_signature_preserves_failure_identity(first, second):
    assert model_error_signature(first) != model_error_signature(second)


def test_model_error_signature_compares_complete_text_and_keeps_original():
    original = "UnexpectedModelBehavior: " + "same reason " * 1000
    assert model_error_signature(original + "empty") != model_error_signature(
        original + "filtered"
    )
    assert original.endswith("same reason ")


def test_model_error_signature_preserves_fixed_fields_among_variable_values():
    error = "ModelHTTPError: " + ", ".join(
        f"status_code: 502, model_name: gpt-6.1, request_id: req-{index}, after {index + 1}ms"
        for index in range(1000)
    )
    signature = model_error_signature(error)
    assert signature.count("status_code: 502") == 1000
    assert signature.count("model_name: gpt-6.1") == 1000
    assert signature.count("<id>") == 1000
    assert signature.count("<duration>") == 1000
