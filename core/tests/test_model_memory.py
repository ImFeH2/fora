import asyncio
import gc
import threading
import weakref
from dataclasses import replace
from io import BytesIO

import pytest
from PIL import Image
from pydantic_ai import ModelMessagesTypeAdapter, models
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from test_runtime import MAIN

from fora.adapters.files.uploads import decode_image
from fora.adapters.model.runner import PydanticModelRunner
from fora.core.attachment import ViewedImage
from fora.runtime.reminder import HistoryPersistenceError, TurnRequest

pytest_plugins = ("test_runtime",)


@pytest.fixture
def model_world(world, monkeypatch):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    world.settings.set_settings(
        "model",
        {"base_url": "https://example.invalid", "api_key": "unused", "model": "local"},
    )
    return world


def request():
    return TurnRequest(
        agent_id=MAIN,
        sequence=1,
        agent_name="Lifecycle",
        prompt="Lifecycle request",
        reminder=None,
        history_json=ModelMessagesTypeAdapter.dump_json(
            [ModelResponse(parts=[TextPart("Saved history")])]
        ).decode(),
        resident="",
        environment=lambda: "",
        ephemeral=lambda: "",
        persist=lambda messages: None,
    )


def assert_released(references):
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize(
    "scenario",
    [
        "text",
        "image",
        "tool_failure",
        "model_failure",
        "cancel",
        "history_failure",
        "close_failure",
    ],
)
def test_completed_model_resources_release_on_all_exit_paths(model_world, scenario):
    references = []
    closed = []
    tool_threads = []
    calls = 0
    buffer = BytesIO()
    Image.new("RGB", (2, 3), (10, 20, 30)).save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    class Tools:
        def view_image(self, path: str):
            tool_threads.append(
                (threading.get_ident(), weakref.ref(threading.current_thread()))
            )
            if scenario == "tool_failure":
                raise OSError("controlled tool failure")
            image = decode_image(image_bytes)
            references.append(weakref.ref(image))
            return ViewedImage(path, image)

    async def respond(messages, info):
        nonlocal calls
        calls += 1
        if scenario == "model_failure":
            raise RuntimeError("controlled model failure")
        if scenario == "cancel":
            raise asyncio.CancelledError("controlled cancellation")
        if scenario in ("image", "tool_failure") and calls == 1:
            return ModelResponse(
                parts=[ToolCallPart("view_image", {"path": "/image.png"}, "image")]
            )
        return ModelResponse(parts=[TextPart("completed")])

    class OwnedModel(FunctionModel):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.append(True)
            if scenario == "close_failure":
                raise RuntimeError("controlled close failure")

    def build(config):
        model = OwnedModel(respond)
        references.extend(
            [
                weakref.ref(model),
                weakref.ref(asyncio.get_running_loop()),
                weakref.ref(asyncio.current_task()),
            ]
        )
        return model

    def persist(messages):
        raise OSError("controlled history failure")

    def execute():
        current = request()
        if scenario == "history_failure":
            current = replace(current, persist=persist)
        references.append(weakref.ref(current))
        runner = PydanticModelRunner(model_world.settings, build_model=build)
        if scenario == "cancel":
            with pytest.raises(asyncio.CancelledError, match="controlled cancellation"):
                runner.run(current, Tools())
        elif scenario == "history_failure":
            with pytest.raises(HistoryPersistenceError):
                runner.run(current, Tools())
        else:
            outcome = runner.run(current, Tools())
            if scenario in ("model_failure", "close_failure"):
                assert f"controlled {scenario.replace('_', ' ')}" in outcome.error
            else:
                assert outcome.error is None
                assert "completed" in outcome.messages_json
                if scenario == "tool_failure":
                    assert "tool_execution_failed" in outcome.messages_json
                if scenario == "image":
                    assert "binary" in outcome.messages_json.lower()

    execute()
    assert closed == [True]
    if scenario in ("image", "tool_failure"):
        assert len(tool_threads) == 1
        assert tool_threads[0][0] != threading.get_ident()
        thread = tool_threads[0][1]()
        if thread is not None:
            thread.join(5)
            assert not thread.is_alive()
        del thread
    assert_released(references)


def test_parallel_turn_completion_preserves_the_other_loop(model_world):
    entered = {name: threading.Event() for name in ("first", "second")}
    release = {name: threading.Event() for name in entered}
    references = {name: [] for name in entered}
    completed = []

    def build(config):
        name = threading.current_thread().name
        references[name].extend(
            [
                weakref.ref(asyncio.get_running_loop()),
                weakref.ref(asyncio.current_task()),
            ]
        )

        async def respond(messages, info):
            entered[name].set()
            assert await asyncio.to_thread(release[name].wait, 5)
            return ModelResponse(parts=[TextPart(name)])

        model = FunctionModel(respond)
        references[name].append(weakref.ref(model))
        return model

    runner = PydanticModelRunner(model_world.settings, build_model=build)

    def execute():
        name = threading.current_thread().name
        outcome = runner.run(request(), None)
        assert outcome.error is None
        assert name in outcome.messages_json
        completed.append(name)

    threads = {name: threading.Thread(target=execute, name=name) for name in entered}
    try:
        for thread in threads.values():
            thread.start()
        assert all(event.wait(5) for event in entered.values())
        release["first"].set()
        threads["first"].join(5)
        assert not threads["first"].is_alive()
        assert threads["second"].is_alive()
        assert_released(references["first"])
        assert all(reference() is not None for reference in references["second"])
        release["second"].set()
        threads["second"].join(5)
        assert not threads["second"].is_alive()
        assert_released(references["second"])
        assert sorted(completed) == ["first", "second"]
    finally:
        for event in release.values():
            event.set()
        for thread in threads.values():
            thread.join(5)
