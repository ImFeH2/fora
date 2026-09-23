from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from huddol.adapters.voice import model
from huddol.core.errors import DomainError


@pytest.mark.parametrize("failure", ["cancel", "short", "oversize", "digest", "http"])
def test_failed_download_preserves_model_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    content = b"test-model-content"
    cancel = threading.Event()
    store = model.ModelStore(tmp_path)
    store.path.parent.mkdir(parents=True)
    store.path.write_bytes(b"existing-model")
    monkeypatch.setattr(model, "MODEL_SIZE", len(content))
    monkeypatch.setattr(model, "MODEL_SHA256", hashlib.sha256(content).hexdigest())
    client_type = httpx.Client
    body = {
        "cancel": content,
        "short": content[:-1],
        "oversize": content + b"!",
        "digest": b"!" * len(content),
        "http": content,
    }[failure]
    response_status = 503 if failure == "http" else 200

    def request(_: httpx.Request) -> httpx.Response:
        if failure == "cancel":
            cancel.set()
        return httpx.Response(response_status, content=body)

    monkeypatch.setattr(
        model.httpx,
        "Client",
        lambda **kwargs: client_type(transport=httpx.MockTransport(request), **kwargs),
    )
    expected = httpx.HTTPStatusError if failure == "http" else DomainError
    with pytest.raises(expected):
        store.download(cancel, Mock())
    assert store.path.read_bytes() == b"existing-model"
    assert not (store.directory / f"{model.MODEL_REVISION}.partial").exists()
    assert not store._download.locked()

    cancel.clear()
    monkeypatch.setattr(
        model.httpx,
        "Client",
        lambda **kwargs: client_type(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=content)
            ),
            **kwargs,
        ),
    )
    progress = Mock()
    store.download(cancel, progress)
    assert store.path.read_bytes() == content
    assert not (store.directory / f"{model.MODEL_REVISION}.partial").exists()
    assert not store._download.locked()
    assert progress.call_args.args == (len(content), len(content))
    assert store.verify(cancel) >= 0
