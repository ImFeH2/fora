from __future__ import annotations

import gc
import multiprocessing
import os
import subprocess
import sys
import weakref
from builtins import ExceptionGroup
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event, local
from unittest.mock import Mock

import pytest

from fora.adapters import file_writes
from fora.adapters.execution import editing
from fora.adapters.execution.local import LocalExecution
from fora.adapters.files import tree as tree_module
from fora.adapters.files.tree import DirectoryTree, content_hash
from fora.core.errors import DomainError
from fora.ports.files import ConflictError
from fora.services.library import Library
from fora.services.workspace import Workspace


def _create_from_process(path: str, content: str, barrier, results) -> None:
    barrier.wait()
    try:
        editing.edit_file(
            path,
            [{"old_text": "", "new_text": content}],
            directories=[str(Path(path).parent)],
            create=True,
        )
    except DomainError as error:
        results.put(error.code)
    else:
        results.put("created")


@pytest.fixture
def ordered_writes(monkeypatch):
    def run(first, second, *, stage="publish"):
        paused = Event()
        attempted = Event()
        worker = local()
        original_lock = file_writes.directory_lock
        original_replace = os.replace
        original_entry = DirectoryTree._entry

        @contextmanager
        def observed_lock(target):
            lock = original_lock(target)
            if getattr(worker, "contender", False):
                acquired = lock.acquire(blocking=False)
                if acquired:
                    lock.release()
                attempted.set()
                assert not acquired, (
                    "The first writer must still hold the directory lock"
                )
            with lock:
                yield

        def pause():
            if not getattr(worker, "contender", False):
                paused.set()
                assert attempted.wait(10), "The competing writer did not reach the lock"

        def publish(source, target):
            if stage == "publish":
                pause()
            return original_replace(source, target)

        def entry(instance, target):
            if stage == "metadata":
                pause()
            return original_entry(instance, target)

        def compete():
            worker.contender = True
            return second()

        with monkeypatch.context() as patch:
            patch.setattr(tree_module, "directory_lock", observed_lock)
            patch.setattr(editing, "directory_lock", observed_lock)
            patch.setattr(file_writes.os, "replace", publish)
            patch.setattr(DirectoryTree, "_entry", entry)
            with ThreadPoolExecutor(max_workers=2) as pool:
                winner = pool.submit(first)
                assert paused.wait(10), "The first writer did not reach the pause point"
                contender = pool.submit(compete)
            return winner, contender

    return run


@pytest.mark.parametrize("name", ["same.txt", "nested/deep/记忆.txt"])
@pytest.mark.parametrize("service", [DirectoryTree, Library, Workspace])
def test_concurrent_creation_has_one_winner(tmp_path, ordered_writes, name, service):
    trees = [DirectoryTree(tmp_path) for _ in range(2)]
    writers = trees if service is DirectoryTree else [service(tree) for tree in trees]
    if service is Workspace:
        writers[0].list()
    winner, contender = ordered_writes(
        lambda: writers[0].write(name, "first"),
        lambda: writers[1].write(name, "second"),
    )
    assert winner.result().size == len(b"first")
    with pytest.raises(DomainError) as error:
        contender.result()
    assert error.value.code == "expected_hash_required"
    assert (tmp_path / name).read_bytes() == b"first"
    expected = {tmp_path / name}
    if service is Workspace and (tmp_path / name).parent == tmp_path:
        expected.add(tmp_path / "MEMORY.md")
    assert set((tmp_path / name).parent.iterdir()) == expected


@pytest.mark.parametrize("stage", ["publish", "metadata"])
def test_concurrent_hash_updates_reject_stale_content(tmp_path, ordered_writes, stage):
    first, second = DirectoryTree(tmp_path), DirectoryTree(tmp_path)
    first.write("same.txt", "original")
    digest = content_hash("original")
    winner, contender = ordered_writes(
        lambda: first.write("same.txt", "first", expected_hash=digest),
        lambda: second.write("same.txt", "second", expected_hash=digest),
        stage=stage,
    )
    assert winner.result().size == len(b"first")
    with pytest.raises(ConflictError) as error:
        contender.result()
    assert error.value.actual == content_hash("first")
    assert first.read("same.txt")[0] == "first"


@pytest.mark.parametrize("first_edit", [False, True])
def test_agent_edit_and_write_share_the_lock(tmp_path, ordered_writes, first_edit):
    tree = DirectoryTree(tmp_path)
    tree.write("same.txt", "original")
    execution = LocalExecution([str(tmp_path)], enforce=False)
    write = lambda: tree.write(
        "same.txt", "written", expected_hash=content_hash("original")
    )
    edit = lambda: execution.edit(
        str(tmp_path / "same.txt"), [{"old_text": "original", "new_text": "edited"}]
    )
    try:
        winner, contender = ordered_writes(
            edit if first_edit else write, write if first_edit else edit
        )
        winner.result()
        with pytest.raises(ConflictError if first_edit else DomainError) as error:
            contender.result()
        if not first_edit:
            assert error.value.code == "no_match"
        assert tree.read("same.txt")[0] == ("edited" if first_edit else "written")
    finally:
        execution.close()


def test_edit_create_thread_race_has_one_winner(tmp_path):
    target = tmp_path / "same.txt"
    start = Barrier(2)

    def create(content):
        start.wait()
        try:
            editing.edit_file(
                str(target),
                [{"old_text": "", "new_text": content}],
                directories=[str(tmp_path)],
                create=True,
            )
        except DomainError as error:
            return error.code
        return "created"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(create, ["first", "second"]))
    assert sorted(results) == ["already_exists", "created"]
    assert target.read_text(encoding="utf-8") in {"first", "second"}
    assert not list(tmp_path.glob(".fora-create-*"))


def test_edit_create_process_race_has_one_winner(tmp_path):
    target = tmp_path / "same.txt"
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_create_from_process,
            args=(str(target), content, barrier, results),
        )
        for content in ("first", "second")
    ]
    try:
        for process in processes:
            process.start()
        outcomes = [results.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0
        assert sorted(outcomes) == ["already_exists", "created"]
        assert target.read_text(encoding="utf-8") in {"first", "second"}
        assert not list(tmp_path.glob(".fora-create-*"))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        results.close()


def test_edit_create_publishes_only_complete_content(tmp_path, monkeypatch):
    target = tmp_path / "new.txt"
    publish = os.link

    def observe(source, destination):
        assert source.parent == destination.parent
        assert source.read_text(encoding="utf-8") == "complete body"
        assert not destination.exists()
        publish(source, destination)
        assert destination.read_text(encoding="utf-8") == "complete body"

    monkeypatch.setattr(file_writes.os, "link", observe)
    result = editing.edit_file(
        str(target),
        [{"old_text": "", "new_text": "complete body"}],
        directories=[str(tmp_path)],
        create=True,
    )
    assert result.replacements == 0
    assert target.read_text(encoding="utf-8") == "complete body"
    assert list(tmp_path.iterdir()) == [target]


def test_edit_create_publish_failure_removes_temporary_file(tmp_path, monkeypatch):
    target = tmp_path / "new.txt"
    failure = PermissionError("hard links are unavailable")
    monkeypatch.setattr(file_writes.os, "link", Mock(side_effect=failure))
    with pytest.raises(PermissionError) as error:
        editing.edit_file(
            str(target),
            [{"old_text": "", "new_text": "complete"}],
            directories=[str(tmp_path)],
            create=True,
        )
    assert error.value is failure
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_edit_create_cleanup_failure_reports_published_file(tmp_path, monkeypatch):
    target = tmp_path / "new.txt"
    original_unlink = Path.unlink
    failure = PermissionError("temporary cleanup failed")

    def unlink(path, *args, **kwargs):
        if path.name.startswith(".fora-create-"):
            raise failure
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", unlink)
        with pytest.raises(DomainError) as error:
            editing.edit_file(
                str(target),
                [{"old_text": "", "new_text": "complete"}],
                directories=[str(tmp_path)],
                create=True,
            )
    assert error.value.code == "write_published"
    assert target.read_text(encoding="utf-8") == "complete"
    assert len(list(tmp_path.glob(".fora-create-*"))) == 1
    for temporary in tmp_path.glob(".fora-create-*"):
        original_unlink(temporary)


def test_tree_edit_holds_lock_through_metadata(tmp_path, ordered_writes):
    first, second = DirectoryTree(tmp_path), DirectoryTree(tmp_path)
    first.write("same.txt", "original")
    winner, contender = ordered_writes(
        lambda: first.edit("same.txt", "original", "edited"),
        lambda: second.write(
            "same.txt", "second", expected_hash=content_hash("edited")
        ),
        stage="metadata",
    )
    assert winner.result()[0].size == len(b"edited")
    assert contender.result().size == len(b"second")
    assert first.read("same.txt")[0] == "second"


def test_same_directory_independent_files_both_succeed(tmp_path, ordered_writes):
    first, second = DirectoryTree(tmp_path), DirectoryTree(tmp_path)
    winner, contender = ordered_writes(
        lambda: first.write("one.txt", "first"),
        lambda: second.write("two.txt", "second"),
    )
    winner.result()
    contender.result()
    assert first.read("one.txt")[0] == "first"
    assert second.read("two.txt")[0] == "second"


def test_different_directories_publish_concurrently(tmp_path, monkeypatch):
    tree = DirectoryTree(tmp_path)
    barrier = Barrier(2)
    original = os.replace

    def publish(source, target):
        barrier.wait(10)
        original(source, target)

    monkeypatch.setattr(file_writes.os, "replace", publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(tree.write, f"{name}/same.txt", name) for name in ("a", "b")
        ]
        for future in futures:
            future.result(timeout=15)
    assert tree.read("a/same.txt")[0] == "a"
    assert tree.read("b/same.txt")[0] == "b"


def test_directory_aliases_share_creation_lock(tmp_path, ordered_writes):
    tree = DirectoryTree(tmp_path)
    tree.mkdir("real")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(tmp_path / "real", target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"Directory symlinks unavailable: {error}")
    winner, contender = ordered_writes(
        lambda: tree.write("real/new.txt", "first"),
        lambda: DirectoryTree(tmp_path).write("alias/new.txt", "second"),
    )
    winner.result()
    with pytest.raises(DomainError) as error:
        contender.result()
    assert error.value.code == "expected_hash_required"
    assert tree.read("alias/new.txt")[0] == "first"


def test_case_aliases_share_creation_lock_when_supported(tmp_path, ordered_writes):
    tree = DirectoryTree(tmp_path)
    tree.mkdir("MixedCase")
    if not (tmp_path / "mixedcase").is_dir():
        pytest.skip("The test filesystem is case-sensitive")
    winner, contender = ordered_writes(
        lambda: tree.write("MixedCase/New.txt", "first"),
        lambda: DirectoryTree(tmp_path).write("mixedcase/new.TXT", "second"),
    )
    winner.result()
    with pytest.raises(DomainError) as error:
        contender.result()
    assert error.value.code == "expected_hash_required"
    assert tree.read("mixedcase/new.txt")[0] == "first"


def test_lock_identity_and_lifetime_use_parent_stat(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    target = tmp_path / "new.txt"
    lock = file_writes.directory_lock(target)
    assert lock is file_writes.directory_lock(tmp_path / "different.txt")
    assert lock is not file_writes.directory_lock(other / "new.txt")
    identity = tmp_path.stat()
    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", lambda self: identity)
        assert lock is file_writes.directory_lock(Path("Alias/New.txt"))
    reference = weakref.ref(lock)
    del lock
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("operation", ["create", "write", "edit", "agent"])
@pytest.mark.parametrize(
    "failure", ["write", "replace", "cleanup", "metadata", "double"]
)
def test_failures_preserve_file_state_and_diagnostics(
    tmp_path, monkeypatch, operation, failure
):
    tree = DirectoryTree(tmp_path)
    target = tmp_path / "document.txt"
    if operation != "create":
        target.write_bytes(b"original\r\n")
    primary = OSError("injected primary failure")
    cleanup = PermissionError("injected cleanup failure")
    temporary_files = []
    original_temporary = file_writes.tempfile.NamedTemporaryFile
    original_unlink = Path.unlink

    @contextmanager
    def temporary(**kwargs):
        with original_temporary(**kwargs) as stream:
            temporary_files.append(Path(stream.name))
            if failure == "write":

                def partial_write(data):
                    stream.write(data[:3])
                    raise primary

                wrapped = Mock(wraps=stream)
                wrapped.name = stream.name
                wrapped.write.side_effect = partial_write
                yield wrapped
            else:
                yield stream

    def unlink(path, *args, **kwargs):
        if path in temporary_files and failure in ("cleanup", "double"):
            raise cleanup
        return original_unlink(path, *args, **kwargs)

    def fail(*args, **kwargs):
        raise primary

    monkeypatch.setattr(file_writes.tempfile, "NamedTemporaryFile", temporary)
    monkeypatch.setattr(Path, "unlink", unlink)
    if failure in ("replace", "double"):
        monkeypatch.setattr(file_writes.os, "replace", fail)
    if failure == "metadata":
        if operation == "agent":
            monkeypatch.setattr(editing.difflib, "unified_diff", fail)
        else:
            monkeypatch.setattr(tree, "_entry", fail)

    def execute():
        if operation in ("create", "write"):
            return tree.write(
                "document.txt",
                "updated\r\n",
                expected_hash=content_hash("original\r\n"),
            )
        if operation == "edit":
            return tree.edit("document.txt", "original", "updated")
        return editing.edit_file(
            str(target),
            [{"old_text": "original", "new_text": "updated"}],
            directories=[str(tmp_path)],
        )

    with pytest.raises((OSError, DomainError, ExceptionGroup)) as caught:
        execute()
    published = failure in ("cleanup", "metadata")
    if published:
        assert target.read_bytes() == b"updated\r\n"
        assert caught.value.code == "write_published"
        assert "already published" in str(caught.value)
        assert caught.value.__cause__ is (cleanup if failure == "cleanup" else primary)
    else:
        assert target.exists() == (operation != "create")
        if target.exists():
            assert target.read_bytes() == b"original\r\n"
        if failure == "double":
            assert caught.value.exceptions == (primary, cleanup)
            assert "published=False" in str(caught.value)
        else:
            assert caught.value is primary
    assert len(temporary_files) == 1
    if failure == "double":
        assert temporary_files[0].read_bytes() == b"updated\r\n"
        original_unlink(temporary_files[0])
    else:
        assert not temporary_files[0].exists()


def test_metadata_and_cleanup_failures_preserve_both_causes(tmp_path, monkeypatch):
    tree = DirectoryTree(tmp_path)
    primary = OSError("metadata failed")
    cleanup = PermissionError("cleanup failed")
    monkeypatch.setattr(tree, "_entry", Mock(side_effect=primary))
    monkeypatch.setattr(Path, "unlink", Mock(side_effect=cleanup))
    with pytest.raises(ExceptionGroup) as caught:
        tree.write("same.txt", "complete")
    published, failed_cleanup = caught.value.exceptions
    assert published.code == "write_published"
    assert published.__cause__ is primary
    assert failed_cleanup is cleanup
    assert "published=True" in str(caught.value)
    assert (tmp_path / "same.txt").read_bytes() == b"complete"
    assert list(tmp_path.iterdir()) == [tmp_path / "same.txt"]


@pytest.mark.parametrize("existing", [False, True])
def test_external_process_writes_are_outside_the_lock(tmp_path, monkeypatch, existing):
    tree = DirectoryTree(tmp_path)
    target = tmp_path / "same.txt"
    if existing:
        target.write_bytes(b"original")
    original_replace = os.replace

    def publish(source, destination):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b'external')",
                str(destination),
            ],
            check=True,
            timeout=10,
        )
        assert destination.read_bytes() == b"external"
        original_replace(source, destination)

    monkeypatch.setattr(file_writes.os, "replace", publish)
    tree.write("same.txt", "cooperative", expected_hash=content_hash("original"))
    assert target.read_bytes() == b"cooperative"


def test_identical_content_keeps_the_expected_hash_valid(tmp_path, ordered_writes):
    tree = DirectoryTree(tmp_path)
    tree.write("same.txt", "original")
    digest = content_hash("original")
    winner, contender = ordered_writes(
        lambda: tree.write("same.txt", "original", expected_hash=digest),
        lambda: DirectoryTree(tmp_path).write(
            "same.txt", "updated", expected_hash=digest
        ),
    )
    winner.result()
    contender.result()
    assert tree.read("same.txt")[0] == "updated"


@pytest.mark.parametrize("operation", ["write", "edit"])
def test_encoding_failure_cleans_temporary_and_releases_lock(tmp_path, operation):
    tree = DirectoryTree(tmp_path)
    tree.write("same.txt", "original")
    with pytest.raises(UnicodeEncodeError):
        if operation == "write":
            tree.write("same.txt", "\ud800", expected_hash=content_hash("original"))
        else:
            tree.edit("same.txt", "original", "\ud800")
    assert tree.read("same.txt")[0] == "original"
    assert list(tmp_path.iterdir()) == [tmp_path / "same.txt"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(tree.edit, "same.txt", "original", "updated")
        assert result.result(timeout=10)[0].size == len(b"updated")


def test_reader_observes_complete_content_at_publication(tmp_path, monkeypatch):
    tree = DirectoryTree(tmp_path)
    original = "old\r\n" * 1000
    updated = "记忆\r\n" * 1000
    tree.write("same.txt", original)
    publish = os.replace

    def inspect(source, target):
        assert source.parent == target.parent
        assert source.read_bytes() == updated.encode("utf-8")
        assert tree.read("same.txt")[0] == original
        publish(source, target)
        assert tree.read("same.txt")[0] == updated

    monkeypatch.setattr(file_writes.os, "replace", inspect)
    tree.write("same.txt", updated, expected_hash=content_hash(original))
    assert list(tmp_path.iterdir()) == [tmp_path / "same.txt"]
