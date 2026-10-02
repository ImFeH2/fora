from __future__ import annotations

from pathlib import Path

import pytest

from fora.adapters.files.tree import DirectoryTree, content_hash
from fora.core.errors import DomainError
from fora.services.library import Library
from fora.services.workspace import Workspace


@pytest.fixture
def library(tmp_path: Path) -> Library:
    return Library(DirectoryTree(tmp_path / "library"))


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    return Workspace(DirectoryTree(tmp_path / "agents" / "13" / "workspace"))


def test_library_round_trips_a_document(library: Library) -> None:
    entry = library.write("guides/onboarding.md", "welcome")
    document = library.read("guides/onboarding.md")
    assert document.content == "welcome"
    assert document.path == entry.path
    assert document.content_hash == content_hash("welcome")


def test_library_concurrent_edit_is_rejected_then_recoverable(library: Library) -> None:
    library.write("shared.md", "v1")
    first = library.read("shared.md")
    library.write("shared.md", "v2", expected_hash=first.content_hash)
    latest = library.read("shared.md")
    assert latest.content == "v2"
    assert library.write("shared.md", "v3", expected_hash=latest.content_hash)


def test_workspace_lists_non_markdown_and_hidden_files_without_changing_index(
    workspace,
) -> None:
    _, digest = workspace.read("MEMORY.md")
    workspace.write("MEMORY.md", "remember", expected_hash=digest)
    workspace.write(".config/settings.json", "{}")
    workspace.write("notes.txt", "details")
    (workspace._tree.root / "image.png").write_bytes(b"\x89PNG\xff")
    assert [entry.path for entry in workspace.list()] == [
        ".config",
        ".config/settings.json",
        "MEMORY.md",
        "image.png",
        "notes.txt",
    ]
    assert workspace.read("notes.txt")[0] == "details"
    with pytest.raises(DomainError) as error:
        workspace.read("image.png")
    assert error.value.code == "not_readable"
    assert workspace.index(16_384) == "remember"


def test_workspace_index_is_empty_without_files(workspace: Workspace) -> None:
    assert workspace.index(16_384) == ""


@pytest.mark.parametrize("content", ["", "  - remember this\n", "记忆😀"])
def test_workspace_index_returns_only_the_unchanged_index(
    workspace: Workspace, content: str
) -> None:
    _, digest = workspace.read("MEMORY.md")
    workspace.write("MEMORY.md", content, expected_hash=digest)
    workspace.write("topics/rewrite.md", "details")
    assert workspace.index(len(content.encode("utf-8"))) == content


def test_workspace_does_not_list_files_without_an_index(workspace: Workspace) -> None:
    workspace.write("topics/rewrite.md", "details")
    assert workspace.index(16_384) == ""


@pytest.mark.parametrize(
    "limit,prefix",
    [
        (0, ""),
        (1, ""),
        (3, "记"),
        (4, "记"),
        (6, "记忆"),
        (7, "记忆"),
        (8, "记忆"),
        (10, "记忆😀"),
    ],
)
def test_workspace_index_is_cut_at_a_utf8_boundary(
    workspace: Workspace, limit: int, prefix: str
) -> None:
    content = "记忆😀 more"
    _, digest = workspace.read("MEMORY.md")
    workspace.write("MEMORY.md", content, expected_hash=digest)
    assert workspace.index(limit) == prefix + (
        f"\n[MEMORY.md is longer than {limit} bytes and was cut here. "
        "Reorganize it: keep only what you must always remember and a map of "
        "your other workspace files.]"
    )
    assert len(prefix.encode("utf-8")) <= limit
    assert workspace.read("MEMORY.md")[0] == content


def test_library_and_workspace_are_separate_trees(
    library: Library, workspace: Workspace
) -> None:
    library.write("shared.md", "org wide")
    workspace.write("private.md", "mine")
    assert [item.path for item in library.list()] == ["shared.md"]
    assert [item.path for item in workspace.list()] == ["MEMORY.md", "private.md"]


def test_workspace_recreates_its_index_after_deletion_or_movement(
    workspace: Workspace,
) -> None:
    entries = workspace.list()
    assert [(entry.path, entry.kind, entry.size) for entry in entries] == [
        ("MEMORY.md", "file", 0)
    ]
    workspace.delete("MEMORY.md")
    assert workspace.index(100) == ""
    workspace.move("MEMORY.md", "other.md")
    assert workspace.index(100) == ""
    assert workspace.read("other.md")[0] == ""


def test_workspace_directory_operations_and_edit(workspace: Workspace) -> None:
    workspace.mkdir("topics")
    workspace.write("topics/note.md", "before")
    entry, diff = workspace.edit("topics/note.md", "before", "after")
    assert workspace.read(entry.path) == ("after", content_hash("after"))
    assert "+after" in diff
    workspace.move("topics", "renamed")
    assert [item.path for item in workspace.list("renamed")] == ["renamed/note.md"]
    workspace.delete("renamed")
    assert [item.path for item in workspace.list()] == ["MEMORY.md"]


def test_library_snapshots_detect_presence_and_metadata_changes(
    library: Library,
) -> None:
    library.write("changed.txt", "before")
    library.write("removed.txt", "gone")
    library.write("retained.txt", "same")
    (library.root / "binary").write_bytes(b"\xff")
    before = library.snapshot()
    library.edit("changed.txt", "before", "after")
    library.delete("removed.txt")
    library.delete("binary")
    library.mkdir("empty")
    library.write("added.txt", "new")
    (library.root / "new-binary").write_bytes(b"\xff")
    after = library.snapshot()
    assert Library.changes(before, after) == (
        "added.txt",
        "binary",
        "changed.txt",
        "new-binary",
        "removed.txt",
    )
    assert Library.changes(after, after) == ()
    assert "empty" not in after
    stat = (library.root / "new-binary").stat()
    assert after["new-binary"] == (1, stat.st_mtime_ns)


def test_library_snapshot_reads_metadata_only_including_hidden_files(
    library, monkeypatch
) -> None:
    import os

    hidden = library.root / ".hidden"
    hidden.mkdir()
    target = hidden / "note.txt"
    target.write_text("before", encoding="utf-8")
    stat = target.stat()

    def fail(*args, **kwargs):
        pytest.fail("Snapshots must not read file contents")

    monkeypatch.setattr(library._tree, "list", fail)
    monkeypatch.setattr(Path, "open", fail)
    before = library.snapshot()
    assert before == {".hidden/note.txt": (stat.st_size, stat.st_mtime_ns)}
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert Library.changes(before, library.snapshot()) == (".hidden/note.txt",)


def test_workspace_index_creation_failure_is_visible(tmp_path: Path) -> None:
    tree = DirectoryTree(tmp_path / "workspace")
    index = tree.root / "MEMORY.md"
    index.mkdir()
    retained = index / "retained.txt"
    retained.write_text("preserve", encoding="utf-8")

    with pytest.raises(DomainError) as error:
        Workspace(tree).index(100)

    assert error.value.code == "invalid_path"
    assert retained.read_text(encoding="utf-8") == "preserve"
    assert Workspace(DirectoryTree(tmp_path / "other")).index(100) == ""


def test_unreadable_workspace_index_does_not_hide_the_tree(tmp_path: Path) -> None:
    tree = DirectoryTree(tmp_path / "workspace")
    (tree.root / "MEMORY.md").write_bytes(b"\xff")
    workspace = Workspace(tree)
    (entry,) = workspace.list()
    assert entry.path == "MEMORY.md" and entry.size == 1
    workspace.write("other.md", "readable")
    assert workspace.read("other.md")[0] == "readable"
    with pytest.raises(DomainError) as error:
        workspace.index(100)
    assert error.value.code == "not_readable"
