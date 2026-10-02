from __future__ import annotations

import os
from pathlib import Path

import pytest

from fora.adapters.files.tree import DirectoryTree, content_hash
from fora.core.errors import DomainError
from fora.ports.files import ConflictError


@pytest.fixture
def tree(tmp_path: Path) -> DirectoryTree:
    return DirectoryTree(tmp_path / "library")


def require_symlinks(tmp_path: Path) -> None:
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(tmp_path)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks are unavailable on this host: {error}")
    probe.unlink()


def test_creates_reads_and_lists_nested_files_and_empty_directories(tree) -> None:
    tree.write("notes.txt", "top level")
    tree.write("guides/protocol.md", "nested")
    tree.mkdir("empty")
    tree.write("guides.txt", "sibling")
    entries = tree.list()
    assert [(item.path, item.kind) for item in entries] == [
        ("empty", "directory"),
        ("guides", "directory"),
        ("guides/protocol.md", "file"),
        ("guides.txt", "file"),
        ("notes.txt", "file"),
    ]
    assert all(item.size == 0 for item in entries[:2])
    assert all(not hasattr(item, "content_hash") for item in entries)
    assert all(item.modified_at.endswith("Z") for item in entries)
    assert [item.path for item in tree.list("guides")] == ["guides/protocol.md"]
    assert tree.read("guides/protocol.md") == ("nested", content_hash("nested"))
    assert tree.root.is_absolute()


@pytest.mark.parametrize("path", ["", ".", "./", "/", " / ", "///", " \t\n", "/ \t/\n"])
def test_root_paths_allow_listing_and_mkdir_but_not_file_operations(tree, path) -> None:
    tree.write("a/b.md", "content")
    assert tree.list(path) == tree.list()
    entry = tree.mkdir(path)
    assert entry.path == "" and entry.kind == "directory"
    assert entry.size == 0
    for operation in (
        lambda: tree.read(path),
        lambda: tree.write(path, "replacement"),
        lambda: tree.edit(path, "content", "replacement"),
        lambda: tree.delete(path),
        lambda: tree.move(path, "x"),
        lambda: tree.move("a/b.md", path),
        lambda: tree.move("a", path),
    ):
        with pytest.raises(DomainError) as error:
            operation()
        assert error.value.code == "invalid_path"
        assert str(error.value) == "Path must name a file or folder inside the tree"
    assert tree.read("a/b.md")[0] == "content"


def test_leading_dot_slash_resolves_relative_paths(tree) -> None:
    assert tree.mkdir("./a").path == "a"
    assert tree.write("./a/b.md", "content").path == "a/b.md"
    assert tree.read("./a/b.md") == tree.read("a/b.md")
    assert tree.list("./a") == tree.list("a")
    tree.edit("./a/b.md", "content", "updated")
    assert tree.read("a/b.md")[0] == "updated"
    assert tree.move("./a/b.md", "./a/c.md").path == "a/c.md"
    tree.delete("./a/c.md")
    assert tree.list("a") == ()


def test_overwrite_requires_matching_expected_hash(tree) -> None:
    tree.write("doc.md", "first")
    _, digest = tree.read("doc.md")
    with pytest.raises(DomainError) as missing:
        tree.write("doc.md", "second")
    assert missing.value.code == "expected_hash_required"
    with pytest.raises(ConflictError) as stale:
        tree.write("doc.md", "second", expected_hash="0" * 16)
    assert stale.value.actual == digest
    tree.write("doc.md", "second", expected_hash=digest)
    assert tree.read("doc.md") == ("second", content_hash("second"))


@pytest.mark.parametrize(
    "path",
    [
        "../escape.md",
        "/etc/passwd.md",
        "a/../../b.md",
        "./../escape.md",
        "a\\..\\b.md",
        "C:\\file.md",
        "bad\0.md",
    ],
)
def test_rejects_invalid_paths(tree, path) -> None:
    for operation in (lambda: tree.write(path, "nope"), lambda: tree.mkdir(path)):
        with pytest.raises(DomainError) as error:
            operation()
        assert error.value.code == "invalid_path"


@pytest.mark.parametrize("path", [".env", "./.hidden/file.txt", "a/.config/data.json"])
def test_hidden_paths_support_all_operations(tree, path) -> None:
    entry = tree.write(path, "before")
    assert tree.read(path)[0] == "before"
    tree.edit(path, "before", "after")
    assert tree.read(path)[0] == "after"
    assert entry.path in [item.path for item in tree.list()]
    assert tree.mkdir(".destination").kind == "directory"
    moved = tree.move(path, ".destination/.renamed")
    assert tree.list(".destination") == (moved,)
    tree.delete(moved.path)
    tree.delete(".destination")
    assert tree.list(".destination") == ()


def test_lists_hidden_paths_and_symlinks_without_recursing_into_links(
    tree, tmp_path
) -> None:
    require_symlinks(tmp_path)
    tree.write("real.md", "content")
    tree.write("visible/nested.txt", "nested")
    (tree.root / ".hidden").mkdir()
    (tree.root / ".hidden/note.md").write_text("hidden")
    (tree.root / ".file.md").write_text("hidden")
    (tree.root / "alias.md").symlink_to(tree.root / "real.md")
    (tree.root / "alias").symlink_to(tree.root / "visible", target_is_directory=True)
    (tree.root / "outside").symlink_to(tmp_path, target_is_directory=True)
    (tree.root / "dangling.md").symlink_to(tree.root / "missing.md")
    (tree.root / "loop").symlink_to(tree.root, target_is_directory=True)
    assert [(item.path, item.kind) for item in tree.list()] == [
        (".file.md", "file"),
        (".hidden", "directory"),
        (".hidden/note.md", "file"),
        ("alias", "directory"),
        ("alias.md", "file"),
        ("dangling.md", "file"),
        ("loop", "directory"),
        ("outside", "directory"),
        ("real.md", "file"),
        ("visible", "directory"),
        ("visible/nested.txt", "file"),
    ]
    assert tree.read("alias.md") == tree.read("real.md")
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("private")
    (tree.root / "external.txt").symlink_to(outside_file)
    for operation in (
        lambda: tree.list("outside"),
        lambda: tree.read("external.txt"),
        lambda: tree.write("outside/escape.md", "no"),
        lambda: tree.edit("external.txt", "private", "no"),
        lambda: tree.mkdir("outside/new"),
        lambda: tree.delete("external.txt"),
        lambda: tree.move("real.md", "outside/moved.md"),
        lambda: tree.move("external.txt", "moved.txt"),
    ):
        with pytest.raises(DomainError) as error:
            operation()
        assert error.value.code == "invalid_path"
    assert not (tmp_path / "escape.md").exists()
    assert outside_file.read_text() == "private"
    with pytest.raises(DomainError) as error:
        tree.move("real.md", "dangling.md")
    assert error.value.code == "already_exists"
    assert tree.read("real.md")[0] == "content"
    assert not (tree.root / "missing.md").exists()


@pytest.mark.parametrize("kind", ["file", "directory", "dangling"])
@pytest.mark.parametrize("operation", ["delete", "move"])
def test_link_mutations_leave_targets_untouched(
    tree, tmp_path, kind, operation
) -> None:
    require_symlinks(tmp_path)
    target = tree.root / "target"
    if kind == "file":
        tree.write("target", "keep")
    elif kind == "directory":
        tree.write("target/note.txt", "keep")
    link = tree.root / "alias"
    link.symlink_to("target", target_is_directory=kind == "directory")
    if operation == "delete":
        tree.delete("alias")
    else:
        entry = tree.move("alias", "renamed")
        assert entry.path == "renamed"
        assert (tree.root / "renamed").is_symlink()
        assert (tree.root / "renamed").readlink() == Path("target")
    assert not link.is_symlink()
    if kind == "file":
        assert target.read_text() == "keep"
    elif kind == "directory":
        assert (target / "note.txt").read_text() == "keep"
    else:
        assert not target.exists()


def test_non_utf8_files_are_listed_but_cannot_be_read_written_or_edited(tree) -> None:
    data = b"\xff\xfe"
    (tree.root / "unreadable.txt").write_bytes(data)
    (entry,) = tree.list()
    assert entry.kind == "file" and entry.size == len(data)
    for operation in (
        lambda: tree.read(entry.path),
        lambda: tree.write(entry.path, "replacement", expected_hash="stale"),
        lambda: tree.edit(entry.path, "x", "y"),
    ):
        with pytest.raises(DomainError) as error:
            operation()
        assert error.value.code == "not_readable"
    assert (tree.root / entry.path).read_bytes() == data


def test_edit_exact_replacements_and_errors(tree) -> None:
    tree.write("doc.txt", "alpha\nbeta\nbeta\n")
    for old, code in [("missing", "no_match"), ("beta", "ambiguous_match")]:
        with pytest.raises(DomainError) as error:
            tree.edit("doc.txt", old, "new")
        assert error.value.code == code
    entry, diff = tree.edit("doc.txt", "beta", "gamma", replace_all=True)
    assert tree.read(entry.path) == (
        "alpha\ngamma\ngamma\n",
        content_hash("alpha\ngamma\ngamma\n"),
    )
    assert "-beta" in diff and "+gamma" in diff
    tree.edit("doc.txt", "alpha", "first")
    with pytest.raises(DomainError) as error:
        tree.edit("missing.txt", "a", "b")
    assert error.value.code == "not_found"


@pytest.mark.parametrize("newline", [b"\r\n", b"\n"], ids=["crlf", "lf"])
def test_edit_preserves_line_endings(tree, newline) -> None:
    name = "doc.txt"
    original = newline.join([b"alpha", b"beta", b""])
    (tree.root / name).write_bytes(original)
    tree.edit(name, "beta", "gamma")
    assert (tree.root / name).read_bytes() == original.replace(b"beta", b"gamma")


def test_large_files_can_be_read_written_and_edited(tree) -> None:
    content = "记" * 400_000
    (tree.root / "large.txt").write_bytes(content.encode("utf-8"))
    assert tree.read("large.txt") == (content, content_hash(content))
    tree.write("written.txt", content)
    assert tree.read("written.txt") == (content, content_hash(content))
    tree.edit("written.txt", "记", "记忆", replace_all=True)
    updated = "记忆" * 400_000
    assert tree.read("written.txt") == (updated, content_hash(updated))
    tree.write("written.txt", content, expected_hash=content_hash(updated))
    assert tree.read("written.txt")[0] == content


def test_listing_reads_metadata_only(tree, monkeypatch) -> None:
    tree.write(".hidden/note.txt", "text")
    (tree.root / "binary").write_bytes(b"\xff")
    (tree.root / "large.txt").write_bytes(b"x" * 1_000_001)

    def fail(*args, **kwargs):
        pytest.fail("Listing must not read file contents")

    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr(DirectoryTree, "_content", fail)
    assert [entry.path for entry in tree.list()] == [
        ".hidden",
        ".hidden/note.txt",
        "binary",
        "large.txt",
    ]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are unavailable")
def test_lists_special_files_without_opening_them(tree) -> None:
    os.mkfifo(tree.root / "pipe")
    (entry,) = tree.list()
    assert entry.path == "pipe" and entry.kind == "file"
    with pytest.raises(DomainError) as error:
        tree.read("pipe")
    assert error.value.code == "not_found"


def test_mkdir_is_idempotent_and_file_directory_types_are_checked(tree) -> None:
    entry = tree.mkdir("a/b")
    assert tree.mkdir("a/b") == entry
    tree.write("file.txt", "content")
    for operation in (
        lambda: tree.mkdir("file.txt"),
        lambda: tree.write("a/b", "content"),
    ):
        with pytest.raises(DomainError) as error:
            operation()
        assert error.value.code == "invalid_path"
    for path in ("a/b", "missing.txt"):
        with pytest.raises(DomainError) as error:
            tree.read(path)
        assert error.value.code == "not_found"


def test_delete_keeps_empty_parents_and_removes_nonempty_directories(tree) -> None:
    tree.write("a/b/c.md", "deep")
    tree.delete("a/b/c.md")
    assert [item.path for item in tree.list()] == ["a", "a/b"]
    tree.write("a/b/c.txt", "deep")
    tree.delete("a")
    assert tree.list() == ()
    with pytest.raises(DomainError) as error:
        tree.delete("a")
    assert error.value.code == "not_found"


def test_move_files_and_directories_without_clobbering(tree) -> None:
    tree.write("old.md", "content")
    tree.write("taken.md", "other")
    assert tree.move("old.md", "folder/new.md").path == "folder/new.md"
    assert tree.move("folder", "parent/renamed").kind == "directory"
    assert tree.read("parent/renamed/new.md")[0] == "content"
    with pytest.raises(DomainError) as error:
        tree.move("parent/renamed/new.md", "taken.md")
    assert error.value.code == "already_exists"
    for destination in ("parent", "parent/child"):
        with pytest.raises(DomainError) as error:
            tree.move("parent", destination)
        assert error.value.code == "invalid_path"
    assert tree.read("taken.md")[0] == "other"


def test_writes_preserve_neighboring_temporary_names_and_exact_utf8_bytes(tree) -> None:
    tree.write("doc.txt.tmp", "keep")
    tree.write("doc.txt", "a\r\nb\n")
    assert tree.read("doc.txt") == ("a\r\nb\n", content_hash("a\r\nb\n"))
    assert tree.read("doc.txt.tmp")[0] == "keep"
    assert [item.path for item in tree.list()] == ["doc.txt", "doc.txt.tmp"]
