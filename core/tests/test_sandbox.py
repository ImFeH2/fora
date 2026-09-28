from __future__ import annotations

import shutil
import sys
from pathlib import Path, PurePosixPath

import pytest

from huddol.adapters.execution.editing import edit_file
from huddol.adapters.execution.local import LocalExecution
from huddol.adapters.sandbox.commands import (
    linux_command,
    macos_command,
    macos_profile,
    windows_command,
)
from huddol.adapters.sandbox.paths import bind_order, is_within, normalize_directories
from huddol.core.errors import DomainError

LINUX_ONLY = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="linux sandbox"
)


def test_directories_are_absolute_resolved_and_deduplicated(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    roots = normalize_directories([str(tmp_path / "a"), str(tmp_path / "a") + "/"])
    assert roots == (tmp_path.resolve() / "a",)

    with pytest.raises(DomainError) as error:
        normalize_directories(["relative/path"])
    assert error.value.code == "invalid_directory"


def test_missing_directories_are_allowed_when_configuring(tmp_path: Path) -> None:
    roots = normalize_directories([str(tmp_path / "later")])
    assert roots[0].name == "later"
    with pytest.raises(DomainError):
        normalize_directories([str(tmp_path / "later")], require_existing=True)


def test_containment_covers_the_root_itself_and_children(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    assert is_within(root, [root])
    assert is_within(root / "deep" / "file.txt", [root])
    assert not is_within(tmp_path.parent, [root])


def test_bind_order_puts_shallow_paths_first() -> None:
    roots = [Path("/a/b/c"), Path("/a"), Path("/a/b")]
    assert bind_order(roots) == (Path("/a"), Path("/a/b"), Path("/a/b/c"))


@pytest.mark.parametrize("root", [False, True])
def test_linux_command_carries_the_required_isolation_flags(root: bool) -> None:
    command = linux_command(
        ["ls"],
        "/work",
        [PurePosixPath("/w/deep/nested"), PurePosixPath("/w")],
        bwrap="/usr/bin/bwrap",
        root=root,
    )
    assert command[:1] == ["/usr/bin/bwrap"]
    for flag in ("--die-with-parent", "--unshare-user"):
        assert flag in command
    assert command[command.index("--ro-bind") + 1 : command.index("--ro-bind") + 3] == [
        "/",
        "/",
    ]
    assert command.index("--ro-bind") < command.index("--proc")
    assert command[command.index("--proc") + 1] == "/proc"
    kept = ["--cap-add", "CAP_SETFCAP"] if root else []
    assert command[-4 - len(kept) :] == ["--cap-drop", "ALL", *kept, "--", "ls"]
    shallow = command.index("/w")
    deep = command.index("/w/deep/nested")
    assert shallow < deep


def test_macos_profile_keeps_dev_null_writable_and_protects_root_directories() -> None:
    profile = macos_profile(2)
    assert '(require-not (literal "/dev/null"))' in profile
    assert profile.count("file-write-unlink") == 2
    assert "(vnode-type DIRECTORY)" in profile
    assert macos_profile(0).count("file-write-unlink") == 0


def test_macos_command_passes_roots_as_parameters(tmp_path: Path) -> None:
    command = macos_command(["ls"], [tmp_path])
    assert command[0] == "/usr/bin/sandbox-exec"
    assert f"-DWRITABLE_0={tmp_path.as_posix()}" in command
    assert command[-2:] == ["--", "ls"]


def test_windows_command_reenters_the_kernel_entrypoint() -> None:
    command = windows_command(
        "S-1-5-21-1-2-3",
        ["cmd", "/c", "echo"],
        [sys.executable, "-I", "-m", "huddol"],
    )
    assert command[:4] == [sys.executable, "-I", "-m", "huddol"]
    assert "--windows-write-sandbox" in command
    assert command[command.index("--windows-write-sandbox") + 1] == "S-1-5-21-1-2-3"
    assert command[-3:] == ["--", "cmd", "/c"] or command[-4:] == [
        "--",
        "cmd",
        "/c",
        "echo",
    ]


def test_edit_replaces_once_and_reports_a_diff(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    sandbox = LocalExecution([str(tmp_path)], enforce=False)
    result = sandbox.edit(str(target), [{"old_text": "beta", "new_text": "gamma"}])
    assert target.read_text(encoding="utf-8") == "alpha\ngamma\n"
    assert result.replacements == 1
    assert "-beta" in result.diff and "+gamma" in result.diff


def test_edit_applies_multiple_replacements_from_the_same_original(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    original = "left: one\nmiddle: two\n标题: 旧值\r\n"
    first.write_bytes(original.encode("utf-8"))
    second.write_bytes(original.encode("utf-8"))
    edits = [
        {"old_text": "one", "new_text": "two"},
        {"old_text": "two", "new_text": "done"},
        {"old_text": "标题: 旧值", "new_text": "标题: 新值\n第二行"},
    ]
    first_result = LocalExecution([str(tmp_path)], enforce=False).edit(
        str(first), edits
    )
    second_result = LocalExecution([str(tmp_path)], enforce=False).edit(
        str(second), list(reversed(edits))
    )
    expected = "left: two\nmiddle: done\n标题: 新值\n第二行\r\n"
    assert first.read_bytes() == expected.encode("utf-8")
    assert second.read_bytes() == expected.encode("utf-8")
    assert first_result.replacements == second_result.replacements == 3
    assert "-left: one" in first_result.diff
    assert "+left: two" in first_result.diff


def test_edit_rejects_unmatched_ambiguous_and_overlapping_items_atomically(
    tmp_path: Path,
) -> None:
    cases = [
        (
            "unmatched.txt",
            "before",
            [{"old_text": "missing", "new_text": "after"}],
            "no_match",
            "edit 1",
        ),
        (
            "ambiguous.txt",
            "x\nx\n",
            [{"old_text": "x", "new_text": "y"}],
            "ambiguous_match",
            "edit 1",
        ),
        (
            "overlapping.txt",
            "abcdef",
            [
                {"old_text": "abc", "new_text": "A"},
                {"old_text": "bcd", "new_text": "B"},
            ],
            "overlapping_edits",
            "edit 2",
        ),
        (
            "replace-all-overlap.txt",
            "abcdef",
            [
                {"old_text": "abc", "new_text": "A", "replace_all": True},
                {"old_text": "cde", "new_text": "C"},
            ],
            "overlapping_edits",
            "edit 2",
        ),
    ]
    for name, original, edits, code, message in cases:
        target = tmp_path / name
        target.write_bytes(original.encode("utf-8"))
        with pytest.raises(DomainError, match=message) as error:
            LocalExecution([str(tmp_path)], enforce=False).edit(str(target), edits)
        assert error.value.code == code
        assert target.read_bytes() == original.encode("utf-8")


def test_edit_create_requires_one_complete_edit(tmp_path: Path) -> None:
    target = tmp_path / "created.txt"
    with pytest.raises(DomainError, match="create requires exactly one edit") as error:
        LocalExecution([str(tmp_path)], enforce=False).edit(
            str(target),
            [
                {"old_text": "", "new_text": "one"},
                {"old_text": "", "new_text": "two"},
            ],
            create=True,
        )
    assert error.value.code == "invalid_edit"
    assert not target.exists()


def test_edit_creates_complete_file_and_allows_empty_content(tmp_path: Path) -> None:
    sandbox = LocalExecution([str(tmp_path)], enforce=False)
    target = tmp_path / "new.txt"
    result = sandbox.edit(
        str(target), [{"old_text": "", "new_text": "alpha\nbeta\n"}], create=True
    )
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert result.path == str(target)
    assert result.replacements == 0
    assert "+alpha" in result.diff and "+beta" in result.diff
    empty = tmp_path / "empty.txt"
    assert (
        sandbox.edit(
            str(empty), [{"old_text": "", "new_text": ""}], create=True
        ).replacements
        == 0
    )
    assert empty.read_bytes() == b""


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling_link"])
def test_edit_create_protects_existing_paths(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "target"
    if kind == "file":
        target.write_text("preserved", encoding="utf-8")
    elif kind == "directory":
        target.mkdir()
    elif kind == "symlink":
        linked = tmp_path / "linked.txt"
        linked.write_text("preserved", encoding="utf-8")
        try:
            target.symlink_to(linked)
        except OSError as error:
            pytest.skip(f"symlinks are unavailable: {error}")
    else:
        try:
            target.symlink_to(tmp_path / "missing")
        except OSError as error:
            pytest.skip(f"symlinks are unavailable: {error}")
    sandbox = LocalExecution([str(tmp_path)], enforce=False)
    with pytest.raises(DomainError) as error:
        sandbox.edit(
            str(target), [{"old_text": "", "new_text": "replacement"}], create=True
        )
    assert error.value.code == "already_exists"
    if kind == "file":
        assert target.read_text(encoding="utf-8") == "preserved"
    elif kind == "directory":
        assert target.is_dir()
    else:
        assert target.is_symlink()
        if kind == "symlink":
            assert linked.read_text(encoding="utf-8") == "preserved"


def test_edit_create_requires_existing_parent_and_writable_root(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    sandbox = LocalExecution([str(allowed)], enforce=False)
    missing = allowed / "missing" / "file.txt"
    with pytest.raises(DomainError) as error:
        sandbox.edit(str(missing), [{"old_text": "", "new_text": "body"}], create=True)
    assert error.value.code == "not_found"
    outside = tmp_path / "outside.txt"
    with pytest.raises(DomainError) as error:
        sandbox.edit(str(outside), [{"old_text": "", "new_text": "body"}], create=True)
    assert error.value.code == "not_writable"
    assert not outside.exists()


def test_edit_create_rejects_parent_symlink_outside_writable_roots(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    alias = allowed / "alias"
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    sandbox = LocalExecution([str(allowed)], enforce=False)
    with pytest.raises(DomainError) as error:
        sandbox.edit(
            str(alias / "file.txt"), [{"old_text": "", "new_text": "body"}], create=True
        )
    assert error.value.code == "not_writable"
    assert not (outside / "file.txt").exists()


def test_edit_create_rejects_incompatible_parameters(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    with pytest.raises(DomainError) as error:
        edit_file(
            str(target),
            [{"old_text": "old", "new_text": "body"}],
            directories=[str(tmp_path)],
            create=True,
        )
    assert error.value.code == "invalid_edit"
    with pytest.raises(DomainError) as error:
        edit_file(
            str(target),
            [{"old_text": "", "new_text": "body", "replace_all": True}],
            directories=[str(tmp_path)],
            create=True,
        )
    assert error.value.code == "invalid_edit"
    assert not target.exists()


def test_edit_refuses_ambiguous_matches_unless_replace_all(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x\nx\n", encoding="utf-8")
    sandbox = LocalExecution([str(tmp_path)], enforce=False)
    with pytest.raises(DomainError) as error:
        sandbox.edit(str(target), [{"old_text": "x", "new_text": "y"}])
    assert error.value.code == "ambiguous_match"
    assert (
        sandbox.edit(
            str(target), [{"old_text": "x", "new_text": "y", "replace_all": True}]
        ).replacements
        == 2
    )


@pytest.mark.parametrize(
    ("replace_all", "code"),
    [(False, "ambiguous_match"), (True, "overlapping_edits")],
)
def test_edit_rejects_overlapping_matches_from_one_item_atomically(
    tmp_path: Path, replace_all: bool, code: str
) -> None:
    target = tmp_path / "file.txt"
    original = b"aaaa"
    target.write_bytes(original)
    item = {"old_text": "aaa", "new_text": "X"}
    if replace_all:
        item["replace_all"] = True
    with pytest.raises(DomainError) as error:
        LocalExecution([str(tmp_path)], enforce=False).edit(str(target), [item])
    assert error.value.code == code
    assert target.read_bytes() == original


def test_edit_rejects_paths_outside_the_writable_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    sandbox = LocalExecution([str(allowed)])
    with pytest.raises(DomainError) as error:
        sandbox.edit(str(outside), [{"old_text": "secret", "new_text": "leaked"}])
    assert error.value.code == "not_writable"
    assert outside.read_text(encoding="utf-8") == "secret"


def test_edit_leaves_no_temporary_files(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("a", encoding="utf-8")
    sandbox = LocalExecution([str(tmp_path)], enforce=False)
    sandbox.edit(str(target), [{"old_text": "a", "new_text": "b"}])
    assert list(tmp_path.glob("*.huddol-tmp")) == []


def test_run_rejects_malformed_argv(tmp_path: Path) -> None:
    sandbox = LocalExecution(enforce=False)
    with pytest.raises(DomainError):
        sandbox.run([], cwd=str(tmp_path))


@pytest.mark.parametrize("writable", [False, True])
def test_describe_environment_names_the_writable_roots(
    tmp_path: Path, writable: bool
) -> None:
    directories = [str(tmp_path)] if writable else []
    sandbox = LocalExecution(directories, enforce=False)
    listing = f"- {tmp_path.resolve()}" if writable else "- none"
    assert sandbox.describe_environment() == (
        f"Commands run on {sys.platform}\nWritable directories:\n{listing}"
    )


@LINUX_ONLY
def test_unsandboxed_run_executes_and_captures_output(tmp_path: Path) -> None:
    sandbox = LocalExecution(enforce=False)
    result = sandbox.run(["echo", "hello"], cwd=str(tmp_path))
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"


@LINUX_ONLY
@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is not installed")
def test_sandboxed_run_allows_writes_inside_and_blocks_them_outside(
    tmp_path: Path,
) -> None:
    writable = tmp_path / "writable"
    writable.mkdir()
    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / "keep.txt").write_text("original", encoding="utf-8")

    sandbox = LocalExecution([str(writable)])

    inside = sandbox.run(
        ["sh", "-c", f"echo ok > {writable}/probe.txt"], cwd=str(tmp_path)
    )
    assert inside.exit_code == 0
    assert (writable / "probe.txt").read_text(encoding="utf-8").strip() == "ok"

    outside = sandbox.run(
        ["sh", "-c", f"echo hacked > {protected}/keep.txt"], cwd=str(tmp_path)
    )
    assert outside.exit_code != 0
    assert "Read-only file system" in outside.stderr
    assert (protected / "keep.txt").read_text(encoding="utf-8") == "original"


def test_posix_paths_are_reported_as_foreign_not_merely_invalid() -> None:
    from huddol.adapters.sandbox.paths import normalize_tolerantly

    result = normalize_tolerantly(["/workspace/app", "relative/thing"])
    reasons = {path: reason for path, reason in result.skipped}
    if sys.platform.startswith("win"):
        assert reasons["/workspace/app"] == "foreign_directory"
    else:
        assert "/workspace/app" not in reasons
    assert reasons["relative/thing"] == "invalid_directory"


def test_tolerant_mode_keeps_the_usable_directories(tmp_path: Path) -> None:
    from huddol.adapters.sandbox.paths import normalize_tolerantly

    usable = tmp_path / "workspace"
    usable.mkdir()
    result = normalize_tolerantly([str(usable), "not-absolute"])
    assert result.accepted == (usable.resolve(),)
    assert [path for path, _ in result.skipped] == ["not-absolute"]


def test_a_sandbox_with_unusable_configuration_still_constructs(tmp_path: Path) -> None:
    sandbox = LocalExecution(["not-absolute"], enforce=False, tolerant=True)
    assert sandbox.write_directories == ()
    assert sandbox.skipped == (("not-absolute", "invalid_directory"),)


def test_strict_mode_still_rejects_bad_input(tmp_path: Path) -> None:
    with pytest.raises(DomainError):
        LocalExecution(["not-absolute"], enforce=False)


@pytest.mark.parametrize("cwd", [None, "", "relative/path"])
def test_run_rejects_missing_or_relative_cwd(cwd) -> None:
    environment = LocalExecution(enforce=False)
    with pytest.raises(DomainError, match="^cwd must be an absolute path$") as error:
        environment.run([sys.executable, "-c", "pass"], cwd=cwd)
    assert error.value.code == "invalid_cwd"


def test_run_rejects_an_omitted_cwd() -> None:
    environment = LocalExecution(enforce=False)
    with pytest.raises(DomainError, match="^cwd must be an absolute path$") as error:
        environment.run([sys.executable, "-c", "pass"])
    assert error.value.code == "invalid_cwd"


@pytest.mark.parametrize("exists", [False, True])
def test_run_rejects_a_cwd_that_is_not_a_directory(
    tmp_path: Path, exists: bool
) -> None:
    target = tmp_path / "file"
    if exists:
        target.touch()
    environment = LocalExecution(enforce=False)
    with pytest.raises(DomainError) as error:
        environment.run([sys.executable, "-c", "pass"], cwd=str(target))
    assert error.value.code == "invalid_cwd"


def test_edit_file_rejects_relative_paths(tmp_path: Path) -> None:
    with pytest.raises(DomainError, match="^path must be an absolute path$") as error:
        edit_file(
            "file.txt",
            [{"old_text": "before", "new_text": "after"}],
            directories=[str(tmp_path)],
        )
    assert error.value.code == "invalid_path"


def test_linux_command_renders_posix_paths_on_every_host() -> None:
    command = linux_command(
        ["ls"],
        PurePosixPath("/work"),
        [PurePosixPath("/w/deep"), PurePosixPath("/w")],
        bwrap="/usr/bin/bwrap",
    )
    assert "--chdir" in command
    assert command[command.index("--chdir") + 1] == "/work"
    for rendered in ("/w", "/w/deep"):
        assert rendered in command
        assert "\\" not in rendered
    assert command.index("/w") < command.index("/w/deep")
