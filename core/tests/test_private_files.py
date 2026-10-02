from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from fora.__main__ import load_token
from fora.adapters.host import read_private, write_private

WINDOWS = pytest.mark.skipif(sys.platform != "win32", reason="Windows ACL")


def powershell(source: str) -> str:
    powershell_exe = (
        Path(os.environ["SystemRoot"])
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe"
    )
    environment = os.environ.copy()
    environment["PSModulePath"] = str(powershell_exe.parent / "Modules")
    result = subprocess.run(
        [
            str(powershell_exe),
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$ErrorActionPreference='Stop'; " + source,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def quoted(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def test_private_file_creation_and_replacement(tmp_path):
    path = tmp_path / "private"
    for payload in ("test-value", "replacement-中文"):
        write_private(path, payload)
        assert read_private(path) == payload
        if os.name != "nt":
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert list(tmp_path.iterdir()) == [path]


@WINDOWS
def test_private_acl_survives_broad_parent_and_replacement(tmp_path):
    from fora.adapters.windows import current_user_sid

    sid = current_user_sid()
    powershell(
        f"$acl=Get-Acl -LiteralPath {quoted(tmp_path)}; "
        f"$acl.SetSecurityDescriptorSddlForm('D:P(A;OICI;FA;;;{sid})(A;OICI;FA;;;BU)', 'Access'); "
        f"[System.IO.Directory]::SetAccessControl({quoted(tmp_path)}, $acl)"
    )
    path = tmp_path / "private"
    path.write_text("inherited-test-value", encoding="utf-8")
    with pytest.raises(PermissionError):
        read_private(path)
    for payload in ("private-test-value", "replaced-test-value"):
        write_private(path, payload)
        assert read_private(path) == payload
        acl = json.loads(
            powershell(
                f"$acl=Get-Acl -LiteralPath {quoted(path)}; "
                "$owner=$acl.GetOwner([System.Security.Principal.SecurityIdentifier]).Value; "
                "$access=@($acl.Access | ForEach-Object { "
                "[ordered]@{"
                "IdentitySid=$_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value; "
                "AccessType=$_.AccessControlType.ToString(); "
                "Rights=[int]$_.FileSystemRights; "
                "IsInherited=$_.IsInherited; "
                "InheritanceFlags=[int]$_.InheritanceFlags; "
                "PropagationFlags=[int]$_.PropagationFlags"
                "}"
                "}); "
                "[ordered]@{"
                "OwnerSid=$owner; "
                "AreAccessRulesProtected=$acl.AreAccessRulesProtected; "
                "Access=$access"
                "} | ConvertTo-Json -Compress -Depth 4"
            )
        )
        assert acl == {
            "OwnerSid": sid,
            "AreAccessRulesProtected": True,
            "Access": [
                {
                    "IdentitySid": sid,
                    "AccessType": "Allow",
                    "Rights": 0x1F01FF,
                    "IsInherited": False,
                    "InheritanceFlags": 0,
                    "PropagationFlags": 0,
                }
            ],
        }
        with path.open("a", encoding="utf-8") as stream:
            stream.write("-writable")
        assert read_private(path) == payload + "-writable"
        assert list(tmp_path.iterdir()) == [path]


@WINDOWS
@pytest.mark.parametrize("rule", ["D:NO_ACCESS_CONTROL", "D:P(A;;FA;;;WD)"])
def test_unsafe_existing_token_is_rejected_without_modification(tmp_path, rule):
    from fora.adapters.windows import current_user_sid

    path = tmp_path / "token"
    path.write_text("test-token-value", encoding="utf-8")
    sid = current_user_sid()
    powershell(
        f"$acl=Get-Acl -LiteralPath {quoted(path)}; "
        f"$acl.SetSecurityDescriptorSddlForm('O:{sid}{rule}', 'Owner,Access'); "
        f"Set-Acl -LiteralPath {quoted(path)} -AclObject $acl"
    )
    with pytest.raises(PermissionError):
        load_token(tmp_path, None)
    assert path.read_text(encoding="utf-8") == "test-token-value"


@WINDOWS
def test_replace_failure_preserves_original_and_removes_temporary(
    tmp_path, monkeypatch
):
    from fora.adapters import windows_files

    path = tmp_path / "private"
    write_private(path, "original-test-value")

    def fail(*args):
        raise PermissionError("replace denied")

    monkeypatch.setattr(windows_files.os, "replace", fail)
    with pytest.raises(PermissionError, match="replace denied"):
        write_private(path, "replacement-test-value")
    assert read_private(path) == "original-test-value"
    assert list(tmp_path.iterdir()) == [path]


@WINDOWS
def test_creation_failure_does_not_remove_existing_file(tmp_path, monkeypatch):
    from fora.adapters import windows_files

    path = tmp_path / "private"
    occupied = tmp_path / ".private.fixed.fora-tmp"
    occupied.write_text("existing-test-value", encoding="utf-8")
    monkeypatch.setattr(windows_files.secrets, "token_hex", lambda count: "fixed")
    with pytest.raises(FileExistsError):
        write_private(path, "test-value")
    assert occupied.read_text(encoding="utf-8") == "existing-test-value"
    assert not path.exists()


@WINDOWS
def test_private_token_can_be_reloaded(tmp_path):
    write_private(tmp_path / "token", "test-token-value")
    assert load_token(tmp_path, None) == "test-token-value"


@WINDOWS
def test_restricted_subject_cannot_modify_private_file(tmp_path):
    from fora.adapters.execution.local import LocalExecution

    path = tmp_path / "private"
    write_private(path, "test-private-content")
    environment = LocalExecution([str(tmp_path)], enforce=True)
    try:
        result = environment.run(
            [
                sys._base_executable,
                "-c",
                f"from pathlib import Path; Path({str(path)!r}).write_text('changed')",
            ],
            cwd=str(tmp_path),
        )
        assert result.exit_code != 0
        assert "PermissionError" in result.stderr
        assert read_private(path) == "test-private-content"
    finally:
        environment.close()


@WINDOWS
def test_read_validation_and_reading_keep_the_same_file(tmp_path, monkeypatch):
    from fora.adapters import windows_files

    path = tmp_path / "private"
    replacement = tmp_path / "replacement"
    write_private(path, "original-test-content")
    write_private(replacement, "replacement-test-content")
    validate = windows_files._validate_private
    handles = []

    def validated(handle, target):
        handles.append(handle)
        validate(handle, target)
        with pytest.raises(PermissionError):
            os.replace(replacement, path)

    monkeypatch.setattr(windows_files, "_validate_private", validated)
    assert read_private(path) == "original-test-content"
    assert len(handles) == 1
    assert replacement.exists()


@WINDOWS
def test_encoding_failure_releases_handle_and_temporary_file(tmp_path):
    path = tmp_path / "private"
    write_private(path, "original-test-content")
    with pytest.raises(UnicodeEncodeError):
        write_private(path, "\ud800")
    assert read_private(path) == "original-test-content"
    assert list(tmp_path.iterdir()) == [path]
