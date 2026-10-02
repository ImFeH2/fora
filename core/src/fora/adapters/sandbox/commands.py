from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path, PurePath

from fora.adapters.sandbox.paths import bind_order
from fora.core.errors import DomainError

BUBBLEWRAP_CANDIDATES = ("/usr/bin/bwrap", "/bin/bwrap")
SANDBOX_EXEC = "/usr/bin/sandbox-exec"


def bubblewrap_executable() -> str:
    for candidate in BUBBLEWRAP_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    found = shutil.which("bwrap")
    if found is None:
        raise DomainError(
            "sandbox_unavailable",
            "bubblewrap is required for filesystem write protection",
        )
    return str(Path(found).resolve())


def linux_command(
    argv: Sequence[str],
    cwd: Path | str,
    write_directories: Sequence[Path],
    *,
    bwrap: str | None = None,
    root: bool | None = None,
) -> list[str]:
    if root is None:
        root = hasattr(os, "geteuid") and os.geteuid() == 0
    command = [
        bwrap or bubblewrap_executable(),
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--unshare-user",
    ]
    for directory in bind_order(write_directories):
        rendered = directory.as_posix()
        command.extend(("--bind", rendered, rendered))
    working = cwd.as_posix() if isinstance(cwd, PurePath) else str(cwd)
    command.extend(("--chdir", working, "--cap-drop", "ALL"))
    if root:
        command.extend(("--cap-add", "CAP_SETFCAP"))
    command.extend(("--", *argv))
    return command


def macos_profile(count: int) -> str:
    if count:
        exclusions = "\n".join(
            f'    (require-not (subpath (param "WRITABLE_{index}")))'
            for index in range(count)
        )
        write_rule = (
            "(deny file-write*\n"
            "  (require-all\n"
            '    (require-not (literal "/dev/null"))\n'
            f"{exclusions}))"
        )
        unlink_rules = "\n".join(
            "(deny file-write-unlink\n"
            "  (require-all\n"
            f'    (literal (param "WRITABLE_{index}"))\n'
            "    (vnode-type DIRECTORY)))"
            for index in range(count)
        )
    else:
        write_rule = '(deny file-write* (require-not (literal "/dev/null")))'
        unlink_rules = ""
    parts = ["(version 1)", "(allow default)", write_rule, unlink_rules]
    return "\n".join(part for part in parts if part)


def macos_command(argv: Sequence[str], write_directories: Sequence[Path]) -> list[str]:
    roots = [root for root in write_directories if root.is_dir()]
    parameters = [
        f"-DWRITABLE_{index}={root.as_posix()}" for index, root in enumerate(roots)
    ]
    return [SANDBOX_EXEC, "-p", macos_profile(len(roots)), *parameters, "--", *argv]


def windows_command(
    sid: str, argv: Sequence[str], entrypoint: Sequence[str]
) -> list[str]:
    return [*entrypoint, "--windows-write-sandbox", sid, "--", *argv]
