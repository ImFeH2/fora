from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from pathlib import Path

from fora.adapters.file_writes import (
    create_file_exclusive,
    directory_lock,
    replace_file,
)
from fora.adapters.sandbox.paths import is_within
from fora.core.errors import DomainError
from fora.ports.execution import EditResult


def _edit_error(code: str, index: int, message: str) -> DomainError:
    return DomainError(code, f"edit {index}: {message}")


def _parse_edits(edits: object, *, create: bool) -> list[tuple[str, str, bool]]:
    if not isinstance(edits, Sequence) or isinstance(edits, (str, bytes, bytearray)):
        raise DomainError("invalid_edit", "edits must be a non-empty array")
    if not edits:
        raise DomainError("invalid_edit", "edits must be a non-empty array")
    parsed: list[tuple[str, str, bool]] = []
    for index, item in enumerate(edits, 1):
        if not isinstance(item, Mapping):
            raise _edit_error("invalid_edit", index, "edit must be an object")
        unknown = set(item) - {"old_text", "new_text", "replace_all"}
        if unknown:
            raise _edit_error(
                "invalid_edit", index, f"unknown fields: {sorted(unknown)}"
            )
        missing = [name for name in ("old_text", "new_text") if name not in item]
        if missing:
            raise _edit_error(
                "invalid_edit", index, f"missing fields: {', '.join(missing)}"
            )
        old_text = item["old_text"]
        new_text = item["new_text"]
        replace_all = item.get("replace_all", False)
        if not isinstance(old_text, str):
            raise _edit_error("invalid_edit", index, "old_text must be a string")
        if not isinstance(new_text, str):
            raise _edit_error("invalid_edit", index, "new_text must be a string")
        if type(replace_all) is not bool:
            raise _edit_error("invalid_edit", index, "replace_all must be a boolean")
        if create:
            if old_text != "":
                raise _edit_error(
                    "invalid_edit", index, "old_text must be empty when creating"
                )
            if replace_all:
                raise _edit_error(
                    "invalid_edit",
                    index,
                    "replace_all cannot be used when creating",
                )
        elif not old_text:
            raise _edit_error(
                "invalid_edit", index, "old_text must be a non-empty string"
            )
        parsed.append((old_text, new_text, replace_all))
    if create and len(parsed) != 1:
        raise DomainError("invalid_edit", "create requires exactly one edit")
    return parsed


def edit_file(
    path: str,
    edits: Sequence[Mapping[str, object]],
    *,
    directories: list[str],
    create: bool = False,
) -> EditResult:
    if type(create) is not bool:
        raise DomainError("invalid_edit", "create must be a boolean")
    parsed = _parse_edits(edits, create=create)
    candidate = Path(path)
    if not candidate.is_absolute():
        raise DomainError("invalid_path", "path must be an absolute path")
    if create and not candidate.name:
        raise DomainError("invalid_path", "path must name a file")
    if create:
        parent = candidate.parent.resolve()
        target = parent / candidate.name
    else:
        target = candidate.resolve()
    if not is_within(target, [Path(item).resolve() for item in directories]):
        raise DomainError(
            "not_writable", "Path is outside the configured writable directories"
        )
    if not target.parent.is_dir():
        raise DomainError("not_found", f"{path} does not exist")
    with directory_lock(target):
        if create:
            content = parsed[0][1]
            with create_file_exclusive(target, content, prefix=".fora-create-"):
                return EditResult(
                    str(target),
                    "".join(
                        difflib.unified_diff(
                            [],
                            content.splitlines(keepends=True),
                            fromfile=str(target),
                            tofile=str(target),
                            n=3,
                        )
                    ),
                    0,
                )
        if not target.is_file():
            raise DomainError("not_found", f"{path} does not exist")
        try:
            original = target.read_bytes().decode("utf-8")
        except UnicodeDecodeError as error:
            raise DomainError("not_text", f"{path} is not valid UTF-8") from error
        matches: list[tuple[int, int, str, int]] = []
        for index, (old_text, new_text, replace_all) in enumerate(parsed, 1):
            occurrences: list[tuple[int, int]] = []
            start = 0
            while True:
                found = original.find(old_text, start)
                if found < 0:
                    break
                occurrences.append((found, found + len(old_text)))
                start = found + 1
            if not occurrences:
                raise _edit_error(
                    "no_match", index, "old_text does not appear in the file"
                )
            if len(occurrences) > 1 and not replace_all:
                raise _edit_error(
                    "ambiguous_match",
                    index,
                    "old_text appears multiple times; pass replace_all to change all",
                )
            matches.extend((start, end, new_text, index) for start, end in occurrences)
        ordered = sorted(matches, key=lambda item: (item[0], item[1], item[3]))
        _, active_end, _, active_index = ordered[0]
        for start, end, _, index in ordered[1:]:
            if start < active_end:
                raise DomainError(
                    "overlapping_edits",
                    f"edit {index} overlaps edit {active_index} "
                    f"at characters {start}-{min(end, active_end)}",
                )
            if end > active_end:
                active_end, active_index = end, index
        pieces: list[str] = []
        cursor = 0
        for start, end, new_text, _ in ordered:
            pieces.extend((original[cursor:start], new_text))
            cursor = end
        pieces.append(original[cursor:])
        updated = "".join(pieces)
        with replace_file(
            target, updated, mode=target.stat().st_mode, prefix=".fora-edit-"
        ):
            return EditResult(
                str(target),
                "".join(
                    difflib.unified_diff(
                        original.splitlines(keepends=True),
                        updated.splitlines(keepends=True),
                        fromfile=str(target),
                        tofile=str(target),
                        n=3,
                    )
                ),
                len(matches),
            )
