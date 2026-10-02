from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from fora.core.errors import DomainError


@dataclass(frozen=True)
class NormalizedDirectories:
    accepted: tuple[Path, ...]
    skipped: tuple[tuple[str, str], ...]


def _normalize_one(value: object, *, require_existing: bool) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise DomainError(
            "invalid_directory", "Writable directories must be non-empty strings"
        )
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        if PurePosixPath(value).is_absolute():
            raise DomainError(
                "foreign_directory",
                f"{value} is a POSIX path and cannot be used on this platform",
            )
        raise DomainError(
            "invalid_directory", "Writable directories must be absolute paths"
        )
    try:
        resolved = candidate.resolve(strict=require_existing)
    except (OSError, RuntimeError, ValueError) as error:
        raise DomainError(
            "invalid_directory", f"Cannot resolve writable directory: {error}"
        ) from error
    if require_existing and not resolved.is_dir():
        raise DomainError(
            "invalid_directory", "Writable directories must be existing directories"
        )
    return resolved


def normalize_directories(
    values: Sequence[str], *, require_existing: bool = False
) -> tuple[Path, ...]:
    normalized: list[Path] = []
    seen: set[str] = set()
    for value in values:
        resolved = _normalize_one(value, require_existing=require_existing)
        identity = os.path.normcase(str(resolved))
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(resolved)
    return tuple(normalized)


def normalize_tolerantly(values: Sequence[str]) -> NormalizedDirectories:
    accepted: list[Path] = []
    skipped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        try:
            resolved = _normalize_one(value, require_existing=False)
        except DomainError as error:
            skipped.append((str(value), error.code))
            continue
        identity = os.path.normcase(str(resolved))
        if identity in seen:
            continue
        seen.add(identity)
        accepted.append(resolved)
    return NormalizedDirectories(tuple(accepted), tuple(skipped))


def is_within(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def bind_order(roots: Sequence[Path]) -> tuple[Path, ...]:
    return tuple(sorted(roots, key=lambda item: len(item.parts)))
