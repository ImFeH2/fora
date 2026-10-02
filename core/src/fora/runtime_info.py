from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class RuntimeInfo:
    version: str
    started_at: str

    @classmethod
    def capture(cls) -> RuntimeInfo:
        package_version = importlib.metadata.version("fora")
        if not package_version:
            raise RuntimeError("Fora package version is missing")
        return cls(
            version=package_version,
            started_at=datetime.now(UTC).isoformat(),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "version": self.version,
            "started_at": self.started_at,
        }
