from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from huddol.core.errors import DomainError


@dataclass(frozen=True)
class VoiceConfig:
    address: str = "wss://api.openai.com/v1/realtime?intent=transcription"
    model: str = "gpt-live-transcribe"
    api_key: str = field(default="", repr=False)

    @classmethod
    def restore(cls, values: dict[str, Any] | None) -> VoiceConfig:
        values = values or {}
        address = values.get("address", cls.address)
        model = values.get("model", cls.model)
        api_key = values.get("api_key", "")
        if not all(isinstance(value, str) for value in (address, model, api_key)):
            raise DomainError("voice_config", "Voice settings must contain text values")
        parsed = urlsplit(address)
        if (
            parsed.scheme != "wss"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise DomainError(
                "voice_config", "Use a WSS service address without credentials"
            )
        if not model.strip():
            model = cls.model
        if any(character in api_key for character in "\r\n"):
            raise DomainError(
                "voice_config", "API key contains an invalid header character"
            )
        return cls(address=address, model=model.strip(), api_key=api_key)

    def public(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "model": self.model,
            "api_key_set": bool(self.api_key),
        }
