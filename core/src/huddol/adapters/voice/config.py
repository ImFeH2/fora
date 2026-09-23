from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

from huddol.core.errors import DomainError


@dataclass(frozen=True)
class VoiceConfig:
    mode: Literal["local", "remote"] = "local"
    address: str = "wss://api.openai.com/v1/realtime?intent=transcription"
    model: str = ""
    api_key: str = field(default="", repr=False)

    @classmethod
    def restore(cls, values: dict[str, Any] | None) -> VoiceConfig:
        values = values or {}
        mode = values.get("mode", "local")
        if mode not in ("local", "remote"):
            raise DomainError("voice_config", "Select local or remote transcription")
        address = values.get("address", cls.address)
        model = values.get("model", "")
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
        if mode == "remote" and (not model.strip() or not api_key.strip()):
            raise DomainError(
                "voice_config", "Remote transcription requires a model and API key"
            )
        if any(character in api_key for character in "\r\n"):
            raise DomainError(
                "voice_config", "API key contains an invalid header character"
            )
        return cls(mode=mode, address=address, model=model.strip(), api_key=api_key)

    def public(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "address": self.address,
            "model": self.model,
            "api_key_set": bool(self.api_key),
        }
