from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from fora.core.errors import DomainError

Decision = Literal["allow", "deny"]


@dataclass(frozen=True)
class Actor:
    member_id: int
    is_agent: bool


AuthorizePolicy = Callable[[Actor, str, object], Decision]


def allow_everything(actor: Actor, capability: str, target: object) -> Decision:
    del actor, capability, target
    return "allow"


class Authorizer:
    def __init__(self, policy: AuthorizePolicy | None = None) -> None:
        self._policy = policy or allow_everything

    def check(self, actor: Actor, capability: str, target: object = None) -> None:
        if self._policy(actor, capability, target) != "allow":
            raise DomainError(
                "not_permitted", f"{capability} is not permitted for this Member"
            )
