from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from huddol.core.errors import DomainError

MIB = 1024 * 1024
MAX_FILES = 10
MAX_FILE_BYTES = 20 * MIB
MAX_MESSAGE_BYTES = 50 * MIB
MAX_IMAGE_BYTES = 5 * MIB
MAX_IMAGE_SIDE = 8192
MAX_IMAGE_PIXELS = 20_000_000
IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})


@dataclass(frozen=True)
class Attachment:
    id: str
    name: str
    size: int
    media_type: str
    sha256: str
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class ImageData:
    data: bytes
    media_type: str
    width: int
    height: int


@dataclass(frozen=True)
class ViewedAttachment:
    discussion_id: int
    message_id: int
    attachment: Attachment
    image: ImageData


@dataclass(frozen=True)
class Upload:
    id: str
    discussion_id: int
    owner_id: int
    client_upload_id: str
    name: str
    size: int
    media_type: str
    state: str
    expires_at: float


def identifier(value: object) -> str:
    if not isinstance(value, str):
        raise DomainError("invalid_id", "Expected a UUID")
    try:
        parsed = UUID(value)
    except ValueError as error:
        raise DomainError("invalid_id", "Expected a UUID") from error
    if str(parsed) != value:
        raise DomainError("invalid_id", "Expected a canonical UUID")
    return value


def validate_upload(name: object, size: object, media_type: object) -> None:
    if (
        not isinstance(name, str)
        or not name.strip()
        or len(name) > 255
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
    ):
        raise DomainError("invalid_filename", "Choose a filename of 1–255 characters")
    if type(size) is not int or not 0 <= size <= MAX_FILE_BYTES:
        raise DomainError("file_too_large", "Files must be at most 20 MiB")
    if not isinstance(media_type, str) or len(media_type) > 127:
        raise DomainError("invalid_media_type", "Invalid file media type")
    if media_type in IMAGE_TYPES and size > MAX_IMAGE_BYTES:
        raise DomainError("image_too_large", "Images must be at most 5 MiB")


def validate_attachments(attachments: tuple[Attachment, ...]) -> None:
    if len(attachments) > MAX_FILES:
        raise DomainError("too_many_files", "A message can contain at most 10 files")
    if sum(item.size for item in attachments) > MAX_MESSAGE_BYTES:
        raise DomainError(
            "attachments_too_large", "Message files must total at most 50 MiB"
        )
