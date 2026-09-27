from __future__ import annotations

import hashlib
import os
import stat
import sys
import warnings
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError

from huddol.core.attachment import (
    IMAGE_TYPES,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_SIDE,
    Attachment,
    ImageData,
    Upload,
    identifier,
)
from huddol.core.errors import DomainError


def decode_image(data: bytes) -> ImageData:
    if len(data) > MAX_IMAGE_BYTES:
        raise DomainError("image_too_large", "Images must be at most 5 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP"}:
                    raise DomainError(
                        "unsupported_image", "Choose a PNG, JPEG or WebP image"
                    )
                width, height = image.size
                if (
                    max(width, height) > MAX_IMAGE_SIDE
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise DomainError(
                        "image_dimensions_exceeded",
                        "Images must fit 8192 pixels per side and 20 MP",
                    )
                if getattr(image, "n_frames", 1) != 1:
                    raise DomainError("animated_image", "Choose a static image")
                media_type = Image.MIME[image.format]
                image.verify()
            with Image.open(BytesIO(data)) as image:
                image.load()
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise DomainError("invalid_image", "The image cannot be decoded") from error
    return ImageData(data, media_type, width, height)


class DirectoryUploads:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, attachment_id: str) -> str:
        return str(self.root / identifier(attachment_id))

    def writer(self, upload_id: str) -> BinaryIO:
        return Path(self.path(upload_id) + ".part").open("xb")

    def discard(self, upload_id: str) -> None:
        Path(self.path(upload_id) + ".part").unlink(missing_ok=True)
        Path(self.path(upload_id)).unlink(missing_ok=True)

    def complete(self, upload: Upload) -> Attachment:
        temporary = Path(self.path(upload.id) + ".part")
        with temporary.open("rb") as source:
            if os.fstat(source.fileno()).st_size != upload.size:
                raise DomainError(
                    "upload_incomplete", "The uploaded file size does not match"
                )
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        media_type = "application/octet-stream"
        width = height = None
        if upload.media_type in IMAGE_TYPES:
            with temporary.open("rb") as source:
                image = decode_image(source.read(MAX_IMAGE_BYTES + 1))
            media_type, width, height = image.media_type, image.width, image.height
        os.replace(temporary, self.path(upload.id))
        if sys.platform != "win32":
            descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return Attachment(
            upload.id, upload.name, upload.size, media_type, digest, width, height
        )

    def reader(self, attachment: Attachment) -> BinaryIO:
        path = Path(self.path(attachment.id))
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size != attachment.size:
                raise DomainError(
                    "attachment_invalid", "The stored attachment is not a valid file"
                )
            source = path.open("rb")
        except FileNotFoundError as error:
            raise DomainError(
                "attachment_missing", "The attachment file is missing"
            ) from error
        try:
            opened = os.fstat(source.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_size != attachment.size
                or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)
            ):
                raise DomainError(
                    "attachment_invalid", "The stored attachment is not a valid file"
                )
            if hashlib.file_digest(source, "sha256").hexdigest() != attachment.sha256:
                raise DomainError(
                    "attachment_changed", "The attachment content has changed"
                )
            source.seek(0)
        except BaseException:
            source.close()
            raise
        return source

    def image(self, attachment: Attachment) -> ImageData:
        if attachment.media_type not in IMAGE_TYPES:
            raise DomainError(
                "unsupported_image", "This attachment is not a supported image"
            )
        with self.reader(attachment) as source:
            data = source.read(MAX_IMAGE_BYTES + 1)
        if (
            len(data) != attachment.size
            or hashlib.sha256(data).hexdigest() != attachment.sha256
        ):
            raise DomainError(
                "attachment_changed", "The attachment content has changed"
            )
        image = decode_image(data)
        if (image.media_type, image.width, image.height) != (
            attachment.media_type,
            attachment.width,
            attachment.height,
        ):
            raise DomainError(
                "attachment_changed", "The attachment image details have changed"
            )
        return image
