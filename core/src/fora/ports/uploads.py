from __future__ import annotations

from typing import BinaryIO, Protocol

from fora.core.attachment import Attachment, ImageData, Upload


class UploadStore(Protocol):
    def create_upload(
        self,
        discussion_id: int,
        owner_id: int,
        client_upload_id: str,
        name: str,
        size: int,
        media_type: str,
    ) -> Upload: ...

    def get_upload(self, upload_id: str, owner_id: int) -> Upload: ...

    def ready_attachment(self, upload_id: str, owner_id: int) -> Attachment: ...

    def start_upload(self, upload_id: str, owner_id: int) -> Upload: ...

    def finish_upload(self, upload: Upload, attachment: Attachment) -> None: ...

    def cancel_upload(self, upload_id: str, owner_id: int) -> Upload: ...

    def abandoned_uploads(self, *, restart: bool = False) -> tuple[Upload, ...]: ...

    def release_upload(self, upload_id: str) -> None: ...

    def attachment(
        self,
        discussion_id: int,
        message_id: int,
        attachment_id: str,
        member_id: int,
    ) -> Attachment: ...


class UploadFiles(Protocol):
    def writer(self, upload_id: str) -> BinaryIO: ...

    def complete(self, upload: Upload) -> Attachment: ...

    def discard(self, upload_id: str) -> None: ...

    def path(self, attachment_id: str) -> str: ...

    def reader(self, attachment: Attachment) -> BinaryIO: ...

    def image(self, attachment: Attachment) -> ImageData: ...
