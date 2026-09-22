from __future__ import annotations

from threading import RLock
from typing import BinaryIO

from huddol.core.attachment import (
    Attachment,
    Upload,
    ViewedAttachment,
    identifier,
    validate_upload,
)
from huddol.core.errors import DomainError
from huddol.ports.uploads import UploadFiles, UploadStore


class Uploads:
    def __init__(self, store: UploadStore, files: UploadFiles) -> None:
        self.store = store
        self.files = files
        self._lock = RLock()
        self._active: set[str] = set()

    def create(
        self,
        discussion_id: int,
        owner_id: int,
        client_upload_id: str,
        name: str,
        size: int,
        media_type: str,
    ) -> Upload:
        identifier(client_upload_id)
        validate_upload(name, size, media_type)
        return self.store.create_upload(
            discussion_id,
            owner_id,
            client_upload_id,
            name,
            size,
            media_type,
        )

    def status(self, upload_id: str, owner_id: int) -> Upload:
        identifier(upload_id)
        with self._lock:
            record = self.store.get_upload(upload_id, owner_id)
            if record.state == "ready":
                attachment = self.store.ready_attachment(upload_id, owner_id)
                try:
                    with self.files.reader(attachment):
                        pass
                except DomainError as error:
                    if error.code not in {
                        "attachment_missing",
                        "attachment_invalid",
                        "attachment_changed",
                    }:
                        raise
                    self.cancel(upload_id, owner_id)
                    return self.store.get_upload(upload_id, owner_id)
            return record

    def begin(self, upload_id: str, owner_id: int) -> tuple[Upload, BinaryIO]:
        identifier(upload_id)
        with self._lock:
            if upload_id in self._active or len(self._active) >= 2:
                raise DomainError(
                    "upload_busy",
                    "Two files are already uploading; retry when they finish",
                )
            record = self.store.start_upload(upload_id, owner_id)
            self._active.add(upload_id)
            try:
                return record, self.files.writer(upload_id)
            except BaseException:
                self._active.remove(upload_id)
                self.store.cancel_upload(upload_id, owner_id)
                self._discard(upload_id)
                raise

    def complete(self, record: Upload) -> Attachment:
        with self._lock:
            current = self.store.get_upload(record.id, record.owner_id)
            if current.state != "receiving":
                raise DomainError("upload_cancelled", "The upload was cancelled")
            result = self.files.complete(record)
            self.store.finish_upload(record, result)
            return result

    def end(self, record: Upload) -> None:
        with self._lock:
            self._active.discard(record.id)
            current = self.store.get_upload(record.id, record.owner_id)
            if current.state == "receiving":
                self.store.cancel_upload(record.id, record.owner_id)
                self._discard(record.id)
            elif current.state == "deleting":
                self._discard(record.id)

    def cancel(self, upload_id: str, owner_id: int) -> None:
        identifier(upload_id)
        with self._lock:
            record = self.store.cancel_upload(upload_id, owner_id)
            if record.state == "deleting" and upload_id not in self._active:
                self._discard(upload_id)

    def _discard(self, upload_id: str) -> None:
        self.files.discard(upload_id)
        self.store.release_upload(upload_id)

    def cleanup(self, *, restart: bool = False) -> None:
        with self._lock:
            for record in self.store.abandoned_uploads(restart=restart):
                if record.id not in self._active:
                    self._discard(record.id)

    def view(
        self,
        discussion_id: int,
        message_id: int,
        attachment_id: str,
        member_id: int,
    ) -> ViewedAttachment:
        identifier(attachment_id)
        attachment = self.store.attachment(
            discussion_id, message_id, attachment_id, member_id
        )
        return ViewedAttachment(
            discussion_id, message_id, attachment, self.files.image(attachment)
        )

    def read(
        self,
        discussion_id: int,
        message_id: int,
        attachment_id: str,
        member_id: int,
    ) -> tuple[Attachment, BinaryIO]:
        identifier(attachment_id)
        attachment = self.store.attachment(
            discussion_id, message_id, attachment_id, member_id
        )
        return attachment, self.files.reader(attachment)
