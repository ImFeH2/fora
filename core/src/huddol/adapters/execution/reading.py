from __future__ import annotations

import errno
import json
import os
import stat
import sys
from collections.abc import Sequence

ERROR_EXIT_CODE = 3


def _diagnostic(code: str) -> int:
    message = {
        "not_found": "The file does not exist",
        "permission_denied": "The file cannot be read",
        "not_regular_file": "The path is not a regular file",
        "file_too_large": "The file exceeds the requested byte limit",
        "read_failed": "The file could not be read",
    }[code]
    payload = json.dumps(
        {"code": code, "message": message},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    sys.stderr.buffer.write(payload)
    sys.stderr.buffer.flush()
    return ERROR_EXIT_CODE


def _error_code(error: OSError) -> str:
    if isinstance(error, FileNotFoundError) or error.errno in {
        errno.ENOENT,
        errno.ENOTDIR,
    }:
        return "not_found"
    if isinstance(error, PermissionError) or error.errno in {errno.EACCES, errno.EPERM}:
        return "permission_denied"
    if error.errno == errno.EISDIR:
        return "not_regular_file"
    return "read_failed"


def read_file(path: str, max_bytes: int) -> int:
    descriptor: int | None = None
    data: bytes | None = None
    failure: str | None = None
    try:
        info = os.stat(path)
        if not stat.S_ISREG(info.st_mode):
            failure = "not_regular_file"
        elif info.st_size > max_bytes:
            failure = "file_too_large"
        else:
            flags = os.O_RDONLY
            if os.name == "nt":
                flags |= getattr(os, "O_BINARY", 0)
            else:
                flags |= getattr(os, "O_NONBLOCK", 0)
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                failure = "not_regular_file"
            elif opened.st_size > max_bytes:
                failure = "file_too_large"
            else:
                data = os.read(descriptor, max_bytes + 1)
    except OSError as error:
        failure = _error_code(error)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as error:
                failure = _error_code(error)
    if failure is not None:
        return _diagnostic(failure)
    assert data is not None
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
    return 0


def dispatch_read_file(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        return _diagnostic("read_failed")
    path, raw_limit = argv
    try:
        max_bytes = int(raw_limit)
    except ValueError:
        return _diagnostic("read_failed")
    if max_bytes <= 0 or str(max_bytes) != raw_limit:
        return _diagnostic("read_failed")
    return read_file(path, max_bytes)
