from __future__ import annotations

import ctypes
import msvcrt
import os
import secrets
from ctypes import wintypes
from pathlib import Path

from huddol.adapters.windows import (
    SecurityAttributes,
    _advapi,
    _kernel,
    _raise_if_error,
    _require,
    _sid_pointer,
    current_user_sid,
    private_security,
)

FILE_ALL_ACCESS = 0x1F01FF
FILE_GENERIC_READ = 0x120089


class Trustee(ctypes.Structure):
    _fields_ = [
        ("pMultipleTrustee", ctypes.c_void_p),
        ("MultipleTrusteeOperation", ctypes.c_int),
        ("TrusteeForm", ctypes.c_int),
        ("TrusteeType", ctypes.c_int),
        ("ptstrName", ctypes.c_void_p),
    ]


class ExplicitAccess(ctypes.Structure):
    _fields_ = [
        ("grfAccessPermissions", wintypes.DWORD),
        ("grfAccessMode", ctypes.c_int),
        ("grfInheritance", wintypes.DWORD),
        ("Trustee", Trustee),
    ]


_kernel.CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(SecurityAttributes),
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
]
_kernel.CreateFileW.restype = wintypes.HANDLE
_advapi.GetSecurityInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.DWORD,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
]
_advapi.GetSecurityInfo.restype = wintypes.DWORD
_advapi.GetSecurityDescriptorControl.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(wintypes.WORD),
    ctypes.POINTER(wintypes.DWORD),
]
_advapi.GetSecurityDescriptorControl.restype = wintypes.BOOL
_advapi.GetExplicitEntriesFromAclW.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(wintypes.ULONG),
    ctypes.POINTER(ctypes.POINTER(ExplicitAccess)),
]
_advapi.GetExplicitEntriesFromAclW.restype = wintypes.DWORD
_advapi.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_advapi.EqualSid.restype = wintypes.BOOL


def _validate_private(handle: int, path: Path) -> None:
    owner = ctypes.c_void_p()
    acl = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    entries = ctypes.POINTER(ExplicitAccess)()
    sid = _sid_pointer(current_user_sid())
    try:
        _raise_if_error(
            _advapi.GetSecurityInfo(
                handle,
                1,
                0x1 | 0x4,
                ctypes.byref(owner),
                None,
                ctypes.byref(acl),
                None,
                ctypes.byref(descriptor),
            )
        )
        control = wintypes.WORD()
        revision = wintypes.DWORD()
        _require(
            _advapi.GetSecurityDescriptorControl(
                descriptor, ctypes.byref(control), ctypes.byref(revision)
            )
        )
        if not owner or not _advapi.EqualSid(owner, sid):
            raise PermissionError(
                f"Private file is not owned by the current user: {path}"
            )
        if not acl or not control.value & 0x1000:
            raise PermissionError(
                f"Private file requires a protected non-NULL DACL: {path}"
            )
        count = wintypes.ULONG()
        _raise_if_error(
            _advapi.GetExplicitEntriesFromAclW(
                acl, ctypes.byref(count), ctypes.byref(entries)
            )
        )
        allowed = 0
        for index in range(count.value):
            entry = entries[index]
            trustee = entry.Trustee
            if (
                entry.grfAccessMode not in (1, 2)
                or entry.grfInheritance
                or trustee.TrusteeForm != 0
                or trustee.pMultipleTrustee
                or not trustee.ptstrName
                or not _advapi.EqualSid(trustee.ptstrName, sid)
            ):
                raise PermissionError(
                    f"Private file has unsupported or non-private access rules: {path}"
                )
            allowed |= entry.grfAccessPermissions
        if allowed & FILE_GENERIC_READ != FILE_GENERIC_READ:
            raise PermissionError(
                f"Private file does not grant the current user read access: {path}"
            )
    finally:
        if entries:
            _kernel.LocalFree(entries)
        if descriptor:
            _kernel.LocalFree(descriptor)
        _kernel.LocalFree(sid)


def _file_descriptor(handle: int, flags: int) -> int:
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(handle, flags)
    except BaseException:
        _require(_kernel.CloseHandle(handle))
        raise


def read_private(path: Path) -> str:
    descriptor = _file_descriptor(
        _kernel.CreateFileW(str(path), 0x80000000 | 0x20000, 0x1, None, 3, 0x80, None),
        os.O_RDONLY | os.O_BINARY,
    )
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        _validate_private(msvcrt.get_osfhandle(stream.fileno()), path)
        return stream.read()


def write_private(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.huddol-tmp")
    created = False
    try:
        with private_security(FILE_ALL_ACCESS) as security:
            handle = _kernel.CreateFileW(
                str(temporary), 0x40000000, 0, ctypes.byref(security), 1, 0x80, None
            )
            error = ctypes.get_last_error()
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(error)
        created = True
        descriptor = _file_descriptor(handle, os.O_WRONLY | os.O_BINARY)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)
