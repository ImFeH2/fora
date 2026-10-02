from __future__ import annotations

import ctypes
import secrets
import subprocess
from collections.abc import Sequence
from ctypes import wintypes
from pathlib import Path
from threading import Lock

from fora.adapters.windows import (
    _advapi,
    _kernel,
    _raise_if_error,
    _require,
    _sid_pointer,
)

TOKEN_ACCESS = 0x0001 | 0x0002 | 0x0008 | 0x0080 | 0x0100
TOKEN_FLAGS = 0x1 | 0x4 | 0x8
DACL_SECURITY_INFORMATION = 0x4
SE_FILE_OBJECT = 1
SE_WINDOW_OBJECT = 7
SET_ACCESS = 2
REVOKE_ACCESS = 4
TRUSTEE_IS_SID = 0
SUB_CONTAINERS_AND_OBJECTS_INHERIT = 0x3
MODIFY_ACCESS = 0x120089 | 0x120116 | 0x1200A0 | 0x10000
STARTF_USESTDHANDLES = 0x100
HANDLE_FLAG_INHERIT = 0x1
CREATE_UNICODE_ENVIRONMENT = 0x400
WINDOW_STATION_ACCESS = 0x000F037F
DESKTOP_ACCESS = 0x000F01FF
INFINITE = 0xFFFFFFFF


class SidAndAttributes(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class Trustee(ctypes.Structure):
    pass


Trustee._fields_ = [
    ("pMultipleTrustee", ctypes.POINTER(Trustee)),
    ("MultipleTrusteeOperation", ctypes.c_int),
    ("TrusteeForm", ctypes.c_int),
    ("TrusteeType", ctypes.c_int),
    ("ptstrName", wintypes.LPWSTR),
]


class ExplicitAccess(ctypes.Structure):
    _fields_ = [
        ("grfAccessPermissions", wintypes.DWORD),
        ("grfAccessMode", ctypes.c_int),
        ("grfInheritance", wintypes.DWORD),
        ("Trustee", Trustee),
    ]


class StartupInfo(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_ubyte)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class ProcessInformation(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel.GetCurrentThreadId.restype = wintypes.DWORD
_kernel.GetStdHandle.argtypes = [wintypes.DWORD]
_kernel.GetStdHandle.restype = wintypes.HANDLE
_kernel.SetHandleInformation.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.DWORD,
]
_kernel.SetHandleInformation.restype = wintypes.BOOL
_kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_kernel.WaitForSingleObject.restype = wintypes.DWORD
_kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
_kernel.GetExitCodeProcess.restype = wintypes.BOOL
_advapi.GetNamedSecurityInfoW.argtypes = [
    wintypes.LPWSTR,
    ctypes.c_int,
    wintypes.DWORD,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
]
_advapi.GetNamedSecurityInfoW.restype = wintypes.DWORD
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
_advapi.SetEntriesInAclW.argtypes = [
    wintypes.ULONG,
    ctypes.POINTER(ExplicitAccess),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
]
_advapi.SetEntriesInAclW.restype = wintypes.DWORD
_advapi.SetNamedSecurityInfoW.argtypes = [
    wintypes.LPWSTR,
    ctypes.c_int,
    wintypes.DWORD,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_advapi.SetNamedSecurityInfoW.restype = wintypes.DWORD
_advapi.SetSecurityInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    wintypes.DWORD,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_advapi.SetSecurityInfo.restype = wintypes.DWORD
_advapi.CreateRestrictedToken.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(SidAndAttributes),
    ctypes.POINTER(wintypes.HANDLE),
]
_advapi.CreateRestrictedToken.restype = wintypes.BOOL
_advapi.CreateProcessAsUserW.argtypes = [
    wintypes.HANDLE,
    wintypes.LPCWSTR,
    wintypes.LPWSTR,
    ctypes.c_void_p,
    ctypes.c_void_p,
    wintypes.BOOL,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.LPCWSTR,
    ctypes.POINTER(StartupInfo),
    ctypes.POINTER(ProcessInformation),
]
_advapi.CreateProcessAsUserW.restype = wintypes.BOOL
_user32.GetProcessWindowStation.restype = wintypes.HANDLE
_user32.GetThreadDesktop.argtypes = [wintypes.DWORD]
_user32.GetThreadDesktop.restype = wintypes.HANDLE


def _updated_acl(
    old_acl: ctypes.c_void_p,
    sid: ctypes.c_void_p,
    mode: int,
    permissions: int,
    inheritance: int,
) -> ctypes.c_void_p:
    trustee = Trustee(
        None,
        0,
        TRUSTEE_IS_SID,
        0,
        ctypes.cast(sid, wintypes.LPWSTR),
    )
    access = ExplicitAccess(
        permissions if mode == SET_ACCESS else 0,
        mode,
        inheritance,
        trustee,
    )
    new_acl = ctypes.c_void_p()
    _raise_if_error(
        _advapi.SetEntriesInAclW(
            1,
            ctypes.byref(access),
            old_acl,
            ctypes.byref(new_acl),
        )
    )
    return new_acl


def _change_ace(
    path: Path, sid: ctypes.c_void_p, mode: int, permissions: int = MODIFY_ACCESS
) -> None:
    old_acl = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    new_acl = ctypes.c_void_p()
    _raise_if_error(
        _advapi.GetNamedSecurityInfoW(
            str(path),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            None,
            None,
            ctypes.byref(old_acl),
            None,
            ctypes.byref(descriptor),
        )
    )
    try:
        new_acl = _updated_acl(
            old_acl,
            sid,
            mode,
            permissions,
            SUB_CONTAINERS_AND_OBJECTS_INHERIT,
        )
        _raise_if_error(
            _advapi.SetNamedSecurityInfoW(
                str(path),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                None,
                None,
                new_acl,
                None,
            )
        )
    finally:
        if new_acl:
            _kernel.LocalFree(new_acl)
        if descriptor:
            _kernel.LocalFree(descriptor)


def _change_user_object_ace(
    handle: wintypes.HANDLE,
    sid: ctypes.c_void_p,
    mode: int,
    permissions: int,
) -> None:
    old_acl = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    new_acl = ctypes.c_void_p()
    _raise_if_error(
        _advapi.GetSecurityInfo(
            handle,
            SE_WINDOW_OBJECT,
            DACL_SECURITY_INFORMATION,
            None,
            None,
            ctypes.byref(old_acl),
            None,
            ctypes.byref(descriptor),
        )
    )
    try:
        new_acl = _updated_acl(old_acl, sid, mode, permissions, 0)
        _raise_if_error(
            _advapi.SetSecurityInfo(
                handle,
                SE_WINDOW_OBJECT,
                DACL_SECURITY_INFORMATION,
                None,
                None,
                new_acl,
                None,
            )
        )
    finally:
        if new_acl:
            _kernel.LocalFree(new_acl)
        if descriptor:
            _kernel.LocalFree(descriptor)


def _current_logon_sid() -> str:
    value = subprocess.run(
        ["whoami.exe", "/logonid"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not value.startswith("S-1-5-5-"):
        raise OSError("Windows logon SID is unavailable")
    return value


class WindowsWriteAccess:
    def __init__(self, roots: Sequence[Path]) -> None:
        values = [secrets.randbits(30) + 1 for _ in range(4)]
        self.sid = "S-1-5-21-" + "-".join(str(value) for value in values)
        self._requested_roots = tuple(roots)
        self._sid = ctypes.c_void_p()
        self._logon_sid = ctypes.c_void_p()
        self._window_station: int | None = None
        self._desktop: int | None = None
        self._grants: list[tuple[Path, ctypes.c_void_p]] = []
        self._lock = Lock()
        self._ready = False
        self._closed = False

    def initialize(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Windows write access is closed")
            if self._ready:
                return
            if not self._sid:
                self._sid = _sid_pointer(self.sid)
            if not self._logon_sid:
                self._logon_sid = _sid_pointer(_current_logon_sid())
            for attribute, permissions, handle in (
                (
                    "_window_station",
                    WINDOW_STATION_ACCESS,
                    _user32.GetProcessWindowStation(),
                ),
                (
                    "_desktop",
                    DESKTOP_ACCESS,
                    _user32.GetThreadDesktop(_kernel.GetCurrentThreadId()),
                ),
            ):
                if getattr(self, attribute) is None:
                    _require(handle)
                    _change_user_object_ace(handle, self._sid, SET_ACCESS, permissions)
                    setattr(self, attribute, handle)
            for path in self._requested_roots:
                if path.is_dir():
                    for sid in (self._sid, self._logon_sid):
                        if (path, sid) not in self._grants:
                            _change_ace(path, sid, SET_ACCESS)
                            self._grants.append((path, sid))
            self._ready = True

    def close(self) -> None:
        with self._lock:
            self._closed = True
            errors: list[OSError] = []
            remaining: list[tuple[Path, ctypes.c_void_p]] = []
            for path, sid in self._grants:
                if path.exists():
                    try:
                        _change_ace(path, sid, REVOKE_ACCESS)
                    except OSError as error:
                        remaining.append((path, sid))
                        errors.append(error)
            self._grants = remaining
            for attribute, permissions in (
                ("_desktop", DESKTOP_ACCESS),
                ("_window_station", WINDOW_STATION_ACCESS),
            ):
                handle = getattr(self, attribute)
                if handle:
                    try:
                        _change_user_object_ace(
                            handle, self._sid, REVOKE_ACCESS, permissions
                        )
                        setattr(self, attribute, None)
                    except OSError as error:
                        errors.append(error)
            if errors:
                raise ExceptionGroup("Windows write access cleanup failed", errors)
            if self._logon_sid:
                if _kernel.LocalFree(self._logon_sid):
                    raise ctypes.WinError(ctypes.get_last_error())
                self._logon_sid = ctypes.c_void_p()
            if self._sid:
                if _kernel.LocalFree(self._sid):
                    raise ctypes.WinError(ctypes.get_last_error())
                self._sid = ctypes.c_void_p()


class StartupInfoEx(ctypes.Structure):
    _fields_ = [("StartupInfo", StartupInfo), ("lpAttributeList", ctypes.c_void_p)]


_kernel.InitializeProcThreadAttributeList.argtypes = [
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(ctypes.c_size_t),
]
_kernel.InitializeProcThreadAttributeList.restype = wintypes.BOOL
_kernel.UpdateProcThreadAttribute.argtypes = [
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_kernel.UpdateProcThreadAttribute.restype = wintypes.BOOL
_kernel.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
_kernel.DeleteProcThreadAttributeList.restype = None
_kernel.CreateProcessW.argtypes = _advapi.CreateProcessAsUserW.argtypes[1:]
_kernel.CreateProcessW.restype = wintypes.BOOL


def _create_command(
    token: wintypes.HANDLE, argv: list[str], cwd: str, process: ProcessInformation
) -> None:
    startup = StartupInfoEx()
    startup.StartupInfo.cb = ctypes.sizeof(startup)
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES
    startup.StartupInfo.lpDesktop = "Winsta0\\Default"
    startup.StartupInfo.hStdInput = _kernel.GetStdHandle(-10 & 0xFFFFFFFF)
    startup.StartupInfo.hStdOutput = _kernel.GetStdHandle(-11 & 0xFFFFFFFF)
    startup.StartupInfo.hStdError = _kernel.GetStdHandle(-12 & 0xFFFFFFFF)
    handles = tuple(
        dict.fromkeys(
            (
                startup.StartupInfo.hStdInput,
                startup.StartupInfo.hStdOutput,
                startup.StartupInfo.hStdError,
            )
        )
    )
    for handle in handles:
        _require(handle)
        _require(
            _kernel.SetHandleInformation(
                handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT
            )
        )
    inherited = (wintypes.HANDLE * len(handles))(*handles)
    size = ctypes.c_size_t()
    result = _kernel.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
    if result or ctypes.get_last_error() != 122:
        raise ctypes.WinError(ctypes.get_last_error())
    attributes = ctypes.create_string_buffer(size.value)
    _require(
        _kernel.InitializeProcThreadAttributeList(attributes, 1, 0, ctypes.byref(size))
    )
    try:
        startup.lpAttributeList = ctypes.cast(attributes, ctypes.c_void_p)
        _require(
            _kernel.UpdateProcThreadAttribute(
                attributes, 0, 0x20002, inherited, ctypes.sizeof(inherited), None, None
            )
        )
        command = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
        arguments = (
            None,
            command,
            None,
            None,
            True,
            CREATE_UNICODE_ENVIRONMENT | 0x80000,
            None,
            cwd,
            ctypes.byref(startup.StartupInfo),
            ctypes.byref(process),
        )
        if token:
            _require(_advapi.CreateProcessAsUserW(token, *arguments))
        else:
            _require(_kernel.CreateProcessW(*arguments))
    finally:
        _kernel.DeleteProcThreadAttributeList(attributes)


def run_restricted_command(sid_text: str | None, argv: list[str], cwd: str) -> int:
    if not argv:
        raise ValueError("Sandbox command is required")
    sid = ctypes.c_void_p()
    everyone_sid = ctypes.c_void_p()
    base_token = wintypes.HANDLE()
    restricted_token = wintypes.HANDLE()
    process = ProcessInformation()
    try:
        if sid_text is not None:
            sid = _sid_pointer(sid_text)
            everyone_sid = _sid_pointer("S-1-1-0")
            _require(
                _advapi.OpenProcessToken(
                    _kernel.GetCurrentProcess(), TOKEN_ACCESS, ctypes.byref(base_token)
                )
            )
            restricting = (SidAndAttributes * 2)(
                SidAndAttributes(sid, 0), SidAndAttributes(everyone_sid, 0)
            )
            _require(
                _advapi.CreateRestrictedToken(
                    base_token,
                    TOKEN_FLAGS,
                    0,
                    None,
                    0,
                    None,
                    2,
                    restricting,
                    ctypes.byref(restricted_token),
                )
            )
        _create_command(restricted_token, argv, cwd, process)
        if _kernel.WaitForSingleObject(process.hProcess, INFINITE) != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        exit_code = wintypes.DWORD()
        _require(_kernel.GetExitCodeProcess(process.hProcess, ctypes.byref(exit_code)))
        return exit_code.value
    finally:
        for handle in (
            process.hThread,
            process.hProcess,
            restricted_token,
            base_token,
        ):
            if handle:
                _require(_kernel.CloseHandle(handle))
        if everyone_sid:
            _kernel.LocalFree(everyone_sid)
        if sid:
            _kernel.LocalFree(sid)
