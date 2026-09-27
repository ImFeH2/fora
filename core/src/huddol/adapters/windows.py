from __future__ import annotations

import ctypes
import secrets
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from ctypes import wintypes

_advapi = ctypes.WinDLL("advapi32", use_last_error=True)
_kernel = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel.GetCurrentProcess.restype = wintypes.HANDLE
_kernel.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel.CloseHandle.restype = wintypes.BOOL
_kernel.LocalFree.argtypes = [ctypes.c_void_p]
_kernel.LocalFree.restype = ctypes.c_void_p
_advapi.ConvertStringSidToSidW.argtypes = [
    wintypes.LPCWSTR,
    ctypes.POINTER(ctypes.c_void_p),
]
_advapi.ConvertStringSidToSidW.restype = wintypes.BOOL
_advapi.OpenProcessToken.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.HANDLE),
]
_advapi.OpenProcessToken.restype = wintypes.BOOL


def _raise_if_error(code: int) -> None:
    if code:
        raise ctypes.WinError(code)


def _require(value: int) -> None:
    if not value:
        raise ctypes.WinError(ctypes.get_last_error())


def _sid_pointer(value: str) -> ctypes.c_void_p:
    sid = ctypes.c_void_p()
    _require(_advapi.ConvertStringSidToSidW(value, ctypes.byref(sid)))
    return sid


class SecurityAttributes(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle", wintypes.BOOL),
    ]


class TokenUser(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class JobBasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class JobExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JobBasicLimits),
        ("IoInfo", IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class JobAccounting(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


_advapi.GetTokenInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
]
_advapi.GetTokenInformation.restype = wintypes.BOOL
_advapi.ConvertSidToStringSidW.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(wintypes.LPWSTR),
]
_advapi.ConvertSidToStringSidW.restype = wintypes.BOOL
_advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_void_p,
]
_advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
_kernel.CreateJobObjectW.argtypes = [
    ctypes.POINTER(SecurityAttributes),
    wintypes.LPCWSTR,
]
_kernel.CreateJobObjectW.restype = wintypes.HANDLE
_kernel.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
_kernel.OpenJobObjectW.restype = wintypes.HANDLE
_kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
_kernel.AssignProcessToJobObject.restype = wintypes.BOOL
_kernel.SetInformationJobObject.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
]
_kernel.SetInformationJobObject.restype = wintypes.BOOL
_kernel.QueryInformationJobObject.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.c_void_p,
]
_kernel.QueryInformationJobObject.restype = wintypes.BOOL
_kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
_kernel.TerminateJobObject.restype = wintypes.BOOL


def current_user_sid() -> str:
    token = wintypes.HANDLE()
    _require(
        _advapi.OpenProcessToken(_kernel.GetCurrentProcess(), 0x8, ctypes.byref(token))
    )
    try:
        size = wintypes.DWORD()
        result = _advapi.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
        if result or ctypes.get_last_error() != 122:
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_string_buffer(size.value)
        _require(
            _advapi.GetTokenInformation(token, 1, buffer, size, ctypes.byref(size))
        )
        user = ctypes.cast(buffer, ctypes.POINTER(TokenUser)).contents
        text = wintypes.LPWSTR()
        _require(_advapi.ConvertSidToStringSidW(user.Sid, ctypes.byref(text)))
        try:
            return str(text.value)
        finally:
            _kernel.LocalFree(text)
    finally:
        _require(_kernel.CloseHandle(token))


@contextmanager
def private_security(permissions: int) -> Iterator[SecurityAttributes]:
    sid = current_user_sid()
    descriptor = ctypes.c_void_p()
    _require(
        _advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            f"O:{sid}D:P(A;;0x{permissions:x};;;{sid})",
            1,
            ctypes.byref(descriptor),
            None,
        )
    )
    try:
        yield SecurityAttributes(ctypes.sizeof(SecurityAttributes), descriptor, False)
    finally:
        _kernel.LocalFree(descriptor)


class WindowsJob:
    def __init__(self) -> None:
        self.name = "Local\\Huddol-" + secrets.token_hex(24)
        with private_security(0x1F003F) as security:
            self.handle = _kernel.CreateJobObjectW(ctypes.byref(security), self.name)
            error = ctypes.get_last_error()
        if not self.handle:
            raise ctypes.WinError(error)
        try:
            if error == 183:
                raise ctypes.WinError(error)
            limits = JobExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x2000
            _require(
                _kernel.SetInformationJobObject(
                    self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
                )
            )
        except BaseException:
            self.close()
            raise

    def active_processes(self) -> int:
        accounting = JobAccounting()
        _require(
            _kernel.QueryInformationJobObject(
                self.handle,
                1,
                ctypes.byref(accounting),
                ctypes.sizeof(accounting),
                None,
            )
        )
        return accounting.ActiveProcesses

    def request_termination(self) -> None:
        _require(_kernel.TerminateJobObject(self.handle, 1))

    def terminate(self, *, deadline: float | None = None) -> None:
        deadline = time.monotonic() + 5 if deadline is None else deadline
        self.request_termination()
        while self.active_processes():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Windows job did not terminate: {self.name}")
            time.sleep(min(0.01, remaining))

    def close(self) -> None:
        if self.handle:
            _require(_kernel.CloseHandle(self.handle))
            self.handle = None


def close_process_handle(process: subprocess.Popen[bytes]) -> None:
    handle = getattr(process, "_handle", None)
    if handle is None:
        raise RuntimeError("Windows process handle is missing")
    if not handle.closed:
        _require(_kernel.CloseHandle(handle))
        handle.Detach()


def join_job(name: str) -> None:
    handle = _kernel.OpenJobObjectW(0x1, False, name)
    _require(handle)
    try:
        _require(_kernel.AssignProcessToJobObject(handle, _kernel.GetCurrentProcess()))
    finally:
        _require(_kernel.CloseHandle(handle))
