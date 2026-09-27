from __future__ import annotations

import ctypes
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
