# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The flock subset used for vLLM's dedicated lock files on Windows.

Supports one shared/exclusive acquisition followed by unlock or close, not
reentrant acquisition or lock conversion. Unlike flock, Windows byte-range
locks also restrict file I/O, so callers must use separate lock files.
"""

import ctypes
import msvcrt
from ctypes import wintypes
from typing import IO, Any

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8

_LOCKFILE_FAIL_IMMEDIATELY = 1
_LOCKFILE_EXCLUSIVE_LOCK = 2
_MAXDWORD = 0xFFFFFFFF


class _OVERLAPPED(ctypes.Structure):
    _fields_ = [
        ("Internal", ctypes.c_size_t),
        ("InternalHigh", ctypes.c_size_t),
        ("Offset", wintypes.DWORD),
        ("OffsetHigh", wintypes.DWORD),
        ("hEvent", wintypes.HANDLE),
    ]


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_lock = _kernel32.LockFileEx
_lock.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(_OVERLAPPED),
]
_lock.restype = wintypes.BOOL
_unlock = _kernel32.UnlockFileEx
_unlock.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(_OVERLAPPED),
]
_unlock.restype = wintypes.BOOL


def flock(fd: int | IO[Any], operation: int) -> None:
    kind = operation & ~LOCK_NB
    if kind not in (LOCK_SH, LOCK_EX, LOCK_UN):
        raise ValueError(f"Unsupported flock operation: {operation}")
    handle = msvcrt.get_osfhandle(fd if isinstance(fd, int) else fd.fileno())
    overlapped = _OVERLAPPED()
    if kind == LOCK_UN:
        result = _unlock(handle, 0, _MAXDWORD, _MAXDWORD, ctypes.byref(overlapped))
    else:
        flags = _LOCKFILE_EXCLUSIVE_LOCK if kind == LOCK_EX else 0
        if operation & LOCK_NB:
            flags |= _LOCKFILE_FAIL_IMMEDIATELY
        result = _lock(handle, flags, 0, _MAXDWORD, _MAXDWORD, ctypes.byref(overlapped))
    if not result:
        raise ctypes.WinError(ctypes.get_last_error())
