"""Windows Data Protection (DPAPI) through ctypes: bytes that only the same Windows
user account on the same computer can unlock. Used to keep the index key without a
password: Windows unlocks it when the user logs in."""
import ctypes
import os
from ctypes import wintypes

ENTROPY = b"Vibekimus index key"   # an extra secret mixed in, so other programs' blobs cannot be swapped in
CRYPTPROTECT_UI_FORBIDDEN = 0x1  # never show a dialog


class DpapiError(RuntimeError):
    """DPAPI is not available here, or refused to unlock (another user or computer)."""


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def is_available():
    return os.name == "nt"


def _blob(data):
    buffer = ctypes.create_string_buffer(data, len(data))
    return _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), buffer


def _call(function, data):
    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    source, _source_buffer = _blob(data)
    entropy, _entropy_buffer = _blob(ENTROPY)
    result = _Blob()
    if not function(ctypes.byref(source), None, ctypes.byref(entropy), None, None,
                    CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(result)):
        raise DpapiError(f"Windows refused ({ctypes.GetLastError()}): the key belongs to another "
                         f"user account or computer.")
    try:
        return ctypes.string_at(result.pbData, result.cbData)
    finally:
        kernel32.LocalFree(result.pbData)


def protect(data):
    if not is_available():
        raise DpapiError("Protection by the Windows account is only available on Windows.")
    return _call(ctypes.windll.crypt32.CryptProtectData, data)


def unprotect(blob):
    if not is_available():
        raise DpapiError("Protection by the Windows account is only available on Windows.")
    return _call(ctypes.windll.crypt32.CryptUnprotectData, blob)
