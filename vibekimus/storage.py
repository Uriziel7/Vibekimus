"""Which drive a file lies on and whether that drive spins (a hard disk), so that reads from
a hard disk go one at a time while SSDs are read in parallel.

settings.STORAGE_MODE: "auto" asks Windows about each drive (whether moving to another
place on it costs a seek, which only spinning disks report), "ssd" or "hdd" decide for
every drive. A drive Windows cannot tell about (a network share, some USB enclosures,
virtual drives) counts as an SSD: reading it in parallel is the safe default for speed.
Each drive is asked once."""
import contextlib
import ctypes
import os
import threading
from enum import StrEnum

from vibekimus import settings

# Windows: DeviceIoControl(IOCTL_STORAGE_QUERY_PROPERTY) for StorageDeviceSeekPenaltyProperty.
IOCTL_STORAGE_QUERY_PROPERTY = 0x002D1400
STORAGE_DEVICE_SEEK_PENALTY_PROPERTY = 7
PROPERTY_STANDARD_QUERY = 0
FILE_SHARE_READ_WRITE = 0x1 | 0x2
OPEN_EXISTING = 3
NO_ACCESS = 0              # enough for this question, and needs no administrator rights
VOLUME_PATH = "\\\\.\\{drive}"


class StorageMode(StrEnum):
    """settings.STORAGE_MODE: ask Windows about each drive, or treat every drive alike."""
    AUTO = "auto"
    SSD = "ssd"
    HDD = "hdd"


class _PropertyQuery(ctypes.Structure):
    _fields_ = [("PropertyId", ctypes.c_int), ("QueryType", ctypes.c_int),
                ("AdditionalParameters", ctypes.c_ubyte * 1)]


class _SeekPenalty(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("Size", ctypes.c_ulong), ("IncursSeekPenalty", ctypes.c_ubyte)]


def drive_of(path):
    """'C:' for a path on drive C, '\\\\SERVER\\SHARE' for a network path (upper case)."""
    return os.path.splitdrive(os.path.abspath(path))[0].upper()


def _seek_penalty(drive):
    """True for a spinning disk, False for an SSD, None when Windows cannot tell."""
    if os.name != "nt" or len(drive) != 2 or drive[1] != ":":
        return None
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    handle = kernel32.CreateFileW(VOLUME_PATH.format(drive=drive), NO_ACCESS, FILE_SHARE_READ_WRITE, None,
                                  OPEN_EXISTING, 0, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        return None
    try:
        query = _PropertyQuery(STORAGE_DEVICE_SEEK_PENALTY_PROPERTY, PROPERTY_STANDARD_QUERY)
        answer, returned = _SeekPenalty(), wintypes.DWORD()
        if not kernel32.DeviceIoControl(handle, IOCTL_STORAGE_QUERY_PROPERTY, ctypes.byref(query),
                                        ctypes.sizeof(query), ctypes.byref(answer), ctypes.sizeof(answer),
                                        ctypes.byref(returned), None):
            return None
        return bool(answer.IncursSeekPenalty)
    finally:
        kernel32.CloseHandle(handle)


class Drives:
    """What is known about each drive, and one read lock for each spinning one."""

    def __init__(self, ask=_seek_penalty):
        self._ask = ask
        self._spinning = {}   # drive -> True / False / None (asked once)
        self._locks = {}
        self._guard = threading.Lock()

    def spins(self, path):
        mode = settings.STORAGE_MODE
        if mode in (StorageMode.SSD, StorageMode.HDD):
            return mode == StorageMode.HDD
        drive = drive_of(path)
        with self._guard:
            if drive not in self._spinning:
                self._spinning[drive] = self._ask(drive)
            return bool(self._spinning[drive])

    def reading(self, path):
        """Hold while reading from `path`: one at a time on a spinning disk, no wait on an SSD."""
        if not self.spins(path):
            return contextlib.nullcontext()
        drive = drive_of(path)
        with self._guard:
            return self._locks.setdefault(drive, threading.Lock())

    def describe(self, path):
        """'C: SSD' / 'D: hard disk' / 'E: unknown (read as SSD)' for the log."""
        drive = drive_of(path)
        if settings.STORAGE_MODE != StorageMode.AUTO:
            return f"{drive} {'hard disk' if settings.STORAGE_MODE == StorageMode.HDD else 'SSD'} (set by hand)"
        self.spins(path)
        known = self._spinning.get(drive)
        return f"{drive} {'unknown (read as SSD)' if known is None else 'hard disk' if known else 'SSD'}"


DRIVES = Drives()
