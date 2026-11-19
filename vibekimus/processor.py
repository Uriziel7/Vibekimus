"""Keeping CPU cores free for the rest of the computer while the program indexes: which
logical processors background work may use, and moving a process onto them at a lower
priority (the program itself while it indexes, the text reading processes for good).
Whole physical cores are kept free (with SMT both of a core's logical processors).
Standard library only (a worker process of the .exe uses it before the downloaded
libraries are importable)."""
import contextlib
import ctypes
import functools
import os
import sys

RELATION_PROCESSOR_CORE = 0
BELOW_NORMAL_PRIORITY_CLASS = 0x4000
LOWER_PRIORITY_NICENESS = 5     # elsewhere: os.nice() step
MIN_WORK_CORES = 1              # background work always keeps at least this many cores


class _ProcessorInformation(ctypes.Structure):
    """SYSTEM_LOGICAL_PROCESSOR_INFORMATION: a mask of logical processors and what they share."""
    _fields_ = [("mask", ctypes.c_size_t), ("relationship", ctypes.c_int), ("union", ctypes.c_byte * 16)]


def _windows_core_masks():
    kernel32 = ctypes.windll.kernel32
    size = ctypes.c_ulong(0)
    kernel32.GetLogicalProcessorInformation(None, ctypes.byref(size))
    count = size.value // ctypes.sizeof(_ProcessorInformation)
    table = (_ProcessorInformation * count)()
    if not kernel32.GetLogicalProcessorInformation(table, ctypes.byref(size)):
        raise OSError("GetLogicalProcessorInformation failed")
    return [entry.mask for entry in table if entry.relationship == RELATION_PROCESSOR_CORE]


@functools.lru_cache(maxsize=None)
def core_masks():
    """One mask of logical processors per physical core (each logical processor its own core
    when this cannot be read)."""
    if sys.platform == "win32":
        try:
            masks = _windows_core_masks()
            if masks:
                return tuple(masks)
        except (AttributeError, OSError):
            pass
    return tuple(1 << n for n in range(os.cpu_count() or 1))


def background_mask(free_cores):
    """The logical processors background work may use: every core but the last `free_cores`."""
    masks = core_masks()
    used = masks[:max(MIN_WORK_CORES, len(masks) - max(0, free_cores))]
    return functools.reduce(lambda total, mask: total | mask, used, 0)


def background_threads(free_cores):
    """How many logical processors background work may use."""
    return bin(background_mask(free_cores)).count("1")


def _kernel32():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p   # typed: a bare -1 would reach Windows as 32 bits
    kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    kernel32.GetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                                ctypes.POINTER(ctypes.c_size_t))
    kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel32.GetPriorityClass.argtypes = (ctypes.c_void_p,)
    return kernel32


def _windows_placement():
    """(affinity mask, priority class) of this process."""
    kernel32 = _kernel32()
    this_process = kernel32.GetCurrentProcess()
    mask, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(this_process, ctypes.byref(mask), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return mask.value, kernel32.GetPriorityClass(this_process)


def _windows_place(mask, priority):
    kernel32 = _kernel32()
    this_process = kernel32.GetCurrentProcess()
    if not (kernel32.SetProcessAffinityMask(this_process, mask) and kernel32.SetPriorityClass(this_process, priority)):
        raise ctypes.WinError(ctypes.get_last_error())


def run_in_background(free_cores):
    """This process (every thread, also those libraries start later) only on the background
    processors, at a priority below the programs in use. For worker processes: for good."""
    mask = background_mask(free_cores)
    if sys.platform == "win32":
        _windows_place(mask, BELOW_NORMAL_PRIORITY_CLASS)
        return
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {n for n in range(mask.bit_length()) if mask >> n & 1})
    if hasattr(os, "nice"):
        os.nice(LOWER_PRIORITY_NICENESS)


@contextlib.contextmanager
def in_background(free_cores):
    """This process in the background while the block runs (indexing), as it was afterwards.
    Windows only (elsewhere a lowered priority cannot be raised again); when Windows refuses,
    the block runs as it is."""
    if sys.platform != "win32":
        yield
        return
    try:
        before = _windows_placement()
        _windows_place(background_mask(free_cores), BELOW_NORMAL_PRIORITY_CLASS)
    except OSError:
        before = None
    try:
        yield
    finally:
        if before is not None:
            _windows_place(*before)
