"""Graphics memory detection and the batch sizes derived from it."""
import functools
import os
import re
import subprocess
import sys

from vibekimus.runlog import safe_print
from vibekimus.runtime.gpu_check import run_nvidia_smi
from vibekimus.units import BYTES_PER_GB, BYTES_PER_MB

# Free memory of the first card as the driver sees it: every process's, also DirectML's,
# which torch.cuda.mem_get_info() does not count (measured: it stayed at 30.1 GB while OCR
# took 3.5 GB). About 30 ms a reading.
FREE_VRAM_QUERY = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
# Share of system RAM treated as graphics memory on Apple Silicon (unified memory).
MPS_USABLE_RAM_SHARE = 0.8

# (minimum VRAM in GB, batch size), largest first. The thresholds sit 1 GB under the
# nominal card sizes (32, 24, 20, 16, 12, 8 GB) because cards report slightly less.
IMAGE_BATCH_BY_VRAM_GB = ((31, 384), (23, 256), (19, 192), (15, 160), (11, 128), (7, 64), (4, 32))
IMAGE_BATCH_MINIMUM = 16
VIDEO_BATCH_BY_VRAM_GB = ((31, 64), (23, 48), (19, 32), (15, 24), (11, 16), (7, 12), (4, 8))
VIDEO_BATCH_MINIMUM = 4
VIDEO_BATCH_UNKNOWN_VRAM = 8


def _vram_from_torch():
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        import psutil
        return int(psutil.virtual_memory().total * MPS_USABLE_RAM_SHARE)
    return None


def _vram_from_windows_wmic():
    output = subprocess.check_output("wmic path win32_VideoController get AdapterRAM", shell=True)
    values = [int(s) for s in re.findall(r"\d+", output.decode("utf-8", errors="ignore"))]
    return max(values) if values else None


def _vram_from_linux_sysfs():
    import glob
    paths = glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")
    if not paths:
        return None
    with open(paths[0], "r") as f:
        return int(f.read().strip())


def _vram_from_macos_profiler():
    output = subprocess.check_output("system_profiler SPDisplaysDataType", shell=True).decode("utf-8")
    match = re.search(r"VRAM.*?(\d+)\s*(MB|GB)", output, re.IGNORECASE)
    if not match:
        return None
    unit = BYTES_PER_GB if match.group(2).upper() == "GB" else BYTES_PER_MB
    return int(match.group(1)) * unit


def _platform_detector():
    if os.name == "nt":
        return _vram_from_windows_wmic
    if sys.platform.startswith("linux"):
        return _vram_from_linux_sysfs
    if sys.platform == "darwin":
        return _vram_from_macos_profiler
    return None


@functools.lru_cache(maxsize=None)
def system_vram_bytes():
    """Graphics memory in bytes, or None if it cannot be detected. Detected once."""
    for detector in (_vram_from_torch, _platform_detector()):
        if detector is None:
            continue
        try:
            vram = detector()
        except Exception:
            continue
        if vram:
            return vram
    return None


def free_vram_bytes():
    """Free graphics memory now (NVIDIA cards), or None when it cannot be read."""
    output = run_nvidia_smi(FREE_VRAM_QUERY)
    try:
        return int(output.split()[0]) * BYTES_PER_MB
    except (AttributeError, IndexError, ValueError):
        return None


def _batch_for(vram_bytes, table, minimum):
    vram_gb = vram_bytes / BYTES_PER_GB
    for min_gb, batch in table:
        if vram_gb >= min_gb:
            return batch
    return minimum


@functools.lru_cache(maxsize=None)
def image_batch_size():
    vram = system_vram_bytes()
    if vram is None:
        safe_print(f"[CONFIG] Could not detect VRAM. Defaulting to Batch Size: {IMAGE_BATCH_MINIMUM}")
        return IMAGE_BATCH_MINIMUM
    batch = _batch_for(vram, IMAGE_BATCH_BY_VRAM_GB, IMAGE_BATCH_MINIMUM)
    safe_print(f"[CONFIG] Detected VRAM: {vram / BYTES_PER_GB:.2f} GB — Batch Size: {batch}")
    return batch


@functools.lru_cache(maxsize=None)
def video_batch_size():
    vram = system_vram_bytes()
    if vram is None:
        return VIDEO_BATCH_UNKNOWN_VRAM
    return _batch_for(vram, VIDEO_BATCH_BY_VRAM_GB, VIDEO_BATCH_MINIMUM)
