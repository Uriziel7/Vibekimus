"""Before anything is downloaded: is there an NVIDIA graphics card the bundled PyTorch
build can use, with a new enough driver? Asks nvidia-smi (installed with the driver)."""
import re
import subprocess
from dataclasses import dataclass

QUERY = ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"]
SUMMARY = ["nvidia-smi"]
DRIVER_CUDA = re.compile(r"CUDA Version:\s*(\d+)\.(\d+)")
TIMEOUT_SECONDS = 20
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)   # no console flashing up from the .exe


@dataclass(frozen=True)
class GpuCheck:
    ok: bool
    name: str      # the card, when one was found
    message: str   # why it cannot be used (empty when ok)


def _version(text):
    return tuple(int(part) for part in text.strip().split("."))


def evaluate(query_output, summary_output, min_compute_capability, cuda):
    """The verdict from nvidia-smi's outputs (None = nvidia-smi could not be run)."""
    if query_output is None:
        return GpuCheck(False, "", "No NVIDIA graphics card (or its driver) was found. Vibekimus needs an NVIDIA "
                                   "graphics card, GeForce RTX 20 / GTX 16 or newer.")
    cards = [line.rsplit(",", 1) for line in query_output.strip().splitlines() if "," in line]
    usable = [(name.strip(), _version(capability)) for name, capability in cards]
    usable.sort(key=lambda card: card[1], reverse=True)
    if not usable:
        return GpuCheck(False, "", "nvidia-smi did not list a graphics card.")
    name, capability = usable[0]
    if capability < min_compute_capability:
        return GpuCheck(False, name, f"{name} is too old for the PyTorch version Vibekimus uses "
                                     f"(it needs a GeForce RTX 20 / GTX 16 or newer).")
    match = DRIVER_CUDA.search(summary_output or "")
    needed_major = _version(cuda)[0]
    if match and int(match.group(1)) < needed_major:
        return GpuCheck(False, name, f"The graphics driver is too old (it supports CUDA {match.group(1)}."
                                     f"{match.group(2)}; CUDA {cuda} is needed). Update the NVIDIA driver and "
                                     f"start Vibekimus again.")
    return GpuCheck(True, name, "")


def run_nvidia_smi(command):
    """nvidia-smi's output, or None when it cannot be run or fails."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
                                creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def check_gpu(lock):
    return evaluate(run_nvidia_smi(QUERY), run_nvidia_smi(SUMMARY), lock.min_compute_capability, lock.cuda)
