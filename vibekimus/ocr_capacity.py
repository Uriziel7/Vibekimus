"""How many processes read the text on pictures (OCR) at a time, decided while they read.

On the graphics card (DirectML) one process runs one OCR model at a time, so several
processes keep the card busier. Each keeps up to settings.OCR_PROCESS_VRAM_GB of graphics
memory and about settings.OCR_CPU_THREADS_PER_PROCESS of the CPU threads indexing may use
(settings.FREE_CPU_CORES are never used). So:
- at most: the CPU threads indexing may use / threads per process (and MAX_OCR_PROCESSES);
- at the start: as many of those as the free graphics memory holds above
  settings.OCR_VRAM_RESERVE_GB;
- while reading (looked at every OCR_VRAM_CHECK_SECONDS): one fewer when the free memory
  drops below the reserve (another program took it); one more when there is room for a whole
  process above the reserve and OCR_GROW_WAIT_SECONDS have passed since the last change (a new
  process takes a while to reach its full memory). The gap between the two keeps the number
  from going up and down.
Reading on the CPU, one process already uses every thread."""
import math
import time
from enum import Enum

from vibekimus import processor, settings
from vibekimus.hardware import free_vram_bytes
from vibekimus.units import BYTES_PER_GB

# Free memory needed to add a process, in processes: its own, and as much again for the
# running ones that have not reached their full memory yet.
GROW_ROOM_PROCESSES = 2


class ProcessChange(Enum):
    FEWER = -1
    SAME = 0
    MORE = 1


class OcrProcessBudget:
    def __init__(self, on_gpu, free_vram=free_vram_bytes, clock=time.monotonic, cpu_threads=None):
        self._on_gpu, self._free_vram, self._clock = on_gpu, free_vram, clock
        self._cpu_threads = cpu_threads or processor.background_threads(settings.FREE_CPU_CORES)
        self._next_check = self._last_change = clock()

    def most(self):
        """The most processes that may read at once (the CPU's share decides)."""
        if not self._on_gpu:
            return 1
        by_cpu = self._cpu_threads // settings.OCR_CPU_THREADS_PER_PROCESS
        return max(1, min(by_cpu, settings.MAX_OCR_PROCESSES))

    def _free_gb(self):
        free = self._free_vram() if self._on_gpu else None
        return None if free is None else free / BYTES_PER_GB

    def to_start(self):
        """As many as the graphics memory free now holds above the reserve (at least 1)."""
        free = self._free_gb()
        if free is None:
            return self.most()
        room = math.floor((free - settings.OCR_VRAM_RESERVE_GB) / settings.OCR_PROCESS_VRAM_GB)
        return max(1, min(self.most(), room))

    def change(self, running):
        """FEWER, SAME or MORE processes than the `running` ones, given the memory free now."""
        now = self._clock()
        if now < self._next_check:
            return ProcessChange.SAME
        self._next_check = now + settings.OCR_VRAM_CHECK_SECONDS
        free = self._free_gb()
        if free is None:
            return ProcessChange.SAME
        if free < settings.OCR_VRAM_RESERVE_GB and running > 1:
            return self._changed(ProcessChange.FEWER, now)
        room_for_one = free >= settings.OCR_VRAM_RESERVE_GB + GROW_ROOM_PROCESSES * settings.OCR_PROCESS_VRAM_GB
        if room_for_one and running < self.most() and now >= self._last_change + settings.OCR_GROW_WAIT_SECONDS:
            return self._changed(ProcessChange.MORE, now)
        return ProcessChange.SAME

    def _changed(self, change, now):
        self._last_change = now
        return change
