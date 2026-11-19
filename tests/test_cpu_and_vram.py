"""How many processes read text (the CPU share and the free graphics memory), and the CPU cores
kept free while indexing."""
import subprocess
import sys

import pytest

from support import GB, FakeModel, rebuild
from vibekimus import processor, settings
from vibekimus.indexing.controller import IMAGES, IndexingController
from vibekimus.indexing.options import IndexingOptions
from vibekimus.ocr_capacity import OcrProcessBudget, ProcessChange

CPU_THREADS = 12
FOUR_CORES = (0b11, 0b1100, 0b110000, 0b11000000)   # two threads each
WINDOWS = sys.platform == "win32"
PROBE = ("import ctypes, sys; sys.path.insert(0, sys.argv[1]); from vibekimus import processor; "
         "processor.run_in_background(2); k = ctypes.windll.kernel32; "
         "k.GetCurrentProcess.restype = ctypes.c_void_p; k.GetPriorityClass.argtypes = (ctypes.c_void_p,); "
         "k.GetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p); "
         "m, s = ctypes.c_size_t(), ctypes.c_size_t(); h = k.GetCurrentProcess(); "
         "k.GetProcessAffinityMask(h, ctypes.byref(m), ctypes.byref(s)); print(m.value, k.GetPriorityClass(h))")


def test_ocr_processes_follow_the_cpu_and_the_free_graphics_memory():
    free_gb, now = [9.5], [0.0]
    budget = OcrProcessBudget(True, free_vram=lambda: free_gb[0] * GB, clock=lambda: now[0], cpu_threads=CPU_THREADS)
    assert budget.most() == 4, "at most what the CPU threads allow (12 → 4)"
    assert budget.to_start() == 2, "started as the free graphics memory holds (9.5 GB → 2)"
    free_gb[0] = 30.0
    assert budget.to_start() == 4
    free_gb[0] = 1.0
    changes = [budget.change(3), budget.change(3)]   # low; the second too soon after the first
    now[0] += settings.OCR_VRAM_CHECK_SECONDS
    changes.append(budget.change(1))   # low, but one is left
    free_gb[0] = 30.0
    now[0] += settings.OCR_VRAM_CHECK_SECONDS
    changes.append(budget.change(1))   # room again, but too soon after the last change
    now[0] += settings.OCR_GROW_WAIT_SECONDS
    changes += [budget.change(1), budget.change(4)]   # room: one more; and later at the most already
    now[0] += settings.OCR_GROW_WAIT_SECONDS
    changes.append(budget.change(4))
    free_gb[0] = settings.OCR_VRAM_RESERVE_GB + 1.0   # above the reserve, no room for a whole process
    now[0] += settings.OCR_GROW_WAIT_SECONDS
    changes.append(budget.change(2))
    assert changes == [ProcessChange.FEWER] + [ProcessChange.SAME] * 3 + [ProcessChange.MORE] + [ProcessChange.SAME] * 3


def test_none_of_this_on_the_cpu_or_when_the_memory_cannot_be_read():
    unknown = OcrProcessBudget(True, free_vram=lambda: None, cpu_threads=CPU_THREADS)
    on_cpu = OcrProcessBudget(False, free_vram=lambda: 0, cpu_threads=CPU_THREADS)
    assert unknown.to_start() == 4 and unknown.change(2) == ProcessChange.SAME
    assert on_cpu.most() == on_cpu.to_start() == 1


def test_cores_kept_free_are_whole_cores_from_the_end_and_one_core_is_always_used(monkeypatch):
    monkeypatch.setattr(processor, "core_masks", lambda: FOUR_CORES)
    assert processor.background_mask(2) == 0b1111
    assert processor.background_mask(10) == 0b11
    assert processor.background_mask(0) == 0b11111111
    assert processor.background_threads(2) == 4


@pytest.mark.skipif(not WINDOWS, reason="process placement is checked on Windows")
def test_a_process_moved_to_the_background_runs_on_those_processors_below_normal_priority(project_root):
    shown = subprocess.run([sys.executable, "-c", PROBE, project_root], capture_output=True, text=True).stdout.split()
    assert shown == [str(processor.background_mask(2)), str(processor.BELOW_NORMAL_PRIORITY_CLASS)]


@pytest.mark.skipif(not WINDOWS, reason="process placement is checked on Windows")
def test_while_it_indexes_the_program_runs_in_the_background_and_afterwards_as_before(details):
    seen = []

    class Watcher:
        def on_activity(self, text):
            pass

        def on_progress(self, percent, text):
            seen.append(processor._windows_placement())

        def on_pass_done(self, outcome, is_last):
            seen.append(processor._windows_placement())
    before = processor._windows_placement()
    IndexingController(details, Watcher()).run(FakeModel(), [rebuild(IMAGES)], IndexingOptions())
    background = (processor.background_mask(settings.FREE_CPU_CORES), processor.BELOW_NORMAL_PRIORITY_CLASS)
    assert seen and all(place == background for place in seen)
    assert processor._windows_placement() == before != background
