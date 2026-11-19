"""The readers of the text pass: OCR engines taking (key, path) jobs from one queue and
putting (key, text, problem) results on another.

The program runs them in worker processes (ProcessReadingPool): in one process the
graphics card runs one OCR model at a time (DirectML), so several processes keep it busier,
and each process has its own models (settings.OCR_PROCESS_VRAM_GB). Stop ends the processes
at once; their graphics memory goes with them. retire_one() lets one process finish the
pictures it is reading and end (the card running out of memory), so nothing is lost, and
add_one() starts another on the same queue (memory free again). A process that crashes
loses only the pictures it was reading. Tests run the same engines in threads of this process
(ThreadReadingPool) with a stand-in engine.

This module imports only the standard library at the top: a worker process of the .exe
loads it before the downloaded libraries are importable (multiprocessing hands the parent's
sys.path over first)."""
import multiprocessing
import os
import queue
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

DRAFT_SIDE = 1024            # pictures are decoded at least this large (OCR shrinks big ones anyway)
UNREADABLE = "cannot be opened"
NO_MORE_JOBS = None          # one per engine after the jobs


def start_engines(jobs, results, make_reader, engines, threads, should_stop):
    """Engine threads reading jobs until each meets NO_MORE_JOBS or should_stop()."""
    workers = [threading.Thread(target=_engine, args=(jobs, results, make_reader, threads, should_stop),
                                daemon=True) for _ in range(engines)]
    for worker in workers:
        worker.start()
    return workers


def _engine(jobs, results, make_reader, threads, should_stop):
    from vibekimus.imaging import open_image
    try:
        reader = make_reader(threads)
    except Exception as error:
        results.put((None, None, f"Text reading could not start: {error}"))
        return
    try:
        while not should_stop():
            job = jobs.get()
            if job is NO_MORE_JOBS:
                return
            key, path = job
            image = open_image(path, draft_size=DRAFT_SIDE)
            if image is None:
                results.put((key, None, UNREADABLE))
                continue
            try:
                results.put((key, reader.read(image), None))
            except Exception as error:
                results.put((key, None, str(error)))
    finally:
        reader.close()


def _queued(jobs, engines, make_queue):
    work = make_queue()
    for job in jobs:
        work.put(job)
    for _ in range(engines):
        work.put(NO_MORE_JOBS)
    return work


class ReadingPool(ABC):
    """Groups of engines reading the jobs given to start(); what the text pass talks to."""

    def __init__(self):
        self._results = None   # a queue of (key, text, problem)

    @abstractmethod
    def start(self, jobs):
        """Begin reading these (key, path) jobs."""

    @abstractmethod
    def stop(self):
        """End the reading now."""

    @abstractmethod
    def running(self):
        """Groups of engines still taking pictures."""

    @abstractmethod
    def _alive(self):
        """Whether any engine is still running (also one finishing its pictures)."""

    def next_result(self, timeout):
        """(key, text, problem), or None when nothing came within the timeout."""
        try:
            return self._results.get(timeout=timeout)
        except queue.Empty:
            return None

    def finished(self):
        """Every engine has ended and every result was taken."""
        return not self._alive() and self._results.empty()

    def retire_one(self):
        """Let one group finish its pictures and end. False: there is none to let go."""
        return False

    def add_one(self):
        """One more group on the same jobs (where the pool can grow)."""


class ThreadReadingPool(ReadingPool):
    """One group of engines in threads of this process (tests: make_reader is a stand-in)."""

    def __init__(self, make_reader, engines, threads):
        super().__init__()
        self._make_reader, self._engines, self._threads = make_reader, engines, threads
        self._stopped = threading.Event()
        self._results = queue.Queue()
        self._workers = []

    def start(self, jobs):
        work = _queued(jobs, self._engines, queue.Queue)
        self._workers = start_engines(work, self._results, self._make_reader, self._engines, self._threads,
                                      self._stopped.is_set)

    def _alive(self):
        return any(worker.is_alive() for worker in self._workers)

    def running(self):
        return 1 if self._alive() else 0

    def stop(self):
        self._stopped.set()


@dataclass(frozen=True)
class _Worker:
    process: object       # multiprocessing.Process
    retire: object        # multiprocessing.Event: finish the pictures being read, then end

    @property
    def reading(self):
        return self.process.is_alive() and not self.retire.is_set()

    @property
    def winding_down(self):
        return self.process.is_alive() and self.retire.is_set()


class ProcessReadingPool(ReadingPool):
    """Engines in worker processes, `engines` threads in each."""

    def __init__(self, processes, engines, threads):
        super().__init__()
        self._count, self._engines, self._threads = processes, engines, threads
        self._workers = []
        self._work = self._context = None

    def start(self, jobs):
        self._context = multiprocessing.get_context("spawn")
        self._work = _queued(jobs, self._count * self._engines, self._context.Queue)
        self._work.cancel_join_thread()   # leaving the program must not wait for jobs nobody takes
        self._results = self._context.Queue()
        for _ in range(self._count):
            self._start_process()

    def _start_process(self):
        from vibekimus import advanced_options
        retire = self._context.Event()
        process = self._context.Process(target=_process_main, daemon=True,
                                        args=(self._work, self._results, self._engines, self._threads,
                                              advanced_options.current(), retire))
        process.start()
        self._workers.append(_Worker(process, retire))

    def add_one(self):
        """One more process on the same queue. Its engines end like the others: each on one
        NO_MORE_JOBS put after every job (spares left by retired processes do no harm)."""
        for _ in range(self._engines):
            self._work.put(NO_MORE_JOBS)
        self._start_process()

    def _alive(self):
        return any(worker.process.is_alive() for worker in self._workers)

    def running(self):
        """Processes still taking pictures."""
        return sum(worker.reading for worker in self._workers)

    def retire_one(self):
        """Let one process finish the pictures it is reading and end, giving its graphics memory
        back. False when only one is left or another one is still finishing."""
        reading = [worker for worker in self._workers if worker.reading]
        if len(reading) <= 1 or any(worker.winding_down for worker in self._workers):
            return False
        reading[-1].retire.set()
        return True

    def stop(self):
        """End the processes now: nothing waits for the pictures being read, and their models
        (and graphics memory) go with them."""
        for worker in self._workers:
            if worker.process.is_alive():
                worker.process.kill()


def _process_main(jobs, results, engines, threads, setting_values, retire):
    """A worker process: the program's settings as they are now, then the engines (until the
    jobs run out or it is asked to retire)."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:   # the .exe has no console
            setattr(sys, name, open(os.devnull, "w"))
    from vibekimus import advanced_options, processor, settings
    advanced_options.apply(setting_values)
    try:
        processor.run_in_background(settings.FREE_CPU_CORES)   # before any library starts its threads
    except OSError:
        pass   # the text is still read, only without the cores kept free
    from vibekimus.text_reading import TextReader
    for worker in start_engines(jobs, results, TextReader, engines, threads, retire.is_set):
        worker.join()
    results.close()
    results.join_thread()   # every result is handed over before the process ends
    os._exit(0)   # the models go with the process (freeing DirectML models one by one breaks the rest)
