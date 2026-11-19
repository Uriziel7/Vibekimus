"""What the indexing passes have in common: Indexer (what the controller runs) and
MediaIndexer (the image and video indexers)."""
import gc
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from vibekimus import settings
from vibekimus.embedding_index import IndexSaveError
from vibekimus.indexing.info_backfill import InfoBackfill
from vibekimus.runlog import log_event, safe_print


PERCENT = 100
MIN_ELAPSED_SECONDS = 1e-6   # keeps rates finite for a run that ends at once


class PassState(Enum):
    FINISHED = "finished"            # new files were encoded and the index saved
    STOPPED = "stopped"              # the user stopped it; what was done is saved
    UP_TO_DATE = "up to date"        # a refresh found nothing new
    NOTHING_FOUND = "nothing found"  # a rebuild found no media at all
    FAILED = "failed"                # could not run; see PassOutcome.message


class IndexingUnavailable(RuntimeError):
    """This kind of media cannot be indexed on this installation."""


class IndexingListener:
    """Receives progress from an indexing run. Called from the indexing thread."""

    def on_activity(self, text):
        """A short description of what is happening now (e.g. scanning)."""

    def on_progress(self, percent, text):
        """Share of the current pass that is done, with a one-line summary."""

    def on_pass_done(self, outcome, is_last):
        """One pass ended. is_last is False when another pass follows immediately."""


@dataclass(frozen=True)
class IndexingRun:
    """What the controller gives every pass of one run; each pass takes what it needs."""
    encoder: object                 # encoders.base.MediaEncoder
    options: object                 # indexing.options.IndexingOptions
    should_stop: Callable           # () -> bool: the user pressed Stop
    listener: IndexingListener


@dataclass
class RunStats:
    """Counters of one encoding run."""
    total: int
    done: int = 0
    skipped_long: int = 0
    gpu_seconds: float = 0.0
    started: float = field(default_factory=time.time)

    @property
    def elapsed(self):
        return max(time.time() - self.started, MIN_ELAPSED_SECONDS)

    @property
    def percent(self):
        return PERCENT * self.done / self.total if self.total else 0

    @property
    def gpu_busy_percent(self):
        return PERCENT * self.gpu_seconds / self.elapsed


class AutosaveTimer:
    """Tells when settings.AUTOSAVE_SECONDS have passed since the last save.

    Saving is time-based on purpose: saving every N files would write hundreds of
    gigabytes while indexing a very large library.
    """

    def __init__(self):
        self._last = time.time()

    def due(self):
        if time.time() - self._last < settings.AUTOSAVE_SECONDS:
            return False
        self._last = time.time()
        return True


def chunked(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


class Indexer(ABC):
    """One pass of the controller: keeps one kind of data of the library current."""

    media_name = ""          # "images" / "videos" / "text" — used in messages and the run log
    skipped_log_name = ""    # file listing what could not be read

    def __init__(self, library, run):
        self._library = library
        self._should_stop = run.should_stop
        self._listener = run.listener
        self.failures = []   # (abs_path, reason) of files that could not be read

    @abstractmethod
    def rebuild(self):
        """Everything from scratch. Returns a PassState."""

    @abstractmethod
    def refresh(self):
        """Only what changed. Returns a PassState."""

    @abstractmethod
    def index_summary(self):
        """What the index holds now, in words (for the run log)."""


class MediaIndexer(Indexer):
    """Scans the library for one kind of media and keeps that kind's index current.

    Subclasses say which index they fill, how to find the files and how to turn a
    list of files into embeddings; the rebuild / refresh logic is shared.
    """

    def __init__(self, library, run):
        super().__init__(library, run)
        self._encoder = run.encoder
        self._options = run.options

    @property
    @abstractmethod
    def index(self):
        """The EmbeddingIndex this indexer fills."""

    @property
    @abstractmethod
    def _decode_mode(self):
        """Where this kind of media is decoded (from the options)."""

    @abstractmethod
    def _scan(self):
        """Yield (abs_path, index_path) for every file of this kind in the library."""

    @abstractmethod
    def _read_info(self, abs_path):
        """The media_info record of one file, from its header only."""

    @abstractmethod
    def _encode_files(self, files, is_update):
        """Encode `files` (absolute paths), add them to the index and save it."""

    # ── the two ways to run ──────────────────────────────────────────────────
    def rebuild(self):
        """Index everything from scratch. The previous contents are put back if the
        scan is stopped or finds nothing."""
        self._encoder.activate()
        index = self.index
        previous = index.snapshot()
        index.start_over()
        files = [abs_path for abs_path, _index_path in self._scan()]
        if self._should_stop():
            index.restore(previous)
            return PassState.STOPPED
        if not files:
            index.restore(previous)
            return PassState.NOTHING_FOUND
        safe_print(f"[INDEX] Found {len(files):,} {self.media_name}.")
        return self._encode(files, is_update=False)

    def refresh(self):
        """Add files that are new on disk and drop entries whose file is gone."""
        self._encoder.activate()
        safe_print(f"\n[SCAN] Scanning for {self.media_name} changes...")
        self._listener.on_activity("Scanning folder...")
        index = self.index
        indexed = index.files()
        on_disk, new_files = set(), []
        for abs_path, index_path in self._scan():
            on_disk.add(index_path)
            if index_path not in indexed:
                new_files.append(abs_path)
        if self._should_stop():
            return PassState.STOPPED

        removed = self._prune(on_disk)
        filled = InfoBackfill(index, self._library.scope, self._read_info, self._should_stop,
                              self._listener.on_activity).run()
        if self._should_stop():
            index.save()
            return PassState.STOPPED
        if not new_files:
            if removed or filled:
                index.save()
            safe_print(f"[SCAN] {self.media_name.capitalize()} index is up to date.")
            return PassState.UP_TO_DATE
        safe_print(f"[SCAN] Found {len(new_files):,} new {self.media_name}.")
        log_event(f"{self.media_name.upper()} scan: {len(on_disk):,} on disk, "
                  f"{len(new_files):,} new, {removed:,} index entries removed")
        return self._encode(new_files, is_update=True)

    def _prune(self, on_disk):
        """Drop entries whose file no longer exists. Entries on a location that is
        not connected right now are kept. Returns the number of rows dropped."""
        index = self.index
        file_of = index.format.file_of
        is_reachable = self._library.scope.reachability_check()
        keep = [row for row, key in enumerate(index.keys)
                if file_of(key) in on_disk or not is_reachable(file_of(key))]
        removed = index.keep_rows(keep)
        if removed:
            safe_print(f"[SCAN] Pruned {removed:,} stale entries.")
        return removed

    def _encode(self, files, is_update):
        self._encode_files(files, is_update)
        return PassState.STOPPED if self._should_stop() else PassState.FINISHED

    # ── helpers for subclasses ───────────────────────────────────────────────
    def _add_new(self, keys, rows, info, known):
        """Add the rows whose key is not in `known` yet (their keys join it). Returns how many were added."""
        new_keys, new_rows, new_info = [], [], []
        for key, row, record in zip(keys, rows, info):
            if key not in known:
                known.add(key)
                new_keys.append(key)
                new_rows.append(row)
                new_info.append(record)
        if new_keys:
            self.index.add(new_keys, np.array(new_rows), new_info)
        return len(new_keys)

    def _autosave(self, done_text):
        """The periodic save of what is new. A failure is reported and indexing goes on:
        the save at the end tries again, and ends the run with the error if it fails too."""
        try:
            self.index.checkpoint(partial=True)
        except IndexSaveError as error:
            safe_print(f"\n[INDEX] Auto-save failed: {error}")
            log_event(f"{self.media_name.upper()} auto-save failed: {error}")
            return
        safe_print(f"\n[INDEX] Auto-saved at {done_text}")

    def _report_scan_progress(self, found):
        self._listener.on_activity(f"Scanning... {found:,} files found")

    def _report_progress(self, run, text):
        self._listener.on_progress(run.percent, text)
        safe_print(f"\r[INDEX] {text}", end="")

    def _free_accelerator_memory(self):
        try:
            self._encoder.empty_cache(synchronize=True)
            gc.collect()
        except Exception as error:
            safe_print(f"[INDEX] VRAM cleanup warning (non-fatal): {error}")
