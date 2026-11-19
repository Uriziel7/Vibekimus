"""The in-memory index of one media kind, and how it is kept on disk."""
import os
import threading
from dataclasses import dataclass

import numpy as np

from vibekimus.encryption import LockedError
from vibekimus.index_files import index_shards, next_shard_path, peek_index_count, remove_shards
from vibekimus.media_info import as_info_array, is_missing, unknown_info
from vibekimus.runlog import safe_print


class IndexSaveError(RuntimeError):
    """The index could not be saved (disk full, key locked, a larger index file this one
    did not come from). What was indexed stays in memory."""


@dataclass(frozen=True)
class IndexView:
    """What one index holds at one moment, its keys, embeddings and info rows always
    matching. An indexing run adds keys first and their embeddings later (see
    EmbeddingIndex), so a reader that took the keys and the arrays separately could get
    lists of different lengths; every reader works on a view instead."""
    format: object
    keys: list
    embeddings: object
    info: object

    def view(self):
        return self

    def kept(self, exclusions):
        """Per row: whether its file is left in by the exclusions (a boolean array)."""
        if not exclusions:
            return np.ones(len(self.keys), dtype=bool)
        file_of = self.format.file_of
        return np.fromiter((not exclusions.matches(file_of(key)) for key in self.keys), dtype=bool,
                           count=len(self.keys))

    def rows_of_files(self, files):
        """{file: its row numbers} for those of `files` that have rows."""
        file_of, rows = self.format.file_of, {}
        for row, key in enumerate(self.keys):
            file = file_of(key)
            if file in files:
                rows.setdefault(file, []).append(row)
        return rows


@dataclass
class IndexSnapshot:
    """Contents of an index at one moment, to put back if a rebuild is abandoned."""
    keys: list
    embeddings: object
    info: object


class EmbeddingIndex:
    """Keys (what was indexed), one embedding row per key and one info record per
    key (date, size, dimensions — see media_info).

    Rows added while indexing are first collected in a pending list and merged into
    the main arrays only before a save or a search: concatenating after every batch
    would copy the whole array each time.
    """

    def __init__(self, index_format):
        self.format = index_format
        self.path = None
        self.keys = []
        self.embeddings = None
        self.info = unknown_info(0)
        self._pending = []        # (embeddings, info) pairs not merged yet
        self._saved_rows = 0      # rows already on disk (main file + partial pieces)
        self._row_of_key = None   # key -> row number, built on first lookup
        self._origin = None       # the file the contents come from (loaded or saved), see _may_replace
        self._lock = threading.RLock()   # guards add/flush/save against a live search

    # ── state ────────────────────────────────────────────────────────────────
    @property
    def has_data(self):
        return self.embeddings is not None and len(self.keys) > 0

    @property
    def is_searchable(self):
        return self.has_data or bool(self._pending)

    def files(self):
        """Set of media files that have at least one row."""
        return {self.format.file_of(key) for key in self.keys}

    def _replace(self, keys, embeddings, info):
        self.keys = keys
        self.embeddings = embeddings
        self.info = info
        self._pending = []
        self._row_of_key = None

    def clear(self):
        """Empty, and no longer the contents of any file (another library is opened)."""
        with self._lock:
            self._replace([], None, unknown_info(0))
            self._origin = None

    def start_over(self):
        """Empty, to be indexed again from nothing: still the index of its file, so the
        finished run replaces that file even when it has fewer entries."""
        with self._lock:
            self._replace([], None, unknown_info(0))

    def snapshot(self):
        copy = self.embeddings.copy() if self.embeddings is not None else None
        return IndexSnapshot(self.keys[:], copy, self.info.copy())

    def restore(self, snapshot):
        with self._lock:
            self._replace(snapshot.keys, snapshot.embeddings, snapshot.info)

    # ── lookup ───────────────────────────────────────────────────────────────
    def embedding_of(self, key):
        """The stored embedding of one key, or None if it is not in the index."""
        with self._lock:
            self.flush()
            if self._row_of_key is None:
                self._row_of_key = {k: row for row, k in enumerate(self.keys)}
            row = self._row_of_key.get(key)
            return None if row is None else self.embeddings[row]

    def view(self):
        """The rows searchable now, as an IndexView (rows still being added are left out)."""
        with self._lock:
            self.flush()
            rows = 0 if self.embeddings is None else min(len(self.embeddings), len(self.keys), len(self.info))
            embeddings = self.embeddings
            if embeddings is not None and len(embeddings) != rows:
                embeddings = embeddings[:rows] if rows else None   # the same array object otherwise: caches key on it
            return IndexView(self.format, self.keys[:rows], embeddings, self.info[:rows])

    def rows_missing_info(self):
        """Row numbers whose info is not known yet (made by an older version)."""
        with self._lock:
            self.flush()
            return np.flatnonzero(is_missing(self.info)).tolist()

    def set_info(self, rows, records):
        with self._lock:
            self.info[rows] = as_info_array(records)

    # ── loading ──────────────────────────────────────────────────────────────
    def load(self, path, validate_files):
        """Replace the contents with the index stored at `path` (main file + pieces).

        validate_files(files) must raise if a stored path is not acceptable; nothing
        is replaced in that case.
        """
        keys, embeddings, info = self.format.read_merged(path)
        validate_files({self.format.file_of(key) for key in keys})
        with self._lock:
            self._replace(keys, embeddings, info)
            self._saved_rows = len(keys)
            self.path = self._origin = path

    # ── growing ──────────────────────────────────────────────────────────────
    def begin_run(self):
        """Call before an indexing run: everything in memory is already on disk."""
        with self._lock:
            self._pending = []
            self._saved_rows = len(self.keys)

    def add(self, keys, features, records):
        """Append rows: their keys, embeddings and info records (tuples or array)."""
        with self._lock:
            self.keys.extend(keys)
            self._pending.append((np.asarray(features), as_info_array(records)))
            self._row_of_key = None

    def flush(self):
        """Merge the pending rows into the main arrays."""
        with self._lock:
            if not self._pending:
                return
            pending = self._pending
            try:
                stacked = np.concatenate([features for features, _info in pending], axis=0)
                embeddings = stacked if self.embeddings is None else np.concatenate([self.embeddings, stacked], axis=0)
                info = np.concatenate([self.info] + [info for _features, info in pending])
            except MemoryError:
                # The rows stay pending: their keys must keep matching their embeddings.
                safe_print(f"[ERROR] Out of memory consolidating {self.format.kind} embeddings — "
                           f"the library may be too large for available RAM.")
                raise
            self.embeddings, self.info, self._pending = embeddings, info, []

    # ── shrinking ────────────────────────────────────────────────────────────
    def keep_rows(self, row_numbers):
        """Keep only the given rows. Returns how many were dropped."""
        with self._lock:
            self.flush()
            dropped = len(self.keys) - len(row_numbers)
            if dropped > 0:
                embeddings = self.embeddings[row_numbers] if (self.embeddings is not None and row_numbers) else None
                self._replace([self.keys[i] for i in row_numbers], embeddings, self.info[row_numbers])
            return dropped

    def remove_files(self, files):
        """Drop every row that belongs to one of `files`. Returns how many were dropped."""
        file_of = self.format.file_of
        return self.keep_rows([i for i, key in enumerate(self.keys) if file_of(key) not in files])

    # ── saving ───────────────────────────────────────────────────────────────
    def checkpoint(self, partial=False):
        """Flush pending rows and save, as one step a concurrent search cannot interleave.
        Raises IndexSaveError."""
        with self._lock:
            self.flush()
            self.save(partial=partial)

    def save(self, partial=False):
        """Save the index. Raises IndexSaveError.

        partial=True (the periodic auto-save) writes only the rows added since the last
        save into a small extra file, so a long run does not rewrite a multi-gigabyte
        index every few minutes. A normal save writes the complete index and removes
        those pieces; an index left without rows removes its files. A file holding more
        entries is replaced only when the contents came from it (see _may_replace).
        """
        with self._lock:
            if not self.path:
                return
            # Keys may be ahead of the embedding array while rows are still
            # pending — save only the part all three agree on.
            rows = 0 if self.embeddings is None else min(len(self.keys), len(self.embeddings), len(self.info))
            try:
                if partial:
                    self._save_new_rows(rows)
                elif rows:
                    self._save_all(rows)
                elif self._origin == self.path:
                    self._remove_files()
            except IndexSaveError:
                raise
            except Exception as error:
                raise IndexSaveError(f"The {self.format.kind} index could not be saved: {error}") from error

    def _save_all(self, rows):
        if not self._may_replace(rows):
            raise IndexSaveError(f"The {self.format.kind} index file has more entries than the {rows:,} indexed "
                                 f"now and was not loaded (damaged, or locked), so it was not replaced: "
                                 f"{self.path}")
        self.format.write(self.path, self.keys[:rows], self.embeddings[:rows], self.info[:rows])
        self._saved_rows, self._origin = rows, self.path
        remove_shards(self.path)
        safe_print(f"[{self.format.log_tag}] Saved {rows:,} {self.format.entry_noun} to {self.path}")

    def _remove_files(self):
        """Every entry is gone: no index file is better than one of files that no longer exist."""
        if os.path.exists(self.path):
            os.remove(self.path)
        remove_shards(self.path)
        self._saved_rows = 0
        safe_print(f"[{self.format.log_tag}] No {self.format.entry_noun} left; removed {self.path}")

    def _save_new_rows(self, rows):
        start = min(self._saved_rows, rows)
        if rows > start:
            self.format.write(next_shard_path(self.path), self.keys[start:rows],
                              self.embeddings[start:rows], self.info[start:rows])
            self._saved_rows = rows

    def _may_replace(self, rows):
        """Whether a complete save may replace the file at self.path. Always, when the
        contents came from it: then fewer entries mean files were pruned or deleted. A
        file this index did not come from (it failed to load) is replaced only by as
        many entries or more, so a library that could not be read is not overwritten by
        a smaller one; one that cannot be decrypted, never."""
        if self._origin == self.path or not os.path.exists(self.path):
            return True
        try:
            existing = peek_index_count(self.path)
        except LockedError:
            return False
        except Exception:
            return True   # damaged beyond reading: replaced
        return existing <= rows

    def shards(self):
        return index_shards(self.path) if self.path else []
