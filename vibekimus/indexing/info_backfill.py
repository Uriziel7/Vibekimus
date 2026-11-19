"""Filling in the file info (date, size, dimensions) that indexes made by an older
version do not have. Only file headers are read — nothing is decoded or encoded."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from vibekimus import settings
from vibekimus.runlog import safe_print

REPORT_EVERY_FILES = 1000


class InfoBackfill:
    def __init__(self, index, scope, read_info, should_stop, on_activity):
        """read_info(abs_path) returns a media_info record for one file."""
        self._index = index
        self._scope = scope
        self._read_info = read_info
        self._should_stop = should_stop
        self._on_activity = on_activity

    def run(self):
        """Fill what is missing; returns how many rows got their info."""
        rows_by_file = self._rows_by_file()
        if not rows_by_file:
            return 0
        safe_print(f"[INDEX] Reading file details for {len(rows_by_file):,} files indexed by an older version")
        filled = 0
        executor = ThreadPoolExecutor(max_workers=settings.LOAD_WORKERS)
        try:
            futures = {executor.submit(self._read_info, self._scope.absolute_path(file)): file
                       for file in rows_by_file}
            for done, future in enumerate(as_completed(futures), start=1):
                if self._should_stop():
                    break
                rows = rows_by_file[futures[future]]
                self._index.set_info(rows, [future.result()] * len(rows))
                filled += len(rows)
                if done % REPORT_EVERY_FILES == 0:
                    self._on_activity(f"Reading file details... {done:,}/{len(rows_by_file):,}")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return filled

    def _rows_by_file(self):
        file_of = self._index.format.file_of
        keys = self._index.keys
        rows_by_file = defaultdict(list)
        for row in self._index.rows_missing_info():
            rows_by_file[file_of(keys[row])].append(row)
        return rows_by_file
