"""Narrowing and ordering search results by file facts: date, folder, size, resolution."""
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from vibekimus.media_info import best_date, local_timestamp
from vibekimus.scope import in_folder

DATE_PATTERN = re.compile(r"^\s*(\d{4})(?:[-./](\d{1,2}))?(?:[-./](\d{1,2}))?\s*$")
MONTHS_PER_YEAR = 12


class SortOrder(Enum):
    RELEVANCE = "relevance"
    NEWEST = "newest"
    OLDEST = "oldest"
    LARGEST_FILE = "largest file"
    MOST_PIXELS = "highest resolution"
    # Duplicate groups: most bytes freed by deleting the copies; single results: largest file.
    SPACE_FREED = "space freed"

    def apply(self, results):
        """Reorder results (best match first when RELEVANCE). Results whose value is
        unknown go last."""
        if self is SortOrder.RELEVANCE:
            return results
        value, newest_first = {
            SortOrder.NEWEST: (lambda r: r.date, True),
            SortOrder.OLDEST: (lambda r: r.date, False),
            SortOrder.LARGEST_FILE: (lambda r: r.bytes, True),
            SortOrder.MOST_PIXELS: (lambda r: r.pixels, True),
            SortOrder.SPACE_FREED: (lambda r: r.bytes, True),
        }[self]
        known = [r for r in results if value(r) > 0]
        unknown = [r for r in results if value(r) <= 0]
        return sorted(known, key=value, reverse=newest_first) + unknown


def parse_date(text, end=False):
    """Seconds since 1970 for a typed date: 2023, 2023-06 or 2023-06-15 (also with
    dots or slashes). end=True gives the moment the period ends instead of starts.
    Empty text gives None; anything else unreadable raises ValueError."""
    if not text or not text.strip():
        return None
    match = DATE_PATTERN.match(text)
    if not match:
        raise ValueError(f"not a date: {text!r} (use 2023, 2023-06 or 2023-06-15)")
    year, month, day = (int(part) if part else None for part in match.groups())
    start = datetime(year, month or 1, day or 1)   # raises ValueError for impossible dates
    if not end:
        return local_timestamp(start)
    if day:
        stop = datetime.fromordinal(start.toordinal() + 1)
    elif month:
        stop = datetime(year + month // MONTHS_PER_YEAR, month % MONTHS_PER_YEAR + 1, 1)
    else:
        stop = datetime(year + 1, 1, 1)
    return local_timestamp(stop)


@dataclass(frozen=True)
class ResultFilters:
    date_from: float = None     # seconds since 1970, inclusive
    date_to: float = None       # seconds since 1970, exclusive
    min_pixels: int = 0
    min_bytes: int = 0
    folder: str = None          # only files inside this folder (absolute path)

    @property
    def active(self):
        return (self.date_from is not None or self.date_to is not None
                or any((self.min_pixels, self.min_bytes, self.folder)))

    def row_mask(self, index, scope):
        """Which rows of an EmbeddingIndex pass, as a boolean array (None = all).
        A row whose needed value is unknown does not pass."""
        if not self.active:
            return None
        info = index.info
        mask = np.ones(len(info), dtype=bool)
        dates = best_date(info)
        if self.date_from is not None:   # an unknown date (stored as UNKNOWN) never passes
            mask &= (dates >= self.date_from) & (dates > 0)
        if self.date_to is not None:
            mask &= (dates < self.date_to) & (dates > 0)
        if self.min_pixels:
            mask &= info["width"].astype(np.int64) * info["height"] >= self.min_pixels
        if self.min_bytes:
            mask &= info["bytes"] >= self.min_bytes
        if self.folder:
            mask &= self._folder_mask(index, scope)
        return mask

    def _folder_mask(self, index, scope):
        prefix = scope.folder_prefix(self.folder)
        if prefix is None:   # a folder outside the library holds none of its files
            return np.zeros(len(index.keys), dtype=bool)
        file_of = index.format.file_of
        return np.fromiter((in_folder(file_of(key), prefix) for key in index.keys),
                           dtype=bool, count=len(index.keys))
