"""Statistics of the open library: how many files and how much space, by year, file
format, folder and kind of picture; the text read on the pictures; the space that
duplicate copies take; the size of the index itself. Excluded files are left out."""
import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from vibekimus import media_types
from vibekimus.duplicates import DuplicateFinder, space_freed, split_groups
from vibekimus.media_info import best_date
from vibekimus.scope import same_or_inside

FORMATS_SHOWN = 10
FOLDERS_SHOWN = 12
OTHER_FORMATS = "Other formats"
OTHER_FOLDERS = "Other folders"
UNKNOWN_YEAR = "Date unknown"
NO_EXTENSION = "(no extension)"


@dataclass(frozen=True)
class Tally:
    label: str
    files: int
    bytes: int     # of the files whose size is known


@dataclass(frozen=True)
class DuplicateSummary:
    groups: int
    copies: int    # files beyond the one kept in each group
    bytes: int     # what deleting those copies frees


@dataclass(frozen=True)
class LibraryStatistics:
    images: Tally
    videos: Tally
    frames: int
    by_year: tuple          # Tally per year, oldest first
    by_format: tuple        # most files first
    by_folder: tuple        # most files first
    by_type: tuple          # kinds of picture (photos, memes…); empty without a model
    texts_read: int
    with_text: int
    duplicates: DuplicateSummary = None    # None: not counted (no model, or indexing)
    index_bytes: int = 0


@dataclass(frozen=True)
class _Files:
    """One entry per indexed file (a video once, not once per frame)."""
    paths: list             # index paths
    info: np.ndarray        # their media_info records


class StatisticsCollector:
    def __init__(self, library):
        self._library = library

    def collect(self, type_labels_of=None, duplicates_device=None, count_duplicates=False,
                should_stop=lambda: False):
        """type_labels_of(index view): per image row, the position of its kind in
        MEDIA_TYPES (None: kinds not counted). count_duplicates: compare every image with
        every other one (on duplicates_device when given)."""
        library = self._library
        image_view, video_view = library.images.view(), library.videos.view()
        image_rows = self._kept_rows(image_view)
        images, videos = self._files(image_view, image_rows), self._video_files(video_view)
        both = _Files(images.paths + videos.paths, np.concatenate([images.info, videos.info]))
        type_labels = type_labels_of(image_view) if type_labels_of is not None and image_view.keys else None
        return LibraryStatistics(
            images=_tally("Images", images.info), videos=_tally("Videos", videos.info),
            frames=len(self._kept_rows(video_view)),
            by_year=self._by_year(both), by_format=self._by_format(both), by_folder=self._by_folder(both),
            by_type=self._by_type(type_labels, image_view, image_rows),
            texts_read=len(library.texts), with_text=library.texts.with_text(),
            duplicates=self._duplicates(duplicates_device, should_stop) if count_duplicates else None,
            index_bytes=self._index_bytes())

    # ── what is counted ──────────────────────────────────────────────────────
    def _kept_rows(self, view):
        return np.flatnonzero(view.kept(self._library.exclusions)).tolist()

    @staticmethod
    def _files(view, rows):
        return _Files([view.keys[row] for row in rows], view.info[rows])

    def _video_files(self, view):
        """The first frame of every video stands for the file."""
        first_row = {}
        for row in self._kept_rows(view):
            first_row.setdefault(view.format.file_of(view.keys[row]), row)
        return _Files(list(first_row), view.info[list(first_row.values())])

    # ── groupings ────────────────────────────────────────────────────────────
    @staticmethod
    def _by_year(files):
        """In local time, as the date filter reads dates."""
        years = [str(datetime.fromtimestamp(date).year) if date > 0 else UNKNOWN_YEAR
                 for date in best_date(files.info)]
        tallies = _grouped(years, files.info)
        known = sorted(year for year in tallies if year != UNKNOWN_YEAR)
        return tuple(tallies[year] for year in known) + ((tallies[UNKNOWN_YEAR],) if UNKNOWN_YEAR in tallies else ())

    @staticmethod
    def _by_format(files):
        formats = [os.path.splitext(path)[1].lstrip(".").upper() or NO_EXTENSION for path in files.paths]
        return _largest(_grouped(formats, files.info), FORMATS_SHOWN, OTHER_FORMATS)

    def _by_folder(self, files):
        folders = [self._top_folder(path) for path in files.paths]
        return _largest(_grouped(folders, files.info), FOLDERS_SHOWN, OTHER_FOLDERS)

    def _top_folder(self, index_path):
        """The library folder a file is in and, below it, the first folder of its path."""
        scope = self._library.scope
        path = scope.absolute_path(index_path)
        root = next((root for root in scope.scan_roots if same_or_inside(path, root)), None)
        if root is None:
            return os.path.dirname(path)
        parts = os.path.relpath(path, root).split(os.sep)
        return os.path.join(root, parts[0]) if len(parts) > 1 else root

    @staticmethod
    def _by_type(type_labels, view, rows):
        if type_labels is None:
            return ()
        labels, info = type_labels[rows], view.info[rows]
        kinds = enumerate(media_types.MEDIA_TYPES)
        tallies = (_tally(kind.label, info[labels == position]) for position, kind in kinds)
        return tuple(sorted((tally for tally in tallies if tally.files), key=lambda tally: -tally.files))

    def _duplicates(self, device, should_stop):
        report = DuplicateFinder(self._library).find(self._library.model.same_picture, device, should_stop)
        groups = split_groups(report.results)
        return DuplicateSummary(report.groups, len(report.to_remove), sum(space_freed(group) for group in groups))

    def _index_bytes(self):
        library = self._library
        files = [library.images.path, library.videos.path, library.texts.path]
        files += library.images.shards() + library.videos.shards()
        return sum(os.path.getsize(path) for path in files if path and os.path.exists(path))


def _tally(label, info):
    sizes = info["bytes"]
    return Tally(label, len(info), int(sizes[sizes > 0].sum()))


def _grouped(labels, info):
    """{label: Tally} of files labelled one by one."""
    rows = {}
    for row, label in enumerate(labels):
        rows.setdefault(label, []).append(row)
    return {label: _tally(label, info[members]) for label, members in rows.items()}


def _largest(tallies, shown, other_label):
    """The `shown` groups with the most files, the rest added up as one."""
    ordered = sorted(tallies.values(), key=lambda tally: (-tally.files, tally.label))
    kept, rest = ordered[:shown], ordered[shown:]
    if rest:
        kept.append(Tally(other_label, sum(t.files for t in rest), sum(t.bytes for t in rest)))
    return tuple(kept)
