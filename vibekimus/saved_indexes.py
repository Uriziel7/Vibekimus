"""The indexes saved on this computer, for the "Open saved index" list: the ones made in
the program (each in its own folder of settings.INDEXES_DIR), the all-drives index, and
indexes kept inside a folder by older versions, which the program remembers once it has
seen them (settings.KNOWN_INDEXES_FILE; "Find on this PC" looks for them on every drive)."""
import json
import os
import re
from dataclasses import dataclass
from enum import Enum

from vibekimus import settings
from vibekimus.encryption import read_file, write_file_atomically
from vibekimus.index_files import ImageIndexFormat, VideoIndexFormat, index_shards, peek_index_count
from vibekimus.scanner import MediaScanner, ScanRules, detect_drive_roots
from vibekimus.scope import AllDrivesScope, MultiFolderScope, same_path, scope_of_index_folder

UNSAFE_NAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
FALLBACK_NAME = "Index"


class IndexPlace(Enum):
    """Where a saved index lives."""
    FOLDER = "folder"            # inside the indexed folder (older versions)
    COLLECTION = "collection"    # its own folder of the program's data
    ALL_DRIVES = "all drives"


@dataclass(frozen=True)
class SavedIndex:
    kind: IndexPlace
    name: str
    data_dir: str
    roots: tuple          # the indexed folders (empty when they cannot be read: encrypted, locked)
    image_file: str       # this model's index files in data_dir
    video_file: str
    modified: float       # newest of them
    size: int             # bytes, both together

    def scope(self):
        return scope_of_index_folder(self.data_dir)

    def counts(self):
        """(images, video frames) in the index, None where it cannot be read now."""
        return tuple(_count(path) for path in (self.image_file, self.video_file))


def _files_of(path):
    """An index file and the partial-save pieces left with it."""
    return ([path] if os.path.exists(path) else []) + index_shards(path)


def _count(path):
    try:
        return sum(peek_index_count(part) for part in _files_of(path))
    except Exception:   # damaged, or encrypted while the key is locked
        return None


def label_of(folder):
    """A short name for a folder: its last part (the whole path for a drive)."""
    return os.path.basename(folder.rstrip("\\/")) or folder


def folder_name(name):
    """A folder name made from what the user called the index."""
    cleaned = UNSAFE_NAME_CHARACTERS.sub(" ", name).strip(" .")
    return cleaned or FALLBACK_NAME


class SavedIndexes:
    """The saved indexes for one model (each model has its own index files)."""

    def __init__(self, model):
        self._model = model

    # ── listing ──────────────────────────────────────────────────────────────
    def list(self):
        """Every saved index that has an index of this model, the latest first."""
        found = []
        for kind, data_dir in self._candidates():
            saved = self._describe(kind, data_dir)
            if saved is not None:
                found.append(saved)
        return sorted(found, key=lambda saved: saved.modified, reverse=True)

    def _candidates(self):
        yield IndexPlace.ALL_DRIVES, settings.ALL_DRIVES_INDEX_DIR
        if os.path.isdir(settings.INDEXES_DIR):
            for entry in sorted(os.scandir(settings.INDEXES_DIR), key=lambda e: e.name.lower()):
                if entry.is_dir():
                    yield IndexPlace.COLLECTION, entry.path
        for folder in self._known_folders():
            yield IndexPlace.FOLDER, folder

    def _describe(self, kind, data_dir):
        image_file = os.path.join(data_dir, ImageIndexFormat(self._model).file_name())
        video_file = os.path.join(data_dir, VideoIndexFormat(self._model).file_name())
        files = _files_of(image_file) + _files_of(video_file)
        if not files:
            return None
        roots, name = self._roots_and_name(kind, data_dir)
        return SavedIndex(kind=kind, name=name, data_dir=data_dir, roots=roots, image_file=image_file,
                          video_file=video_file, modified=max(os.path.getmtime(f) for f in files),
                          size=sum(os.path.getsize(f) for f in files))

    @staticmethod
    def _roots_and_name(kind, data_dir):
        """What the list shows: read as the index will be opened (a folder the program remembers
        may hold an index of several folders or of all drives, made by another copy of it)."""
        try:
            scope = scope_of_index_folder(data_dir)
        except Exception:   # damaged, or encrypted and locked
            return (), AllDrivesScope.NAME if kind == IndexPlace.ALL_DRIVES else os.path.basename(data_dir)
        return tuple(scope.scan_roots), scope.name or label_of(data_dir)

    # ── folder indexes the program has seen ──────────────────────────────────
    def _known_folders(self):
        try:
            folders = json.loads(read_file(settings.KNOWN_INDEXES_FILE).decode("utf-8"))
        except Exception:
            return []
        return [f for f in folders if isinstance(f, str)] if isinstance(folders, list) else []

    def _save_known(self, folders):
        os.makedirs(os.path.dirname(settings.KNOWN_INDEXES_FILE), exist_ok=True)
        write_file_atomically(settings.KNOWN_INDEXES_FILE, json.dumps(folders, indent=1).encode("utf-8"))

    def remember(self, folder):
        """A folder holding an index (of any model) is on the list from now on."""
        folders = self._known_folders()
        if not any(same_path(folder, known) for known in folders):
            self._save_known(folders + [os.path.abspath(folder)])

    def forget(self, folder):
        """Take a folder index off the list (its files stay where they are)."""
        self._save_known([known for known in self._known_folders() if not same_path(folder, known)])

    def find_on_pc(self, should_stop=None, on_found=None):
        """Look on every fixed drive for folder indexes (of any model) and remember them.
        Returns how many were new."""
        before = len(self._known_folders())
        rules = ScanRules(prune_system_folders=True, skip_cloud_only_files=True,
                          skip_dirs=frozenset(os.path.normcase(os.path.abspath(d))
                                              for d in (settings.INDEXES_DIR, settings.ALL_DRIVES_INDEX_DIR)))
        for path in MediaScanner(rules, should_stop).files(detect_drive_roots(), (settings.INDEX_FILE_SUFFIX,)):
            if os.path.basename(path).startswith(settings.INDEX_FILE_PREFIX):
                self.remember(os.path.dirname(path))
                if on_found:
                    on_found(os.path.dirname(path))
        return len(self._known_folders()) - before

    # ── new indexes ──────────────────────────────────────────────────────────
    @staticmethod
    def create(name, roots):
        """A new index of these folders in its own folder of settings.INDEXES_DIR."""
        base = os.path.join(settings.INDEXES_DIR, folder_name(name))
        data_dir, number = base, 1
        while os.path.exists(data_dir):
            number += 1
            data_dir = f"{base} ({number})"
        scope = MultiFolderScope(name, roots, data_dir)
        scope.save()
        return scope
