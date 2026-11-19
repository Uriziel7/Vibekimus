"""What a library covers — one folder, several folders or all drives — and how its files
are addressed.

The scopes differ in where the index lives, how a file is named inside it and what a
scan leaves out; everything else in the program is the same for all of them.
"""
import json
import os
from abc import ABC, abstractmethod

from vibekimus import settings
from vibekimus.encryption import read_file, write_file_atomically
from vibekimus.files import forward_slashes
from vibekimus.scanner import ScanRules, detect_drive_roots
from vibekimus.units import BYTES_PER_KB

UNSAFE_PATH_MESSAGE = "index file contains a path outside the indexed location"
KIND_KEY = "kind"   # in LIBRARY_FILE: which kind of index (an index of folders, or of all drives)


def _has_parent_reference(path):
    return ".." in forward_slashes(path).split("/")


def in_folder(index_path, folder_prefix):
    """Whether an index path lies inside the folder given by its index-path prefix
    (see LibraryScope.folder_prefix). Case-insensitive, as Windows paths are."""
    return folder_prefix == "" or index_path.casefold().startswith(folder_prefix.casefold() + "/")


def is_safe_rel_path(path):
    """True for a plain relative path that stays inside the indexed folder."""
    if not isinstance(path, str) or not path or "\x00" in path:
        return False
    normalized = forward_slashes(path)
    is_absolute = normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":")
    return not is_absolute and not _has_parent_reference(normalized)


def is_safe_abs_path(path):
    """True for an absolute path without parent references."""
    return (isinstance(path, str) and bool(path) and "\x00" not in path
            and os.path.isabs(path) and not _has_parent_reference(path))


class LibraryScope(ABC):
    """The set of locations a library indexes."""

    name = None                      # what the user called the index (an index of several folders)
    kept_in_indexed_folder = False   # the index lives inside the folder it indexes (older versions)
    folders_editable = True          # folders can be added and removed (Library → Folders of this index)

    @property
    @abstractmethod
    def description(self):
        """Text shown to the user for this scope."""

    @property
    @abstractmethod
    def data_dir(self):
        """Where the index, exclusion list and skipped-file logs are kept."""

    @property
    @abstractmethod
    def scan_roots(self):
        """Folders a scan starts from."""

    @abstractmethod
    def index_path(self, abs_path):
        """Path as stored in the index (forward slashes)."""

    @abstractmethod
    def absolute_path(self, index_path):
        """Inverse of index_path: the real location of an indexed file."""

    @abstractmethod
    def _is_safe(self, index_path):
        """Whether a path read from an index file may be acted on."""

    def folder_prefix(self, folder):
        """How the index names a folder: the start of the index paths of the files in
        it ("" for the whole library), or None when the folder is outside the library."""
        try:
            prefix = self.index_path(os.path.abspath(folder)).rstrip("/")
        except ValueError:   # Windows: a folder on another drive than the library
            return None
        if prefix == ".":
            return ""
        return None if _has_parent_reference(prefix) else prefix

    def validate(self, index_paths):
        """Refuse an index whose paths could point somewhere unexpected — Copy / Move /
        Delete act on these paths."""
        if not all(self._is_safe(path) for path in index_paths):
            raise ValueError(UNSAFE_PATH_MESSAGE)

    def scan_rules(self, skip_small_files):
        """skip_small_files: the caller is looking for a media kind where tiny files
        are noise (icons, thumbnails)."""
        return ScanRules()

    def reachability_check(self):
        """Returns a function telling whether an index path is on a location that is
        connected right now. Entries that are not must not be pruned as 'deleted'."""
        return lambda index_path: True

    def prepare(self):
        """Create whatever the scope needs on disk before it is used."""


class FolderScope(LibraryScope):
    """One folder. The index lives inside it and stores paths relative to it, so the
    folder can be moved or copied together with its index."""

    kept_in_indexed_folder = True

    def __init__(self, folder):
        self._folder = folder

    @property
    def description(self):
        return self._folder

    @property
    def data_dir(self):
        return self._folder

    @property
    def scan_roots(self):
        return [self._folder]

    def index_path(self, abs_path):
        return forward_slashes(os.path.relpath(abs_path, self._folder))

    def absolute_path(self, index_path):
        return os.path.normpath(os.path.join(self._folder, index_path))

    def _is_safe(self, index_path):
        return is_safe_rel_path(index_path)


def same_path(first, second):
    """Whether two paths name the same file or folder (Windows: case-insensitive)."""
    return os.path.normcase(os.path.abspath(first)) == os.path.normcase(os.path.abspath(second))


def same_or_inside(path, folder):
    """Whether `path` is `folder` or lies inside it (Windows: case-insensitive)."""
    path, folder = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(folder))
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


class MultiFolderScope(LibraryScope):
    """Folders anywhere (on several drives too) with one combined index in its own folder
    of the program's data. There is no common folder, so the index stores full paths, and
    nothing is ever written to the indexed folders. Its name and folders are saved next to
    the index (settings.LIBRARY_FILE), so it can be opened again from the saved list."""

    KIND = "folders"

    def __init__(self, name, roots, data_dir):
        self.name = name
        self._roots = [os.path.abspath(root) for root in roots]
        self._data_dir = data_dir

    @staticmethod
    def load(data_dir):
        """The scope saved in data_dir: an index of several folders, or the all-drives index —
        also one kept elsewhere than settings.ALL_DRIVES_INDEX_DIR (the .exe keeps its data in
        another folder than a run from the source, and may open the index of the other).
        Raises OSError / ValueError / LockedError."""
        data = json.loads(read_file(os.path.join(data_dir, settings.LIBRARY_FILE)).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("the index's folder list is damaged")
        name, roots = data.get("name"), data.get("roots")
        if not isinstance(name, str) or not isinstance(roots, list) or not all(isinstance(r, str) for r in roots):
            raise ValueError("the index's folder list is damaged")
        if AllDrivesScope.is_saved_as(data.get(KIND_KEY), name, roots):
            return AllDrivesScope(roots, data_dir)
        return MultiFolderScope(name, roots, data_dir)

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        data = json.dumps({KIND_KEY: self.KIND, "name": self.name, "roots": self._roots}, indent=1).encode("utf-8")
        write_file_atomically(os.path.join(self.data_dir, settings.LIBRARY_FILE), data)

    @property
    def description(self):
        return f"{self.name}: " + ", ".join(self._roots)

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def scan_roots(self):
        return list(self._roots)

    def covers(self, folder):
        """Whether the folder is one of the indexed folders or lies inside one."""
        return any(same_or_inside(folder, root) for root in self._roots)

    def add_root(self, folder):
        """Index this folder too (a folder inside it that was indexed alone is now part of
        it). Returns False when the folder is already covered."""
        if self.covers(folder):
            return False
        self._roots = [root for root in self._roots if not same_or_inside(root, folder)] + [os.path.abspath(folder)]
        self.save()
        return True

    def remove_root(self, folder):
        """Stop indexing this folder; its entries must be dropped by the caller."""
        self._roots = [root for root in self._roots if not same_or_inside(root, folder)]
        self.save()

    def index_path(self, abs_path):
        return forward_slashes(abs_path)

    def absolute_path(self, index_path):
        return os.path.normpath(index_path)

    def _is_safe(self, index_path):
        return is_safe_abs_path(index_path)

    def folder_prefix(self, folder):
        return self.index_path(os.path.abspath(folder)).rstrip("/") if self.covers(folder) else None

    def validate(self, index_paths):
        """Also refuse files outside the index's folders: the list of folders and the
        index are two files, and Copy / Move / Delete must stay within the folders."""
        index_paths = list(index_paths)   # may be a generator; read twice
        super().validate(index_paths)
        folders = tuple(os.path.normcase(root).rstrip(os.sep) + os.sep for root in self.scan_roots)
        for path in index_paths:
            if not (os.path.normcase(os.path.normpath(path)) + os.sep).startswith(folders):
                raise ValueError(UNSAFE_PATH_MESSAGE)

    def reachability_check(self):
        connected = tuple(forward_slashes(root).lower() for root in self._roots if os.path.isdir(root))
        return lambda index_path: index_path.lower().startswith(connected)

    def prepare(self):
        self.save()


class AllDrivesScope(MultiFolderScope):
    """Every fixed drive. Its index is kept in its own place (settings.ALL_DRIVES_INDEX_DIR),
    and a scan leaves out system folders, tiny images and files that are only in the cloud."""

    NAME = "All drives"
    KIND = "all drives"
    folders_editable = False   # it covers every drive

    def __init__(self, roots, data_dir=None):
        """data_dir: where the index is kept, when not in settings.ALL_DRIVES_INDEX_DIR."""
        super().__init__(self.NAME, roots, data_dir)

    @classmethod
    def is_saved_as(cls, kind, name, roots):
        """Whether a LIBRARY_FILE describes an all-drives index. Files written before the kind
        was stored: by its name and its folders, which are whole drives."""
        if kind is not None:
            return kind == cls.KIND
        return name == cls.NAME and bool(roots) and all(os.path.splitdrive(root)[1] in ("\\", "/") for root in roots)

    @classmethod
    def saved(cls):
        """The all-drives scope with the drives it was made with (detected again for an
        index from before they were saved)."""
        try:
            return cls(MultiFolderScope.load(settings.ALL_DRIVES_INDEX_DIR).scan_roots)
        except Exception:
            return cls(detect_drive_roots())

    @property
    def data_dir(self):
        return self._data_dir or settings.ALL_DRIVES_INDEX_DIR

    def folder_prefix(self, folder):
        return self.index_path(os.path.abspath(folder)).rstrip("/")

    def validate(self, index_paths):
        """Any drive: the drives of an older index were not saved and are detected again,
        so one not connected now must not make the whole index unreadable."""
        LibraryScope.validate(self, index_paths)

    def scan_rules(self, skip_small_files):
        min_bytes = settings.ALL_DRIVES_MIN_IMAGE_KB * BYTES_PER_KB if skip_small_files else 0
        return ScanRules(
            prune_system_folders=not settings.ALL_DRIVES_SCAN_SYSTEM_FOLDERS,
            min_file_bytes=min_bytes,
            skip_cloud_only_files=True,
            skip_dirs=frozenset({os.path.normcase(os.path.abspath(self.data_dir))}),
        )


def scope_of_index_folder(data_dir):
    """The library whose index files are kept in data_dir: the all-drives index, an index of
    several folders (its LIBRARY_FILE) or a folder index (older versions). Raises OSError,
    ValueError or LockedError when the list of folders cannot be read."""
    if same_path(data_dir, settings.ALL_DRIVES_INDEX_DIR):
        return AllDrivesScope.saved()
    if os.path.exists(os.path.join(data_dir, settings.LIBRARY_FILE)):
        return MultiFolderScope.load(data_dir)
    return FolderScope(data_dir)
