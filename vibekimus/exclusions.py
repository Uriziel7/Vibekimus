"""User-defined rules for paths that must be left out of the index and the results:
text patterns typed by hand, and folders picked in the folder dialog."""
import json
import os

from vibekimus import settings
from vibekimus.encryption import read_file, write_file_atomically
from vibekimus.files import forward_slashes
from vibekimus.runlog import safe_print
from vibekimus.scope import in_folder

PATTERNS_KEY = "excluded_patterns"
FOLDERS_KEY = "excluded_folders"


class ExclusionsNotSaved(RuntimeError):
    """The exclusions could not be written (or their file must not be overwritten)."""


class Exclusions:
    """A path is excluded when it contains any pattern (case-sensitive substring) or
    lies inside any excluded folder. Folders are kept as index-path prefixes
    (LibraryScope.folder_prefix), so they move together with a library folder."""

    def __init__(self):
        self._patterns = set()
        self._folders = set()
        self._file = None
        self._load_error = None   # why the file could not be read; it is then never overwritten

    def __bool__(self):
        return bool(self._patterns or self._folders)

    def __len__(self):
        return len(self._patterns) + len(self._folders)

    def sorted(self):
        """The text patterns."""
        return sorted(self._patterns)

    def folders(self):
        """The excluded folders, as index-path prefixes."""
        return sorted(self._folders)

    def matches(self, index_path):
        if not self:
            return False
        normalized = forward_slashes(index_path)
        return (any(pattern in normalized for pattern in self._patterns)
                or any(in_folder(normalized, folder) for folder in self._folders))

    def add(self, pattern):
        """Returns True if the pattern was new."""
        return self._add_to(self._patterns, pattern)

    def remove(self, pattern):
        self._remove_from(self._patterns, pattern)

    def add_folder(self, folder_prefix):
        """Returns True if the folder was new."""
        return self._add_to(self._folders, folder_prefix)

    def remove_folder(self, folder_prefix):
        self._remove_from(self._folders, folder_prefix)

    def _add_to(self, entries, entry):
        """Raises ExclusionsNotSaved (and adds nothing)."""
        if entry in entries:
            return False
        entries.add(entry)
        try:
            self.save()
        except ExclusionsNotSaved:
            entries.discard(entry)
            raise
        return True

    def _remove_from(self, entries, entry):
        """Raises ExclusionsNotSaved (and removes nothing)."""
        if entry not in entries:
            return
        entries.discard(entry)
        try:
            self.save()
        except ExclusionsNotSaved:
            entries.add(entry)
            raise

    @property
    def is_readable(self):
        """False when the file exists but could not be read (damaged, or encrypted and locked)."""
        return self._load_error is None

    def load(self, data_dir):
        """Read the rules kept in `data_dir` (none if there is no file)."""
        self._file = os.path.join(data_dir, settings.EXCLUSIONS_FILE)
        self._patterns, self._folders = set(), set()
        self._load_error = None
        if not os.path.exists(self._file):
            return
        try:
            data = json.loads(read_file(self._file).decode("utf-8"))
            self._patterns = _strings(data.get(PATTERNS_KEY))
            self._folders = _strings(data.get(FOLDERS_KEY))
            safe_print(f"[EXCLUSIONS] Loaded {len(self._patterns)} pattern(s), {len(self._folders)} folder(s)")
        except Exception as e:
            self._load_error = e
            safe_print(f"[EXCLUSIONS] Load error: {e}")

    def save(self):
        """Raises ExclusionsNotSaved. A file that could not be read is never overwritten:
        the rules in it would be lost."""
        if not self._file:
            return
        if self._load_error is not None:
            raise ExclusionsNotSaved(f"The exclusions file could not be read ({self._load_error}), so it is "
                                     f"not overwritten: {self._file}")
        try:
            data = {PATTERNS_KEY: self.sorted(), FOLDERS_KEY: self.folders()}
            write_file_atomically(self._file, json.dumps(data, indent=2).encode("utf-8"))
        except Exception as error:
            raise ExclusionsNotSaved(f"The exclusions could not be saved: {error}") from error
        safe_print(f"[EXCLUSIONS] Saved {len(self._patterns)} pattern(s), {len(self._folders)} folder(s)")


def _strings(value):
    """The non-empty strings of a list read from the file."""
    return {item for item in value if isinstance(item, str) and item} if isinstance(value, list) else set()
