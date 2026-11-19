"""Finding media files on disk. Reads directory listings only, never file contents."""
import ctypes
import os
from dataclasses import dataclass

from vibekimus import settings

DRIVE_LETTER_COUNT = 26
FIRST_DRIVE_LETTER = ord("A")
WINDOWS_DRIVE_FIXED = 3          # GetDriveTypeW result for an internal disk
REPARSE_TAG_JUNCTION = 0xA0000003
# FILE_ATTRIBUTE_OFFLINE | RECALL_ON_OPEN | RECALL_ON_DATA_ACCESS: the file's content
# is in the cloud (OneDrive etc.) and reading it would download it to the disk.
CLOUD_ONLY_ATTRIBUTES = 0x1000 | 0x40000 | 0x400000


def detect_drive_roots():
    """Locations an all-drives scan covers."""
    if settings.ALL_DRIVES_ROOTS:
        return [root for root in settings.ALL_DRIVES_ROOTS if os.path.isdir(root)]
    if os.name != "nt":
        return [os.path.expanduser("~")]
    roots = []
    try:
        kernel32 = ctypes.windll.kernel32
        mask = kernel32.GetLogicalDrives()
        for i in range(DRIVE_LETTER_COUNT):
            if mask & (1 << i):
                root = f"{chr(FIRST_DRIVE_LETTER + i)}:\\"
                if kernel32.GetDriveTypeW(ctypes.c_wchar_p(root)) == WINDOWS_DRIVE_FIXED:
                    roots.append(root)
    except Exception:
        pass
    return roots


@dataclass(frozen=True)
class ScanRules:
    """What a scan leaves out."""
    prune_system_folders: bool = False
    min_file_bytes: int = 0
    skip_cloud_only_files: bool = False
    skip_dirs: frozenset = frozenset()   # normcase'd absolute folder paths

    @property
    def needs_file_stat(self):
        return bool(self.min_file_bytes) or self.skip_cloud_only_files


class MediaScanner:
    """Walks folders and yields the absolute path of every file with a wanted extension.

    Symlinked folders and junctions are not followed, so nothing is visited twice
    and loops are impossible.
    """

    def __init__(self, rules=ScanRules(), should_stop=None):
        self._rules = rules
        self._should_stop = should_stop or (lambda: False)

    def files(self, roots, extensions):
        stack = [(root, 0) for root in reversed(list(roots))]
        while stack:
            if self._should_stop():
                return
            directory, depth = stack.pop()
            try:
                with os.scandir(directory) as listing:
                    entries = sorted(listing, key=lambda e: e.name.lower())
            except OSError:
                continue   # no permission, drive unplugged, ...
            subdirs = []
            for entry in entries:
                if self._should_stop():   # a folder may hold tens of thousands of files
                    return
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if self._should_enter(entry, depth):
                            subdirs.append(entry.path)
                    elif entry.name.lower().endswith(extensions) and self._accepts_file(entry):
                        yield entry.path
                except OSError:
                    continue
            for subdir in reversed(subdirs):
                stack.append((subdir, depth + 1))

    def _should_enter(self, entry, depth):
        if entry.is_symlink():
            return False
        info = entry.stat(follow_symlinks=False)
        if getattr(info, "st_reparse_tag", 0) == REPARSE_TAG_JUNCTION:
            return False
        name = entry.name.lower()
        rules = self._rules
        if rules.prune_system_folders and (
                name in settings.SYSTEM_ANY_DIRS or (depth == 0 and name in settings.SYSTEM_ROOT_DIRS)):
            return False
        if rules.skip_dirs and os.path.normcase(os.path.abspath(entry.path)) in rules.skip_dirs:
            return False
        return True

    def _accepts_file(self, entry):
        rules = self._rules
        if not rules.needs_file_stat:
            return True
        info = entry.stat()
        if info.st_size < rules.min_file_bytes:
            return False
        if rules.skip_cloud_only_files and getattr(info, "st_file_attributes", 0) & CLOUD_ONLY_ATTRIBUTES:
            return False
        return True
