"""Copying, moving and deleting the files a user selected. No user interface here."""
import ctypes
import os
import shutil
from dataclasses import dataclass, field

from vibekimus.files import plain_path

MAX_ERRORS_LISTED = 5
# Files sent to the Recycle Bin per call: the call itself, not the file, takes most of the time.
TRASH_BATCH = 100
COINIT_APARTMENTTHREADED = 0x2


class TrashUnavailable(RuntimeError):
    """send2trash is not installed, so files cannot be deleted recoverably."""


@dataclass
class FileOperationReport:
    done: list = field(default_factory=list)     # paths as given by the caller
    skipped: int = 0                             # destination already existed
    errors: list = field(default_factory=list)   # "name: reason"

    def summary(self, verb):
        """Human-readable result; `verb` is the past participle, e.g. "copied"."""
        lines = []
        if self.done:
            lines.append(f"✓  {len(self.done)} file(s) {verb} successfully")
        if self.skipped:
            lines.append(f"⚠  {self.skipped} file(s) skipped — already exist at destination")
        if self.errors:
            lines.append(f"✗  {len(self.errors)} file(s) failed:")
            lines.extend(f"    {error}" for error in self.errors[:MAX_ERRORS_LISTED])
            if len(self.errors) > MAX_ERRORS_LISTED:
                lines.append(f"    … and {len(self.errors) - MAX_ERRORS_LISTED} more")
        return "\n".join(lines)


def _for_each(paths, action):
    """Apply action(clean_path) to every path. action returns False to mark a skip."""
    report = FileOperationReport()
    for path in paths:
        try:
            if action(plain_path(path)) is False:
                report.skipped += 1
            else:
                report.done.append(path)
        except Exception as error:
            report.errors.append(f"{os.path.basename(path)}: {error}")
    return report


def _transfer(paths, destination_dir, operation):
    def transfer_one(path):
        if os.path.exists(os.path.join(destination_dir, os.path.basename(path))):
            return False   # never overwrite a file at the destination
        operation(path, destination_dir)
    return _for_each(paths, transfer_one)


def copy_files(paths, destination_dir):
    return _transfer(paths, destination_dir, shutil.copy2)


def move_files(paths, destination_dir):
    return _transfer(paths, destination_dir, shutil.move)


def load_trash():
    """The send-to-Recycle-Bin function. Raises TrashUnavailable if it is missing —
    there is deliberately no fallback to permanent deletion."""
    try:
        from send2trash import send2trash
    except ImportError as error:
        raise TrashUnavailable() from error
    return send2trash


def prepare_worker_thread():
    """Windows: a thread doing these operations needs its own (single-threaded) COM apartment
    for the shell's Recycle Bin. Call once at the start of the thread."""
    if os.name == "nt":
        ctypes.windll.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)


def trash_files(paths, send_to_trash):
    """To the Recycle Bin, TRASH_BATCH files per call; when a call fails, its files are tried
    one by one to tell which ones could not be moved (a file already gone counts as done)."""
    report = FileOperationReport()
    paths = list(paths)
    for start in range(0, len(paths), TRASH_BATCH):
        batch = paths[start:start + TRASH_BATCH]
        try:
            send_to_trash([plain_path(path) for path in batch])
            report.done.extend(batch)
        except Exception:
            one_by_one = _for_each(batch, lambda path: send_to_trash(path) if os.path.exists(path) else None)
            report.done.extend(one_by_one.done)
            report.errors.extend(one_by_one.errors)
    return report
