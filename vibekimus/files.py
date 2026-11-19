"""Paths and files: the forms a path takes (as an index stores it, as Windows opens a deep one,
as Explorer wants it) and writing a file so that a crash never leaves half of it behind.
Standard library only: the .exe's installer uses it before any library is installed."""
import os

WINDOWS_LONG_PATH_PREFIX = "\\\\?\\"
WINDOWS_NT_PATH_PREFIX = "\\??\\"
TEMP_SUFFIX = ".tmp"


def forward_slashes(path):
    """The path with / between its parts, as indexes store paths."""
    return path.replace("\\", "/")


def long_path(path):
    """Windows: prefix the path so files deeper than 260 characters can be opened."""
    if os.name == "nt":
        path = os.path.normpath(path)
        if not path.startswith(WINDOWS_LONG_PATH_PREFIX):
            path = WINDOWS_LONG_PATH_PREFIX + path
    return path


def plain_path(path):
    """Inverse of long_path: a normal absolute path for Explorer, shutil and the Recycle Bin."""
    clean = path.replace(WINDOWS_LONG_PATH_PREFIX, "").replace(WINDOWS_NT_PATH_PREFIX, "")
    return os.path.normpath(os.path.abspath(clean))


def write_atomically(path, data):
    """Write bytes to a temporary file and swap it in, so a crash mid-save never leaves a
    truncated file behind."""
    temp_path = path + TEMP_SUFFIX
    with open(temp_path, "wb") as f:
        f.write(data)
    os.replace(temp_path, path)
