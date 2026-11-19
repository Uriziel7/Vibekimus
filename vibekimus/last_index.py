"""The index that was open last, opened again when the program starts.

Kept in settings.LAST_INDEX_FILE, encrypted like every other file that names the user's
folders. Only the folder holding the index files is stored; what kind of index it is
(all drives, several folders, a folder index) is read from there again."""
import json
import os

from vibekimus import settings
from vibekimus.encryption import read_file, write_file_atomically
from vibekimus.runlog import safe_print
from vibekimus.scope import scope_of_index_folder

DATA_DIR_KEY = "data_dir"


def _holds_an_index(data_dir):
    """An index of several folders (even before its first scan) or index files of any model."""
    if not os.path.isdir(data_dir):
        return False
    if os.path.exists(os.path.join(data_dir, settings.LIBRARY_FILE)):
        return True
    return any(name.startswith(settings.INDEX_FILE_PREFIX) for name in os.listdir(data_dir))


class LastIndex:
    """remember() the open library; scope() gives it back at the next start."""

    @staticmethod
    def remember(scope):
        """Never stops the library from opening: a failure is only reported."""
        data = json.dumps({DATA_DIR_KEY: os.path.abspath(scope.data_dir)}).encode("utf-8")
        try:
            os.makedirs(os.path.dirname(settings.LAST_INDEX_FILE), exist_ok=True)
            write_file_atomically(settings.LAST_INDEX_FILE, data)
        except Exception as error:   # disk full, or encryption on with the key locked
            safe_print(f"[LIBRARY] Could not remember the open index: {error}")

    @staticmethod
    def scope():
        """The library to open again; None when nothing was open, when it is gone (its
        folder or its index files removed) or when it cannot be read now (damaged, or
        encrypted while the key is locked)."""
        try:
            data_dir = json.loads(read_file(settings.LAST_INDEX_FILE).decode("utf-8"))[DATA_DIR_KEY]
            if not isinstance(data_dir, str) or not _holds_an_index(data_dir):
                return None
            return scope_of_index_folder(data_dir)
        except Exception:
            return None
