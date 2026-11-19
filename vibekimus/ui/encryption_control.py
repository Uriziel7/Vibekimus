"""The main window's part of index encryption: unlocking the key at start, the
Index encryption window, and bringing the open library's files in line with the
choice (encrypting them, or writing them readable again)."""
from tkinter import messagebox

from vibekimus import settings
from vibekimus.dpapi import DpapiError
from vibekimus.index_protection import IndexProtection
from vibekimus.key_store import KeyFileError
from vibekimus.ui import theme
from vibekimus.ui.encryption_dialog import TITLE, EncryptionDialog, unlock_with_password

UNREADABLE_KEY = ("The index encryption key cannot be unlocked:\n{error}\n\n"
                  "Encrypted indexes cannot be opened until an exported key is imported "
                  "(Library → Index encryption → Import key…).")


class EncryptionControl:
    """preferences() gives the current Preferences; reload_library() loads the open
    library's files again (after a key became available)."""

    def __init__(self, root, library, preferences, set_status, reload_library):
        self.protection = IndexProtection(settings.KEY_FILE)
        self._root, self._library = root, library
        self._preferences, self._set_status, self._reload_library = preferences, set_status, reload_library

    def start(self):
        """Unlock the key: at once when Windows protects it, by asking when a password does."""
        try:
            self.protection.start(self._preferences().encrypt_indexes)
        except (DpapiError, KeyFileError) as error:
            messagebox.showerror(TITLE, UNREADABLE_KEY.format(error=error))
            return
        if self.protection.needs_password and not unlock_with_password(self._root, self.protection):
            self._set_status("Index files are locked: unlock them in Library → Index encryption",
                             theme.STATUS_BUSY)

    def open_dialog(self):
        EncryptionDialog(self._root, self.protection, self._on_changed)

    def _on_changed(self, reload):
        preferences = self._preferences()
        preferences.encrypt_indexes = self.protection.enabled
        preferences.save()
        if self._library.is_open:
            if reload:
                self._reload_library()
            else:
                self.bring_files_in_line()

    def bring_files_in_line(self):
        """Encrypt the open library's files while encryption is on, or make them
        readable again while it is off."""
        if not self._library.needs_rewrite():
            return
        action = "Encrypting" if self.protection.enabled else "Writing unencrypted"
        self._set_status(f"{action} the index files...", theme.STATUS_BUSY)
        self._root.update_idletasks()
        try:
            self._library.rewrite_files()
        except Exception as error:   # IndexSaveError, an unwritable exclusions or text file…
            self._set_status(f"{action} the index files failed", theme.STATUS_ERROR)
            messagebox.showerror(TITLE, f"{action} the index files failed:\n\n{error}")
            return
        self._set_status(f"{action} the index files: done", theme.STATUS_OK)
