"""Library → Index encryption: turning it on and off, and how the key is unlocked
(the Windows account, or a password at start); changing the password, exporting
the key for another computer and importing it there. Every choice applies at once."""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from vibekimus.key_store import KeyFileError, Protection
from vibekimus.ui import theme
from vibekimus.ui.password_dialog import ask_password, ask_until_right
from vibekimus.ui.windows import new_window

WRAP = 440
TITLE = "Index encryption"
EXPLANATION = (
    "Encrypts what the program writes about your files: the indexes (file names, folders, what the pictures "
    "look like), the text read from pictures and the exclusions. A copied or stolen index file is then useless "
    "without the key. Indexes are encrypted the next time they are saved; the open library at once.\n\n"
    "Keep an exported copy of the key: without the key, encrypted indexes can only be made again by indexing.")
WINDOWS_CHOICE = ("My Windows account (no password)",
                  "Unlocked when you log in to Windows. Another account or computer cannot open the files. "
                  "After reinstalling Windows, use an exported key.")
PASSWORD_CHOICE = ("A password, asked when the program starts",
                   "Works on any computer with the same password. A forgotten password cannot be recovered.")
DISK_NOTE = ("Older unencrypted copies may stay on an SSD for a while; for a stolen computer, "
             "full-disk encryption (BitLocker) is the stronger protection.")
EXPORT_PROMPT = ("Choose a password for the exported key. It is needed to import the key on another "
                 "computer; keep the file and the password safe and apart.")
IMPORT_WARNING = ("Importing replaces the key this program uses now. Indexes encrypted with the current key "
                  "will no longer open (export it first if you still need them).\n\nImport a key?")
UNLOCK_PROMPT = "The index files are encrypted. Enter the password of the key."
KEY_FILE_TYPES =[("Vibekimus key", "*.json")]


class EncryptionDialog:
    """protection: index_protection.IndexProtection. on_changed(reload) runs after
    every change; reload=True after a key import (files may open now)."""

    def __init__(self, parent, protection, on_changed):
        self._protection = protection
        self._on_changed = on_changed
        self._window = window = new_window(parent, TITLE, resizable=False, modal=True, close_on_escape=True)
        body = ttk.Frame(window, padding=theme.PAD * 2)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=EXPLANATION, wraplength=WRAP, justify="left").pack(anchor="w")
        self._enabled = tk.BooleanVar(value=protection.enabled)
        ttk.Checkbutton(body, text="Encrypt index files", variable=self._enabled,
                        command=self._on_enabled_click).pack(anchor="w", pady=(theme.PAD, 0))
        ttk.Label(body, text="How the key is unlocked", font=theme.FONT_HEADING).pack(
            anchor="w", pady=(theme.PAD, theme.GAP))
        self._protection_choice = tk.StringVar(value=protection.protection or Protection.WINDOWS)
        self._radios = [self._choice(body, value, *texts)
                        for value, texts in ((Protection.WINDOWS, WINDOWS_CHOICE),
                                             (Protection.PASSWORD, PASSWORD_CHOICE))]
        self._build_buttons(body)
        self._status = ttk.Label(body, text="", wraplength=WRAP)
        self._status.pack(anchor="w", pady=(theme.PAD, 0))
        ttk.Label(body, text=DISK_NOTE, wraplength=WRAP, style=theme.MUTED_LABEL, justify="left").pack(
            anchor="w", pady=(theme.PAD, 0))
        closing = ttk.Frame(body)
        closing.pack(fill="x", pady=(theme.PAD, 0))
        ttk.Button(closing, text="Close", command=window.destroy).pack(side="right")
        self._refresh()

    def _choice(self, parent, value, label, description):
        radio = ttk.Radiobutton(parent, text=label, value=value, variable=self._protection_choice,
                                command=self._on_protection_click)
        radio.pack(anchor="w")
        ttk.Label(parent, text=description, wraplength=WRAP, style=theme.MUTED_LABEL, justify="left").pack(
            anchor="w", padx=(theme.GROUP_GAP, 0), pady=(0, theme.GAP))
        return radio

    def _build_buttons(self, parent):
        row = ttk.Frame(parent)
        row.pack(anchor="w", pady=(theme.GAP, 0))
        self._unlock_button = ttk.Button(row, text="Unlock…", command=self._unlock)
        self._change_button = ttk.Button(row, text="Change password…", command=self._change_password)
        self._export_button = ttk.Button(row, text="Export key…", command=self._export)
        self._import_button = ttk.Button(row, text="Import key…", command=self._import)
        for button in (self._unlock_button, self._change_button, self._export_button, self._import_button):
            button.pack(side="left", padx=(0, theme.GAP))

    # ── state ────────────────────────────────────────────────────────────────
    def _refresh(self):
        protection = self._protection
        has_key, unlocked = protection.protection is not None, protection.is_unlocked
        self._enabled.set(protection.enabled)
        self._protection_choice.set(protection.protection or Protection.WINDOWS)
        for radio in self._radios:
            radio.state(["!disabled"] if unlocked else ["disabled"])
        self._set_enabled(self._unlock_button, protection.needs_password)
        self._set_enabled(self._change_button, unlocked and protection.protection == Protection.PASSWORD)
        self._set_enabled(self._export_button, unlocked)
        self._status.config(text=self._status_text(has_key), foreground=theme.STATUS_OK
                            if protection.enabled and unlocked else theme.STATUS_BUSY)

    def _status_text(self, has_key):
        protection = self._protection
        if protection.needs_password:
            return "The key is locked: click Unlock… and enter its password."
        if not protection.enabled:
            return "Off: index files are written unencrypted." + (" (A key exists.)" if has_key else "")
        how = "your Windows account" if protection.protection == Protection.WINDOWS else "a password at start"
        return f"On: index files are encrypted; the key is unlocked by {how}."

    @staticmethod
    def _set_enabled(button, enabled):
        button.state(["!disabled"] if enabled else ["disabled"])

    def _changed(self, reload=False):
        self._on_changed(reload)
        self._refresh()

    def _show_error(self, error):
        messagebox.showerror(TITLE, str(error), parent=self._window)

    # ── actions ──────────────────────────────────────────────────────────────
    def _on_enabled_click(self):
        try:
            if self._enabled.get():
                self._protection.enable()
            else:
                self._protection.disable()
        except Exception as error:
            self._show_error(error)
        self._changed()

    def _on_protection_click(self):
        try:
            if self._protection_choice.get() == Protection.WINDOWS:
                self._protection.protect_with_windows()
            else:
                password = self._new_password("Protect the key with a password")
                if password is not None:
                    self._protection.protect_with_password(password)
        except Exception as error:
            self._show_error(error)
        self._refresh()

    def _change_password(self):
        password = self._new_password("New password for the key")
        if password is not None:
            self._protection.protect_with_password(password)
            self._refresh()

    def _new_password(self, title):
        return ask_password(self._window, title, "The password is asked every time the program starts.",
                            confirm=True)

    def _unlock(self):
        if unlock_with_password(self._window, self._protection):
            self._changed(reload=True)

    def _export(self):
        path = filedialog.asksaveasfilename(parent=self._window, title="Export key", defaultextension=".json",
                                            initialfile="vibekimus key.json", filetypes=KEY_FILE_TYPES)
        if not path:
            return
        password = ask_password(self._window, "Export key", EXPORT_PROMPT, confirm=True)
        if password is not None:
            self._protection.export_key(path, password)
            messagebox.showinfo(TITLE, f"The key was exported to:\n{path}", parent=self._window)

    def _import(self):
        if not messagebox.askyesno(TITLE, IMPORT_WARNING, parent=self._window):
            return
        path = filedialog.askopenfilename(parent=self._window, title="Import key", filetypes=KEY_FILE_TYPES)
        if not path:
            return
        try:
            imported = ask_until_right(self._window, "Import key", "The password the key was exported with.",
                                       lambda password: self._protection.import_key(path, password))
        except KeyFileError as damaged:
            self._show_error(damaged)
            return
        if imported:
            self._changed(reload=True)


def unlock_with_password(parent, protection):
    """Ask for the key's password until it is right. False if cancelled."""
    return ask_until_right(parent, "Unlock index files", UNLOCK_PROMPT, protection.unlock)
