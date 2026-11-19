"""Asking for a password: once to unlock, twice (with a minimum length) to set one."""
from tkinter import ttk

from vibekimus import settings
from vibekimus.key_store import WrongPasswordError
from vibekimus.ui import theme
from vibekimus.ui.windows import new_window

WRAP = 360
MASK = "•"
FIELD_CHARS = 36
TOO_SHORT = "Use at least {count} characters."
NOT_THE_SAME = "The two passwords are not the same."


class PasswordDialog:
    """Modal. `value` is the password after OK, None after Cancel. With confirm=True
    the password is typed twice and must be long enough. `error` is shown at once
    (e.g. "Wrong password." when asking again)."""

    def __init__(self, parent, title, prompt, confirm=False, error=""):
        self.value = None
        self._confirm = confirm
        self._window = window = new_window(parent, title, resizable=False, close_on_escape=True)
        body = ttk.Frame(window, padding=theme.PAD * 2)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=prompt, wraplength=WRAP, justify="left").pack(anchor="w")
        self._first = self._field(body, "Password")
        self._second = self._field(body, "Repeat the password") if confirm else None
        self._error = ttk.Label(body, text=error, foreground=theme.STATUS_ERROR, wraplength=WRAP)
        self._error.pack(anchor="w", pady=(theme.GAP, 0))
        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=(theme.PAD, 0))
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="right")
        ttk.Button(buttons, text="OK", command=self._ok, style=theme.PRIMARY_BUTTON).pack(
            side="right", padx=(0, theme.GAP))
        window.bind("<Return>", lambda event: self._ok())
        self._first.focus_set()
        window.grab_set()
        window.wait_window()

    @staticmethod
    def _field(parent, label):
        ttk.Label(parent, text=label, style=theme.MUTED_LABEL).pack(anchor="w", pady=(theme.PAD, 0))
        entry = ttk.Entry(parent, show=MASK, width=FIELD_CHARS)
        entry.pack(fill="x")
        return entry

    def _ok(self):
        password = self._first.get()
        if self._confirm and len(password) < settings.MIN_PASSWORD_CHARS:
            self._error.config(text=TOO_SHORT.format(count=settings.MIN_PASSWORD_CHARS))
        elif self._confirm and password != self._second.get():
            self._error.config(text=NOT_THE_SAME)
        elif password:
            self.value = password
            self._window.destroy()


def ask_password(parent, title, prompt, confirm=False, error=""):
    """The password typed, or None if cancelled."""
    return PasswordDialog(parent, title, prompt, confirm, error).value


def ask_until_right(parent, title, prompt, try_password):
    """Ask for a password until try_password(password) takes it; a wrong one (it raises
    key_store.WrongPasswordError) is asked for again with the reason shown. Other errors
    pass through. False if cancelled."""
    error = ""
    while True:
        password = ask_password(parent, title, prompt, error=error)
        if password is None:
            return False
        try:
            try_password(password)
            return True
        except WrongPasswordError as wrong:
            error = str(wrong)
