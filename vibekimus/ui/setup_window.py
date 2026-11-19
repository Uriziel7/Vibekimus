"""The .exe's first start: says what has to be downloaded (and from where, and how much),
checks the graphics card and the free disk space, and installs the libraries with a
progress bar. Standard library only: nothing else is installed yet."""
import tkinter as tk
from tkinter import ttk

from vibekimus import settings
from vibekimus.ui import theme
from vibekimus.ui.download_progress import DownloadPanel
from vibekimus.ui.formatting import format_size

TITLE = "Vibekimus — setup"
WRAP = 460
FIRST_HEADING = "Vibekimus needs a few libraries first"
UPDATE_HEADING = "This version of Vibekimus needs updated libraries"
SOURCES = "pypi.org and download.pytorch.org"
STOPPED = "Stopped. What was downloaded is kept: start Vibekimus again to continue."
FAILED = "The download failed:\n{error}\n\nCheck the internet connection and try again."
NO_ROOM = "Not enough free disk space: {needed} is needed on the drive of\n{folder}"


class SetupWindow:
    """run() returns True once the libraries are installed (False: closed or impossible)."""

    def __init__(self, lock, installer, gpu):
        self._installer = installer
        self._wheels = installer.missing()
        self._installed = False
        self._closing = False   # the window was closed during a download: it goes once that stops
        self._root = root = tk.Tk()
        root.title(TITLE)
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.configure(bg=theme.PANEL_BG)
        root.resizable(False, False)
        theme.apply_styles()
        theme.apply_icon(root)
        theme.apply_dark_title_bar(root)
        self._body = body = ttk.Frame(root, padding=theme.PAD * 3)
        body.pack(fill="both", expand=True)
        updating = len(self._wheels) < len(lock.wheels)
        ttk.Label(body, text=UPDATE_HEADING if updating else FIRST_HEADING, font=theme.FONT_HEADING).pack(anchor="w")
        self._paragraph(self._explanation(lock, updating))
        self._paragraph(f"Graphics card: {gpu.name} ✓" if gpu.ok else gpu.message,
                        colour=theme.FG if gpu.ok else theme.STATUS_ERROR)
        room = installer.has_room(self._wheels)
        if gpu.ok and not room:
            self._paragraph(NO_ROOM.format(needed=format_size(installer.space_needed(self._wheels)),
                                           folder=settings.RUNTIME_DIR), colour=theme.STATUS_ERROR)
        self._download = DownloadPanel(body, WRAP, STOPPED, FAILED)
        if gpu.ok and room:
            self._download.buttons.show(("Download and start", self._start, theme.PRIMARY_BUTTON),
                                        ("Cancel", root.destroy, None))
        else:
            self._download.buttons.show(("Close", root.destroy, None))

    def run(self):
        self._root.mainloop()
        return self._installed

    def _explanation(self, lock, updating):
        size = format_size(sum(w.size for w in self._wheels))
        if updating:
            return f"{len(self._wheels)} libraries to download: {size} from {SOURCES}."
        return (f"It runs the search models on the graphics card with PyTorch (CUDA) and {len(lock.wheels) - 1} other "
                f"libraries. They are downloaded once: {size} from {SOURCES}, about "
                f"{format_size(lock.installed_bytes)} on disk in\n{settings.RUNTIME_DIR}\n\n"
                f"Nothing about you or your files is sent anywhere. Every file is checked against its known "
                f"fingerprint (SHA-256) before it is used.")

    def _paragraph(self, text, colour=theme.FG):
        ttk.Label(self._body, text=text, wraplength=WRAP, justify="left", foreground=colour).pack(
            anchor="w", pady=(theme.PAD, 0))

    def _close(self):
        if self._download.running:
            self._closing = True
            self._download.cancel()
            return
        self._root.destroy()

    def _start(self):
        self._download.start(sum(w.size for w in self._installer.missing()), self._installer.install, self._finished)

    def _finished(self, error, cancelled):
        if error is None and not cancelled:
            self._installed = True
        if self._installed or self._closing:
            self._root.destroy()
            return
        self._download.show_outcome(error, cancelled)
        close = ("Close", self._root.destroy, None)
        if cancelled:
            self._download.buttons.show(close)
        else:
            self._download.buttons.show(("Try again", self._start, theme.PRIMARY_BUTTON), close)
