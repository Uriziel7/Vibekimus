"""Downloading a model that is not on this computer yet: a choice of model on first start,
or a question when one is picked in Settings (or the text reader is switched on); then a
progress bar. What was downloaded before a Stop is kept for next time."""
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

from vibekimus.encoders.catalog import MODELS
from vibekimus.model_store import OCR_KEY
from vibekimus.ui import theme
from vibekimus.ui.download_progress import DownloadPanel
from vibekimus.ui.formatting import format_size
from vibekimus.ui.windows import new_window

WRAP = 440
SOURCE = "huggingface.co"
CHOOSE_TITLE = "Choose a search model"
CHOOSE_TEXT = ("Vibekimus needs a model to understand pictures and text. Pick one (the other can be "
               "downloaded later in Settings):")
CONFIRM_TITLE = "Download {label}?"
CONFIRM_TEXT = "{label} is not on this computer yet."
FAILED = "The download failed:\n{error}"
STOPPED = "Stopped. What was downloaded is kept for the next try."


@dataclass(frozen=True)
class Downloadable:
    key: str
    label: str
    description: str


OCR = Downloadable(OCR_KEY, "Text reader (PP-OCRv6 small)", "Reads the text on pictures, for \"quoted\" searches.")


def downloadable(key):
    return next((Downloadable(s.key, s.label, s.description) for s in MODELS if s.key == key), OCR)


class ModelDownloadDialog:
    """choose(...): first start, the user picks one of the models; confirm(...): one model.
    on_ready(key) once it is downloaded; on_cancel() otherwise. Both run on the Tk thread."""

    @classmethod
    def choose(cls, parent, store, on_ready, on_cancel):
        return cls(parent, store, [downloadable(spec.key) for spec in MODELS], CHOOSE_TITLE, CHOOSE_TEXT,
                   on_ready, on_cancel)

    @classmethod
    def confirm(cls, parent, store, key, on_ready, on_cancel):
        item = downloadable(key)
        return cls(parent, store, [item], CONFIRM_TITLE.format(label=item.label),
                   CONFIRM_TEXT.format(label=item.label), on_ready, on_cancel)

    def __init__(self, parent, store, items, title, text, on_ready, on_cancel):
        self._store, self._on_ready, self._on_cancel = store, on_ready, on_cancel
        self._window = window = new_window(parent, title, resizable=False, modal=True, on_close=self._close)
        self._body = body = ttk.Frame(window, padding=theme.PAD * 3)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=text, wraplength=WRAP, justify="left").pack(anchor="w")
        self._choice = tk.StringVar(value=items[0].key)
        for item in items:
            self._item(item, show_choice=len(items) > 1)
        ttk.Label(body, text=f"Downloaded from {SOURCE}; every file is checked against its known fingerprint "
                             f"(SHA-256).", wraplength=WRAP, justify="left", style=theme.MUTED_LABEL).pack(
            anchor="w", pady=(theme.PAD, 0))
        self._download = DownloadPanel(body, WRAP, STOPPED, FAILED)
        self._download.buttons.show(("Download", self._start, theme.PRIMARY_BUTTON), ("Cancel", self._close, None))

    def _item(self, item, show_choice):
        sizes = (f"Download {format_size(self._store.download_bytes(item.key))}, "
                 f"{format_size(self._store.disk_bytes(item.key))} on disk")
        if show_choice:
            ttk.Radiobutton(self._body, text=item.label, value=item.key, variable=self._choice).pack(
                anchor="w", pady=(theme.PAD, 0))
        else:
            ttk.Label(self._body, text=item.label, font=theme.FONT_BOLD).pack(anchor="w", pady=(theme.PAD, 0))
        for line in (item.description, sizes):
            ttk.Label(self._body, text=line, wraplength=WRAP, justify="left", style=theme.MUTED_LABEL).pack(
                anchor="w", padx=(theme.GROUP_GAP, 0))

    def _start(self):
        key = self._choice.get()
        self._download.start(self._store.download_bytes(key),
                             lambda counter, should_stop: self._store.download(key, counter, should_stop),
                             lambda error, cancelled: self._finished(key, error, cancelled))

    def _finished(self, key, error, cancelled):
        if error is None and not cancelled:
            self._window.destroy()
            self._on_ready(key)
            return
        self._download.show_outcome(error, cancelled)
        self._download.buttons.show(("Try again", self._start, theme.PRIMARY_BUTTON), ("Cancel", self._close, None))

    def _close(self):
        if self._download.running:
            self._download.cancel()   # _finished follows; the window stays until then
            return
        self._window.destroy()
        self._on_cancel()
