"""Library → Statistics…: what the open index holds, by year, kind of picture, file
format and folder, with the space each takes. Counted in the background."""
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import ttk

from vibekimus.ui import theme
from vibekimus.ui.background import when_done
from vibekimus.ui.formatting import format_size
from vibekimus.ui.widgets import ScrollableFrame
from vibekimus.ui.windows import new_window

GEOMETRY = "640x760"
MIN_WIDTH, MIN_HEIGHT = 420, 360
POLL_MS = 50
LABEL_WRAP = 230           # long folder names wrap within this many pixels
BAR_HEIGHT = 12
ROW_SPACING = 1
BAR_MIN_WIDTH = 2          # a group with files is never drawn empty
COUNTING = "Counting…"
FAILED = "The statistics could not be counted: {error}"
NO_TYPES_NOTE = "Kinds of pictures are counted once the model has loaded."
NO_DUPLICATES_NOTE = "not counted (needs the loaded model, and no indexing running)"


class StatisticsWindow:
    def __init__(self, parent, title, collect):
        """collect(should_stop) -> LibraryStatistics; runs in a worker thread."""
        self._closed = False
        self._window = window = new_window(parent, title, geometry=GEOMETRY, min_size=(MIN_WIDTH, MIN_HEIGHT),
                                           on_close=self.close, close_on_escape=True)
        self._page = ScrollableFrame(window)
        self._page.pack(fill="both", expand=True)
        window.bind("<MouseWheel>", self._page.scroll)   # this page, not the results behind the window
        self._status = ttk.Label(self._page.body, text=COUNTING, style=theme.MUTED_LABEL)
        self._status.pack(anchor="w")
        self._pool = ThreadPoolExecutor(max_workers=1)
        when_done(self._window, self._pool.submit(collect, lambda: self._closed), self._counted, POLL_MS)

    def close(self):
        self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._window.destroy()

    def _counted(self, future):
        error = future.exception()
        if error is not None:
            self._status.config(text=FAILED.format(error=error), foreground=theme.STATUS_ERROR)
            return
        self._status.destroy()
        self._show(future.result())

    # ── contents ─────────────────────────────────────────────────────────────
    def _show(self, stats):
        self._summary(stats)
        self._section("By year (taken, or else last changed)", stats.by_year)
        self._section("Kinds of pictures", stats.by_type, NO_TYPES_NOTE)
        self._section("File formats", stats.by_format)
        self._section("Folders", stats.by_folder)

    def _summary(self, stats):
        lines = [f"Images: {stats.images.files:,}  ·  {format_size(stats.images.bytes)}"]
        if stats.videos.files:
            lines.append(f"Videos: {stats.videos.files:,} ({stats.frames:,} frames)  ·  "
                         f"{format_size(stats.videos.bytes)}")
        if stats.texts_read:
            lines.append(f"Text read on {stats.texts_read:,} pictures, found on {stats.with_text:,}")
        duplicates = stats.duplicates
        if duplicates is None:
            lines.append(f"Duplicates: {NO_DUPLICATES_NOTE}")
        else:
            lines.append(f"Duplicates: {duplicates.copies:,} extra copies in {duplicates.groups:,} groups  ·  "
                         f"{format_size(duplicates.bytes)} to free")
        lines.append(f"Index files: {format_size(stats.index_bytes)}")
        self._heading("Library")
        for line in lines:
            ttk.Label(self._page.body, text=line).pack(anchor="w")

    def _heading(self, text):
        ttk.Label(self._page.body, text=text, font=theme.FONT_HEADING).pack(anchor="w", pady=(theme.PAD * 2,
                                                                                             theme.GAP))

    def _section(self, title, tallies, empty_note=None):
        if not tallies and empty_note is None:
            return
        self._heading(title)
        if not tallies:
            self._page.wrapping_label(self._page.body, empty_note, style=theme.MUTED_LABEL).pack(anchor="w")
            return
        table = ttk.Frame(self._page.body)
        table.pack(fill="x")
        table.columnconfigure(1, weight=1)
        most = max(tally.files for tally in tallies)
        for row, tally in enumerate(tallies):
            ttk.Label(table, text=tally.label, wraplength=LABEL_WRAP, justify="left").grid(
                row=row, column=0, sticky="w", pady=ROW_SPACING)
            _Bar(table, tally.files / most if most else 0).grid(row=row, column=1, sticky="ew", padx=theme.PAD)
            ttk.Label(table, text=f"{tally.files:,}").grid(row=row, column=2, sticky="e")
            ttk.Label(table, text=format_size(tally.bytes), style=theme.MUTED_LABEL).grid(
                row=row, column=3, sticky="e", padx=(theme.PAD, 0))


class _Bar(tk.Canvas):
    """A horizontal bar filling `share` (0–1) of its width."""

    def __init__(self, parent, share):
        super().__init__(parent, height=BAR_HEIGHT, bg=theme.PANEL_BG, highlightthickness=0)
        self._share = share
        self.bind("<Configure>", lambda event: self._draw())

    def _draw(self):
        self.delete("all")
        width = max(BAR_MIN_WIDTH, self.winfo_width() * self._share) if self._share > 0 else 0
        if width:
            self.create_rectangle(0, 0, width, BAR_HEIGHT, fill=theme.ACCENT, width=0)
