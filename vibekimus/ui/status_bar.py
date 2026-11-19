"""The status line at the bottom of the main window: what is happening, a progress bar while
something runs, what the open index holds, and which model runs on which device."""
from tkinter import ttk

from vibekimus.ui import theme

PROGRESS_LENGTH = 180
STARTING = "Starting…"
DEVICE_PENDING = "…"
DETERMINATE, INDETERMINATE = "determinate", "indeterminate"


def counts_text(counts):
    """What the index holds (a library.LibraryCounts), in a few words."""
    videos = f"{counts.videos:,} videos ({counts.frames:,} frames)"
    if counts.images and counts.frames:
        text = f"{counts.images:,} images | {videos}"
    elif counts.images:
        text = f"{counts.images:,} images indexed"
    elif counts.frames:
        text = videos
    else:
        text = ""
    if counts.texts_read:
        text += f" | text read on {counts.texts_read:,} ({counts.with_text:,} with text)"
    return text


class StatusBar:
    def __init__(self, root):
        bar = ttk.Frame(root, style=theme.STATUS_BAR, padding=(theme.PAD, theme.GAP))
        bar.pack(side="bottom", fill="x")
        options = dict(background=theme.BG)
        self._device = ttk.Label(bar, text=DEVICE_PENDING, foreground=theme.ACCENT_HOVER, **options)
        self._device.pack(side="right")
        self._counts = ttk.Label(bar, text="", foreground=theme.FG_MUTED, **options)
        self._counts.pack(side="right", padx=theme.GROUP_GAP)
        self._progress = ttk.Progressbar(bar, mode=DETERMINATE, length=PROGRESS_LENGTH)   # shown while something runs
        self._status = ttk.Label(bar, text=STARTING, anchor="w", **options)
        self._status.pack(side="left", fill="x", expand=True, padx=(0, theme.GROUP_GAP))

    @property
    def text(self):
        return self._status.cget("text")

    def set_status(self, text, colour=theme.FG):
        self._status.config(text=text, foreground=colour)

    def set_progress(self, percent, text):
        """Indexing progress: the bar, and the text in the status line. 0 hides the bar."""
        self._progress["value"] = percent
        self._show_bar(percent > 0)
        if text:
            self.set_status(text, theme.STATUS_BUSY)

    def clear_progress(self):
        self._progress.stop()
        self.set_progress(0, "")

    def show_busy(self):
        """A bar that only shows something is going on (a search)."""
        self._progress.config(mode=INDETERMINATE)
        self._show_bar(True)
        self._progress.start()

    def hide_busy(self):
        self._progress.stop()
        self._progress.config(mode=DETERMINATE)
        self._show_bar(False)

    def show_device(self, text):
        self._device.config(text=text)

    def show_counts(self, counts):
        self._counts.config(text=counts_text(counts))

    def _show_bar(self, visible):
        if visible:
            self._progress.pack(side="right", after=self._counts)
        else:
            self._progress.pack_forget()
