"""Turns what the indexing thread reports into status text, progress and dialogs."""
from dataclasses import dataclass
from tkinter import messagebox
from typing import Callable

from vibekimus.indexing.base import IndexingListener, PassState
from vibekimus.indexing.controller import IMAGES, TEXT, VIDEOS
from vibekimus.ui import theme


@dataclass(frozen=True)
class MediaWording:
    """How results are worded for one media kind; each function takes LibraryCounts."""
    finished_status: str
    finished_message: Callable
    saved: Callable


WORDING = {
    IMAGES: MediaWording(
        finished_status="Indexing Complete",
        finished_message=lambda counts: f"Index complete.\nTotal images: {counts.images:,}",
        saved=lambda counts: f"{counts.images:,} images",
    ),
    TEXT: MediaWording(
        finished_status="Text on pictures read",
        finished_message=lambda counts: (f"Text read on {counts.texts_read:,} pictures.\n"
                                         f"{counts.with_text:,} of them have text.\n\n"
                                         f'Search for exact text in "quotes".'),
        saved=lambda counts: f"the text of {counts.texts_read:,} pictures",
    ),
    VIDEOS: MediaWording(
        finished_status="Video indexing complete",
        finished_message=lambda counts: (f"Video index complete.\n"
                                         f"{counts.videos:,} videos | {counts.frames:,} frames"),
        saved=lambda counts: f"{counts.frames:,} frames from {counts.videos:,} videos",
    ),
}


class IndexingReporter(IndexingListener):
    """Called from the indexing thread; every UI change is handed to the Tk thread."""

    def __init__(self, window):
        self._window = window

    def on_activity(self, text):
        self._window.schedule(lambda: self._window.set_status(text, theme.STATUS_BUSY))

    def on_progress(self, percent, text):
        self._window.schedule(lambda: self._window.set_progress(percent, text))

    def on_pass_done(self, outcome, is_last):
        self._window.schedule(lambda: self._show_outcome(outcome, is_last))

    # ── Tk thread from here on ───────────────────────────────────────────────
    def _show_outcome(self, outcome, is_last):
        window = self._window
        window.set_progress(0, "")
        window.refresh_stats()
        if not is_last:
            return   # the pass that follows reports when everything is done
        wording = WORDING[outcome.media]
        counts = window.library.counts()
        state = outcome.state
        if state is PassState.FINISHED:
            window.set_status(wording.finished_status, theme.STATUS_OK)
            messagebox.showinfo("Done", wording.finished_message(counts))
            window.repeat_search()
        elif state is PassState.STOPPED:
            window.set_status(f"Stopped. Saved {wording.saved(counts)}.", theme.STATUS_STOPPED)
            window.repeat_search()
        elif state is PassState.UP_TO_DATE:
            window.set_status("Up to date", theme.STATUS_OK)
        elif state is PassState.NOTHING_FOUND:
            window.set_status(f"No {outcome.media} found", theme.STATUS_BUSY)
        else:
            window.set_status("Indexing failed", theme.STATUS_ERROR)
            messagebox.showerror("Indexing Failed", outcome.message)
