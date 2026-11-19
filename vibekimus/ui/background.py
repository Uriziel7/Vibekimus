"""Waiting on the Tk thread for work done in another thread. Tk may only be used from its own
thread, so the work's Future is looked at every few milliseconds (widget.after) and the
answer is handed over there."""
import tkinter as tk

POLL_MS = 50


class FutureWatch:
    """Calls on_done(future) on the Tk thread once the future is done — unless the widget
    is gone by then or the watch was cancelled."""

    def __init__(self, widget, future, on_done, poll_ms=POLL_MS):
        self._widget, self._future, self._on_done, self._poll_ms = widget, future, on_done, poll_ms
        self._timer = None
        self._check()

    def _check(self):
        self._timer = None
        try:
            if not self._widget.winfo_exists():
                return
        except tk.TclError:   # the whole application is closing
            return
        if not self._future.done():
            self._timer = self._widget.after(self._poll_ms, self._check)
            return
        self._on_done(self._future)

    def cancel(self):
        if self._timer is not None:
            self._widget.after_cancel(self._timer)
            self._timer = None


def when_done(widget, future, on_done, poll_ms=POLL_MS):
    """Watch `future` from `widget`'s Tk thread; returns the FutureWatch (to cancel it)."""
    return FutureWatch(widget, future, on_done, poll_ms)
