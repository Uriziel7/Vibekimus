"""A download in progress: a bar, what is being done, and "1.2 GB of 3.0 GB · 25 MB/s ·
about 2 min left". The work runs in a thread; the widget only reads its progress.
Standard library only (the first-start window uses it before anything is installed)."""
import threading
import time
from tkinter import ttk

from vibekimus.runtime.downloader import DownloadCancelled, ProgressCounter
from vibekimus.ui import theme
from vibekimus.ui.formatting import format_size
from vibekimus.ui.widgets import ButtonRow
from vibekimus.units import SECONDS_PER_MINUTE

POLL_MS = 200
SPEED_WINDOW_SECONDS = 5.0     # the speed shown is the average over this long
BAR_STEPS = 1000
BAR_LENGTH = 420
STOP = "Stop"


def _time_left(seconds):
    if seconds < SECONDS_PER_MINUTE:
        return "less than a minute left"
    return f"about {round(seconds / SECONDS_PER_MINUTE)} min left"


class DownloadProgress(ttk.Frame):
    """run(counter, work, on_done): work(should_stop) runs in a thread and reports to
    counter (runtime.downloader.ProgressCounter); on_done(error, cancelled) on the Tk thread."""

    def __init__(self, parent):
        super().__init__(parent)
        self._item = ttk.Label(self, text="", style=theme.MUTED_LABEL)
        self._item.pack(anchor="w")
        self._bar = ttk.Progressbar(self, maximum=BAR_STEPS, length=BAR_LENGTH, mode="determinate")
        self._bar.pack(fill="x", pady=theme.GAP)
        self._numbers = ttk.Label(self, text="")
        self._numbers.pack(anchor="w")
        self._stop = threading.Event()
        self._samples = []   # (time, bytes done) over the last SPEED_WINDOW_SECONDS
        self._outcome = None
        self._thread = None

    def run(self, counter, work, on_done):
        self._counter, self._on_done = counter, on_done
        self._outcome = None
        self._stop = threading.Event()   # a new one: "Try again" after Stop must not stop at once
        self._samples = []
        self._thread = threading.Thread(target=self._work, args=(work,), daemon=True)
        self._thread.start()
        self.after(POLL_MS, self._poll)

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def cancel(self):
        self._stop.set()

    def _work(self, work):
        try:
            work(self._stop.is_set)
            self._outcome = (None, False)
        except DownloadCancelled:
            self._outcome = (None, True)
        except Exception as error:
            self._outcome = (error, False)

    def _poll(self):
        done, total, item = self._counter.snapshot()
        self._show(done, total, item)
        # on_done only once the thread has ended and let go of the work (and the window it
        # belongs to): Tk objects must never be freed in that thread
        if self._outcome is None or self._thread.is_alive():
            self.after(POLL_MS, self._poll)
        else:
            self._on_done(*self._outcome)

    def _show(self, done, total, item):
        now = time.monotonic()
        self._samples = [s for s in self._samples if now - s[0] <= SPEED_WINDOW_SECONDS] + [(now, done)]
        self._item.config(text=item)
        self._bar.config(value=BAR_STEPS * done / total if total else 0)
        text = f"{format_size(done)} of {format_size(total)}"
        start_time, start_done = self._samples[0]
        elapsed = now - start_time
        speed = (done - start_done) / elapsed if elapsed > 0 else 0
        if speed > 0:
            text += f"  ·  {format_size(speed)}/s  ·  {_time_left((total - done) / speed)}"
        self._numbers.config(text=text)


class DownloadPanel:
    """A download's part of a window: the progress while it runs (with Stop), and what went
    wrong after a stop or a failure. The window shows its own buttons in `buttons` otherwise."""

    def __init__(self, parent, wrap, stopped_text, failed_text):
        """failed_text has an {error} field."""
        self._progress = DownloadProgress(parent)
        self._message = ttk.Label(parent, text="", wraplength=wrap, justify="left")
        self._stopped_text, self._failed_text = stopped_text, failed_text
        self.buttons = ButtonRow(parent)
        self.buttons.pack(fill="x", side="bottom", pady=(theme.PAD * 2, 0))
        self.running = False

    def start(self, total, work, on_end):
        """work(counter, should_stop) runs in a thread and reports its bytes to counter
        (of `total`); on_end(error, cancelled) on the Tk thread once it has ended."""
        self.running = True
        self._message.pack_forget()
        self._progress.pack(fill="x", pady=(theme.PAD * 2, 0))
        self.buttons.show((STOP, self.cancel, None))
        counter = ProgressCounter(total)
        self._progress.run(counter, lambda should_stop: work(counter, should_stop),
                           lambda error, cancelled: self._ended(error, cancelled, on_end))

    def cancel(self):
        self._progress.cancel()

    def _ended(self, error, cancelled, on_end):
        self.running = False
        on_end(error, cancelled)

    def show_outcome(self, error, cancelled):
        """After a stop (what was downloaded is kept) or a failure (why)."""
        self._progress.pack_forget()
        self._message.config(text=self._stopped_text if cancelled else self._failed_text.format(error=error),
                             foreground=theme.FG if cancelled else theme.STATUS_ERROR)
        self._message.pack(anchor="w", pady=(theme.PAD * 2, 0))
