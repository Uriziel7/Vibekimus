"""Single and double clicks on one widget, told apart by the program instead of by Tk.

Tk's own double click has a fixed 500 ms window and always delivers the first click
of a double click as a single click first. Here a click is reported only once
settings.DOUBLE_CLICK_MS have passed without a second press, so a double click never
shows the single click's effect (a tile selected and deselected again) on its way."""
from vibekimus import settings

CLICK_SLOP_PX = 5   # a press that moves further than this before it comes up is a drag, not a click


class ClickSorter:
    """Calls on_click() for a single click, on_double_click() for a double click; presses
    that continue a double click (a third, a fourth) are ignored."""

    def __init__(self, widget, on_click, on_double_click):
        self._widget = widget
        self._on_click = on_click
        self._on_double_click = on_double_click
        self._pressed_at = None   # where the button went down, until it comes up
        self._waiting = None      # timer: a click waits to see whether a second press follows
        self._settling = None     # timer: presses right after a double click belong to it
        widget.bind("<ButtonPress-1>", self._on_press, add="+")
        widget.bind("<ButtonRelease-1>", self._on_release, add="+")
        widget.bind("<Destroy>", lambda event: self._cancel_timers(), add="+")

    def forget_press(self):
        """The coming release is no click (a drag began)."""
        self._pressed_at = None

    def _on_press(self, event):
        self._pressed_at = None
        if self._settling:
            return
        if self._waiting:
            self._cancel_timers()
            self._settling = self._widget.after(settings.DOUBLE_CLICK_MS, self._settled)
            self._on_double_click()
            return
        self._pressed_at = (event.x, event.y)

    def _on_release(self, event):
        pressed_at, self._pressed_at = self._pressed_at, None
        if pressed_at is None or max(abs(event.x - pressed_at[0]), abs(event.y - pressed_at[1])) > CLICK_SLOP_PX:
            return   # a drag, a press that began elsewhere, or the end of a double click
        self._waiting = self._widget.after(settings.DOUBLE_CLICK_MS, self._clicked)

    def _clicked(self):
        self._waiting = None
        self._on_click()

    def _settled(self):
        self._settling = None

    def _cancel_timers(self):
        """Also when the widget goes away: a timer left behind would call into a destroyed tile."""
        for timer in (self._waiting, self._settling):
            if timer:
                self._widget.after_cancel(timer)
        self._waiting = self._settling = None
