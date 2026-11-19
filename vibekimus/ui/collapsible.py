"""A frame that opens and closes with a short sliding animation."""
import tkinter as tk

ANIMATION_MS = 180       # how long opening or closing takes
FRAME_MS = 15            # time between animation steps (about 60 per second)


def _ease_out(progress):
    """Fast at first, slowing down at the end (cubic)."""
    return 1 - (1 - progress) ** 3


class Collapsible(tk.Frame):
    """Closed at first. `body` holds the content; open() slides it into view by
    growing the frame from 0 to the content's height, close() the other way.
    on_step runs after every step (e.g. to update a scroll region)."""

    def __init__(self, parent, on_step=lambda: None, **frame_options):
        super().__init__(parent, height=0, **frame_options)
        self.pack_propagate(False)
        self.body = tk.Frame(self, **frame_options)
        self.body.pack(fill="x", anchor="n")
        self._on_step = on_step
        self._timer = None
        self.is_open = False

    def open(self, animate=True):
        self.is_open = True
        self.update_idletasks()
        self._slide_to(self.body.winfo_reqheight(), animate)

    def close(self, animate=True):
        self.is_open = False
        self._slide_to(0, animate)

    def toggle(self):
        self.close() if self.is_open else self.open()

    def _slide_to(self, target, animate):
        if self._timer:
            self.after_cancel(self._timer)
            self._timer = None
        start = self.winfo_height() if self.winfo_ismapped() else int(self.cget("height"))
        self.pack_propagate(False)   # the animation sets the height, not the content
        steps = max(1, ANIMATION_MS // FRAME_MS) if animate else 1
        self._step(start, target, 1, steps)

    def _step(self, start, target, step, steps):
        height = round(start + (target - start) * _ease_out(step / steps))
        self.configure(height=height)
        self._on_step()
        if step < steps:
            self._timer = self.after(FRAME_MS, lambda: self._step(start, target, step + 1, steps))
        else:
            self._timer = None
            if self.is_open:
                self.pack_propagate(True)    # from now on follow the content's height
            else:
                self.pack_propagate(False)
