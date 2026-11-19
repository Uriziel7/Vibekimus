"""A frame that lays its children out left to right and wraps them onto new rows
when the window is too narrow — so no control ever disappears off the edge."""
import tkinter as tk

from vibekimus.ui import theme


def flow_layout(sizes, available_width, gap_x, gap_y):
    """Place boxes of the given (width, height) in rows no wider than available_width.

    Returns ([(x, y) per box], total height). A box wider than the row still gets a
    row of its own. Each row is as tall as its tallest box; boxes are centred on it.
    """
    rows = []
    row, row_width = [], 0
    for index, (width, height) in enumerate(sizes):
        needed = width if not row else row_width + gap_x + width
        if row and needed > available_width:
            rows.append(row)
            row, needed = [], width
        row.append(index)
        row_width = needed
    if row:
        rows.append(row)

    positions = [(0, 0)] * len(sizes)
    y = 0
    for number, row in enumerate(rows):
        row_height = max(sizes[i][1] for i in row)
        x = 0
        for i in row:
            width, height = sizes[i]
            positions[i] = (x, y + (row_height - height) // 2)
            x += width + gap_x
        y += row_height + (gap_y if number < len(rows) - 1 else 0)
    return positions, y


class FlowFrame(tk.Frame):
    """Children are created with this frame as parent and added with add(); they are
    positioned by the frame itself (do not pack or grid them)."""

    def __init__(self, parent, gap_x=theme.GROUP_GAP, gap_y=theme.ROW_GAP, **options):
        options.setdefault("bg", theme.PANEL_BG)
        super().__init__(parent, **options)
        self._gap_x, self._gap_y = gap_x, gap_y
        self._items = []
        self._pending = None
        self.bind("<Configure>", lambda event: self._schedule())

    def add(self, widget):
        self._items.append(widget)
        widget.bind("<Configure>", lambda event: self._schedule(), add="+")
        self._schedule()
        return widget

    def _schedule(self):
        """Lay out once after the current burst of size changes."""
        if self._pending is None:
            self._pending = self.after_idle(self._layout)

    def _layout(self):
        self._pending = None
        visible = [item for item in self._items if item.winfo_exists()]
        sizes = [(item.winfo_reqwidth(), item.winfo_reqheight()) for item in visible]
        width = max(self.winfo_width(), 1)
        positions, height = flow_layout(sizes, width, self._gap_x, self._gap_y)
        for item, (x, y) in zip(visible, positions):
            item.place(x=x, y=y)
        if int(self.cget("height")) != height:
            self.configure(height=height)
