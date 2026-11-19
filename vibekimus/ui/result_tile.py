"""What every shown search result has in common, whether a card in the grid or a
row in the duplicates list: selection by path, clicks, the context menu and
dragging the file into other programs."""
import tkinter as tk
from dataclasses import dataclass
from typing import Callable

from vibekimus.ui import theme
from vibekimus.ui.clicks import ClickSorter

DRAG_ACTION = "copy"   # dragged files are copied, never moved away from their folder


@dataclass(frozen=True)
class CardCallbacks:
    on_click: Callable          # (tile) — single click on a list row: show in the file manager
    on_open: Callable           # (tile) — open the image / play the video from its moment
    on_context_menu: Callable   # (event, tile)
    on_toggle: Callable         # (path, selected)
    on_drag: Callable           # (tile) -> paths to drag out
    on_drag_end: Callable       # () — the drag is over (dropped anywhere, or given up)
    rating_of: Callable         # (result) -> relevance.Rating
    on_rate: Callable           # (result, mark) — ▲ / ▼ clicked on the tile


class ResultTile(tk.Frame):
    """Several tiles may show the same file (frames of one video); selection is by path."""

    def __init__(self, parent, result, selected, callbacks, drag_type, **frame_options):
        super().__init__(parent, **frame_options)
        self.result = result
        self.path = result.path
        self._callbacks = callbacks
        self._drag_type = drag_type   # tkinterdnd2's file type, or None without drag and drop
        self._selected = tk.BooleanVar(value=selected)
        self._click_sorters = []
        self.bind("<Button-3>", self._open_menu)

    def set_selected(self, selected):
        self._selected.set(selected)

    def show_rating(self):
        """Draw the result's rating again (tiles that cannot be rated show none)."""

    def _select_box(self, parent, background, text="Select"):
        return tk.Checkbutton(parent, text=text, variable=self._selected, bg=background, fg=theme.FG,
                              selectcolor=theme.BG, activebackground=background,
                              command=lambda: self._callbacks.on_toggle(self.path, self._selected.get()))

    def _make_interactive(self, widget, draggable=False):
        """Clicks and the context menu on a part of the tile; draggable for its picture."""
        self._click_sorters.append(ClickSorter(widget, self._clicked, lambda: self._callbacks.on_open(self)))
        widget.bind("<Button-3>", self._open_menu)
        if draggable and self._drag_type is not None:
            self._make_draggable(widget)

    def _clicked(self):
        """A single click (a double click opens the file instead)."""
        self._callbacks.on_click(self)

    def _open_menu(self, event):
        self._callbacks.on_context_menu(event, self)

    def _make_draggable(self, widget):
        """Drag the file (or the whole selection) into another program (needs tkinterdnd2)."""
        drag_type = self._drag_type
        try:
            widget.drag_source_register(1, drag_type)
            widget.dnd_bind("<<DragInitCmd>>", lambda event: (DRAG_ACTION, drag_type, tuple(self._drag_started())))
            widget.dnd_bind("<<DragEndCmd>>", lambda event: self._callbacks.on_drag_end())
        except Exception:
            pass   # dragging out is optional

    def _drag_started(self):
        """The paths to drag out; the press that began the drag is no click."""
        for sorter in self._click_sorters:
            sorter.forget_press()
        return self._callbacks.on_drag(self)
