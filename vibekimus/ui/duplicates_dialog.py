"""Asking how alike two images must be to count as duplicates."""
import tkinter as tk
from tkinter import ttk

from vibekimus.ui import theme
from vibekimus.ui.windows import new_window

WRAP = 380
EXPLANATION = ("Every image is compared with every other one. A group is shown whole when at least one of its "
               "copies passes the filters (type, dates, resolution, size, folder); groups follow the chosen sort "
               "order. In each group the best copy "
               "(highest resolution, then largest file) is marked to keep. Nothing is selected for you: tick "
               "the files to remove, or a whole group at once. Nothing is deleted until you press Delete, "
               "and then only to the Recycle Bin.")


class DuplicatesDialog:
    """Calls on_start(threshold) with the similarity chosen for the current model."""

    def __init__(self, parent, model, on_start):
        self._on_start = on_start
        self._window = window = new_window(parent, "Find duplicates", resizable=False, modal=True,
                                           close_on_escape=True)

        body = ttk.Frame(window, padding=theme.PAD * 2)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Which images count as duplicates?", font=theme.FONT_HEADING).pack(anchor="w")
        self._threshold = tk.DoubleVar(value=model.same_picture)
        for value, label in ((model.same_picture, "The same picture (re-saved, resized, recompressed)"),
                             (model.very_similar, "Also very similar ones (burst shots, small edits)")):
            ttk.Radiobutton(body, text=label, value=value, variable=self._threshold).pack(
                anchor="w", pady=(theme.GAP, 0))
        ttk.Label(body, text=EXPLANATION, wraplength=WRAP, style=theme.MUTED_LABEL,
                  justify="left").pack(anchor="w", pady=(theme.PAD, 0))
        buttons = ttk.Frame(body)
        buttons.pack(fill="x", pady=(theme.PAD * 2, 0))
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="right")
        ttk.Button(buttons, text="Find duplicates", command=self._start, style=theme.PRIMARY_BUTTON).pack(
            side="right", padx=(0, theme.GAP))

    def _start(self):
        threshold = self._threshold.get()
        self._window.destroy()
        self._on_start(threshold)
