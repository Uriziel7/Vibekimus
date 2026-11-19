"""The search field of the main window with its Search and Search by image… buttons, and the
field's right-click menu."""
import tkinter as tk
from tkinter import ttk

from vibekimus.clipboard import may_hold_picture
from vibekimus.ui import theme
from vibekimus.ui.menus import SEPARATOR, MenuItem, popup_menu

MAX_QUERY_CHARS = 500


class SearchBar:
    """on_search(), on_image(): the buttons (Enter searches too); paste_picture(): searches with
    the picture on the clipboard (from the field's menu)."""

    def __init__(self, parent, on_search, on_image, paste_picture):
        self._paste_picture = paste_picture
        bar = ttk.Frame(parent, padding=(0, theme.PAD))
        bar.pack(fill="x")
        bar.columnconfigure(0, weight=1)
        self._text = tk.StringVar()
        self._text.trace_add("write", self._limit_length)
        self.entry = tk.Entry(bar, font=theme.FONT_INPUT, bg=theme.CARD_BG, fg=theme.FG, insertbackground=theme.FG,
                              relief="flat", highlightthickness=1, highlightcolor=theme.ACCENT,
                              highlightbackground=theme.BORDER, textvariable=self._text)
        self.entry.grid(row=0, column=0, sticky="ew", ipady=theme.GAP)
        self.entry.bind("<Return>", lambda event: on_search())
        self.entry.bind("<Button-3>", self._show_menu)
        self.entry.focus_set()
        ttk.Button(bar, text="Search", command=on_search,
                   style=theme.PRIMARY_BUTTON).grid(row=0, column=1, padx=(theme.GAP, 0), sticky="ns")
        ttk.Button(bar, text="Search by image…", command=on_image).grid(row=0, column=2, padx=(theme.GAP, 0),
                                                                        sticky="ns")

    @property
    def query(self):
        """The text in the field; with a picture it searches for both together."""
        return self.entry.get().strip()

    def clear(self):
        self.entry.delete(0, tk.END)

    def _limit_length(self, *_trace_arguments):
        text = self._text.get()
        if len(text) > MAX_QUERY_CHARS:
            self._text.set(text[:MAX_QUERY_CHARS])

    def _show_menu(self, event):
        entry = self.entry
        has_selection = bool(entry.selection_present())
        has_text = bool(entry.get())
        try:
            has_clipboard = bool(entry.clipboard_get())
        except tk.TclError:   # nothing (or no text) on the clipboard
            has_clipboard = False
        popup_menu(entry, event, [
            MenuItem("Cut", lambda: entry.event_generate("<<Cut>>"), has_selection),
            MenuItem("Copy", lambda: entry.event_generate("<<Copy>>"), has_selection),
            MenuItem("Paste", lambda: entry.event_generate("<<Paste>>"), has_clipboard),
            MenuItem("Search by pasted picture  (Ctrl+V)", self._paste_picture, may_hold_picture()),
            SEPARATOR,
            MenuItem("Select All", lambda: entry.select_range(0, tk.END), has_text),
            MenuItem("Delete", lambda: entry.delete(0, tk.END), has_text),
        ])
