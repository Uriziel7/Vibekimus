"""Library → Open saved index: every index saved for the selected model, with the folders
of the selected one beside the list; the classic file dialog stays available (Browse…),
and "Find on this PC" looks on every drive for indexes kept inside folders."""
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import ttk

from vibekimus.saved_indexes import IndexPlace
from vibekimus.ui import theme
from vibekimus.ui.background import when_done
from vibekimus.ui.formatting import format_date, format_size
from vibekimus.ui.widgets import ButtonRow, scrolled_listbox
from vibekimus.ui.windows import new_window

GEOMETRY = "1040x480"
POLL_MS = 300
LIST_WIDTH = 490        # the table keeps this width; the folders of the selected index get the rest
DETAILS_WRAP = 480
PLACES = {IndexPlace.COLLECTION: "Program data", IndexPlace.FOLDER: "Inside the folder",
          IndexPlace.ALL_DRIVES: "All drives"}
COLUMNS = (("name", "Index", 180), ("place", "Kept", 110), ("updated", "Updated", 130), ("size", "Size", 70))
EMPTY = "No saved index of {model} yet. Find on this PC looks for indexes kept inside folders; Browse… opens a file."
UNKNOWN = "?"
FOLDERS_UNREADABLE = "(not readable now — encrypted with a locked key?)"


def _count_text(count, noun):
    return f"{UNKNOWN if count is None else f'{count:,}'} {noun}"


class SavedIndexesDialog:
    """on_open(saved index) or on_browse() once the user has chosen; the window closes first."""

    def __init__(self, parent, saved_indexes, model_label, on_open, on_browse):
        self._saved_indexes, self._model_label = saved_indexes, model_label
        self._on_open, self._on_browse = on_open, on_browse
        self._items = {}            # tree row -> SavedIndex
        self._searcher = ThreadPoolExecutor(max_workers=1)   # "Find on this PC"
        self._search = None         # its background.FutureWatch while it looks
        self._stop_search = threading.Event()
        self._window = window = new_window(parent, f"Open saved index — {model_label}", geometry=GEOMETRY,
                                           modal=True, on_close=self._close)
        body = ttk.Frame(window, padding=theme.PAD * 2)
        body.pack(fill="both", expand=True)
        self._build_buttons(body)
        columns = ttk.Frame(body)
        columns.pack(fill="both", expand=True)
        self._tree = self._build_tree(columns)
        self._details = self._build_details(columns)
        self._fill()

    # ── layout ───────────────────────────────────────────────────────────────
    def _build_tree(self, parent):
        holder = ttk.Frame(parent, width=LIST_WIDTH)
        holder.pack(side="left", fill="y")
        holder.pack_propagate(False)
        tree = ttk.Treeview(holder, columns=[c[0] for c in COLUMNS], show="headings", selectmode="browse")
        for key, heading, width in COLUMNS:
            tree.heading(key, text=heading, anchor="w")
            tree.column(key, width=width, anchor="w", stretch=key == "name")
        tree.pack(fill="both", expand=True)
        tree.bind("<<TreeviewSelect>>", lambda event: self._show_selected())
        tree.bind("<Double-Button-1>", lambda event: self._open())
        self._empty = ttk.Label(holder, text="", wraplength=LIST_WIDTH - theme.PAD, justify="left",
                                style=theme.MUTED_LABEL)
        return tree

    def _build_details(self, parent):
        panel = ttk.Frame(parent, padding=(theme.PAD * 2, 0, 0, 0))
        panel.pack(side="left", fill="both", expand=True)
        self._title = ttk.Label(panel, text="", font=theme.FONT_HEADING)
        self._title.pack(anchor="w")
        self._counts = ttk.Label(panel, text="", style=theme.MUTED_LABEL)
        self._counts.pack(anchor="w", pady=(theme.GAP, theme.PAD))
        ttk.Label(panel, text="Folders in this index", font=theme.FONT_BOLD).pack(anchor="w", pady=(0, theme.GAP))
        holder = ttk.Frame(panel)
        holder.pack(fill="both", expand=True)
        folders = scrolled_listbox(holder)
        self._kept = ttk.Label(panel, text="", style=theme.MUTED_LABEL, wraplength=DETAILS_WRAP, justify="left")
        self._kept.pack(anchor="w", pady=(theme.GAP, 0))
        return folders

    def _build_buttons(self, body):
        bottom = ttk.Frame(body)
        bottom.pack(fill="x", side="bottom", pady=(theme.PAD * 2, 0))
        self._find_button = ttk.Button(bottom, text="Find on this PC", command=self._find)
        self._find_button.pack(side="left")
        self._forget_button = ttk.Button(bottom, text="Remove from list", command=self._forget)
        self._forget_button.pack(side="left", padx=(theme.GAP, 0))
        self._status = ttk.Label(bottom, text="", style=theme.MUTED_LABEL)
        self._status.pack(side="left", padx=(theme.PAD, 0))
        buttons = ButtonRow(bottom)
        buttons.pack(side="right")
        buttons.show(("Open", self._open, theme.PRIMARY_BUTTON), ("Browse…", self._browse, None),
                     ("Cancel", self._close, None))

    # ── contents ─────────────────────────────────────────────────────────────
    def _fill(self):
        self._tree.delete(*self._tree.get_children())
        self._items = {}
        for saved in self._saved_indexes.list():
            row = self._tree.insert("", tk.END, values=(saved.name, PLACES[saved.kind],
                                                        format_date(saved.modified, with_time=True),
                                                        format_size(saved.size)))
            self._items[row] = saved
        if self._items:
            self._empty.pack_forget()
            first = next(iter(self._items))
            self._tree.selection_set(first)
            self._tree.focus(first)
        else:
            self._empty.config(text=EMPTY.format(model=self._model_label))
            self._empty.pack(anchor="w", pady=(theme.PAD, 0))
        self._show_selected()

    def _selected(self):
        selection = self._tree.selection()
        return self._items.get(selection[0]) if selection else None

    def _show_selected(self):
        saved = self._selected()
        self._details.delete(0, tk.END)
        if saved is None:
            for label in (self._title, self._counts, self._kept):
                label.config(text="")
            self._forget_button.state(["disabled"])
            return
        images, frames = saved.counts()
        self._title.config(text=saved.name)
        self._counts.config(text=f"{_count_text(images, 'images')}  ·  {_count_text(frames, 'video frames')}")
        for folder in saved.roots or (FOLDERS_UNREADABLE,):
            self._details.insert(tk.END, folder)
        self._kept.config(text=f"Kept in: {saved.data_dir}")
        self._forget_button.state(["!disabled"] if saved.kind == IndexPlace.FOLDER else ["disabled"])

    # ── actions ──────────────────────────────────────────────────────────────
    def _open(self):
        saved = self._selected()
        if saved is not None:
            self._close()
            self._on_open(saved)

    def _browse(self):
        self._close()
        self._on_browse()

    def _forget(self):
        saved = self._selected()
        if saved is not None and saved.kind == IndexPlace.FOLDER:
            self._saved_indexes.forget(saved.data_dir)
            self._fill()

    def _find(self):
        if self._search is not None:   # pressed again: stop
            self._stop_search.set()
            return
        self._stop_search.clear()
        future = self._searcher.submit(self._saved_indexes.find_on_pc, self._stop_search.is_set)
        self._search = when_done(self._window, future, self._searched, POLL_MS)
        self._find_button.config(text="Stop searching")
        self._status.config(text="Looking on every drive…")

    def _searched(self, future):
        self._search = None
        self._find_button.config(text="Find on this PC")
        new = 0 if future.exception() is not None else future.result()
        self._status.config(text=f"Found {new} more" if new else "Nothing new found")
        self._fill()

    def _close(self):
        self._stop_search.set()
        if self._search is not None:
            self._search.cancel()
        self._searcher.shutdown(wait=False)
        self._window.destroy()
