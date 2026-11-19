"""The scrollable view of search results: pages, thumbnails, selection.

Ordinary results are shown as a grid of cards; the results of a duplicates search
as a list of groups (see result_layouts)."""
import gc
import math
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Callable

from vibekimus import settings
from vibekimus.media_kind import MediaKind
from vibekimus.relevance import Ratings
from vibekimus.runlog import safe_print
from vibekimus.shell import reveal_in_file_manager
from vibekimus.ui import theme
from vibekimus.ui.formatting import GIVEN_MARK, RATING_LABELS, format_moment, matching_moments
from vibekimus.ui.menus import SEPARATOR, MenuItem, SubMenu, popup_menu
from vibekimus.ui.preview_window import PreviewHooks, PreviewWindow
from vibekimus.ui.result_layouts import GridLayout, GroupListLayout
from vibekimus.ui.result_tile import CardCallbacks, ResultTile
from vibekimus.ui.rubber_band import RubberBandSelector
from vibekimus.ui.thumbnail_loader import ThumbnailLoader
from vibekimus.ui.widgets import WINDOWS_WHEEL_DELTA

MIN_PAGE_SIZE = 10
# Ctrl + wheel: the page is redrawn at the new size once the wheel has been still this long.
ZOOM_SETTLE_MS = 250
SCROLL_MARGIN_PX = 15
MENU_QUERY_CHARS = 30          # longest search text quoted in a menu entry
CURRENT_MOMENT, OTHER_MOMENT ="●  ", "    "   # marks the tile's own moment in "Matching moments"
OPENCV_MISSING_MESSAGE = ("OpenCV is not installed - video thumbnails cannot be displayed.\n\n"
                          "Install it with:\n    pip install opencv-python")


@dataclass(frozen=True)
class ResultsViewHooks:
    """What the results view needs from the window that hosts it."""
    set_status: Callable          # (text, colour)
    is_indexing: Callable         # () -> bool; the status line belongs to indexing then
    page_size: Callable           # () -> int, the Results Per Page setting
    copy_selected: Callable       # () — to a folder
    move_selected: Callable
    delete_selected: Callable
    open_result: Callable         # (result) — play a video from its moment, open an image
    find_similar: Callable        # (result, text) — text "" for the picture alone
    typed_query: Callable         # () -> the text in the search field
    copy_files: Callable          # (paths) — to the clipboard
    copy_picture: Callable        # (result) — the image or video frame, to the clipboard
    thumbnail_size_changed: Callable   # (size) — Ctrl + wheel chose another thumbnail size
    text_of: Callable             # (result) -> the text read on the picture ("" when none)
    refine: Callable              # () — search again with the ratings given (or, all taken back, without)


class ResultsView:
    """Shows one page of results at a time and tracks which files are selected.

    A search is identified by a generation number; a search that is no longer the
    newest never reaches the view. Pages are made of items: results in the grid,
    groups in the duplicates list.
    """

    def __init__(self, parent, hooks, thumbnail_size, drag_type=None):
        self._parent = parent
        self.drag_type = drag_type    # tkinterdnd2's file type, or None without drag and drop
        self._root = parent.winfo_toplevel()
        self._hooks = hooks
        self.thumbnail_size = thumbnail_size   # longest side of a grid thumbnail, pixels
        self.results = []          # every result of the current search, best first
        self.selected = set()      # paths of selected files
        self.ratings = Ratings()   # ▲ / ▼ given to results of the shown search
        self.is_searching = False
        self._items = []           # what pages are made of (results or groups)
        self._page_start = 0       # index of the shown page's first item
        self._offset = 0           # number of items up to the end of the shown page
        self._generation = 0
        self._stopped = False
        self._zoom_timer = None
        self._dragging_out = False   # a result is being dragged (it may come back onto the grid)
        self._scroll_after_load = None
        self._summary = None
        self._opencv_warned = False
        self._preview = None       # the PreviewWindow last opened
        self.loader = ThumbnailLoader(self._root, self._warn_opencv_missing)
        self.tile_callbacks = CardCallbacks(self._on_card_click, self._on_card_open,
                                            self._show_card_menu, self._select_path, self._paths_to_drag,
                                            self._drag_ended, self.ratings.of, self.rate)
        self._grid_layout, self._list_layout = GridLayout(self), GroupListLayout(self)
        self._layout = self._grid_layout
        self._build()

    # ── construction ─────────────────────────────────────────────────────────
    def _build(self):
        # The footer is packed before the grid so a short window shrinks the grid, not it.
        footer = ttk.Frame(self._parent, padding=(0, theme.GAP, 0, 0))
        footer.pack(side="bottom", fill="x")
        self._selection_label = ttk.Label(footer, text="", style=theme.MUTED_LABEL)
        self._selection_label.pack(side="left")
        self._refine_bar = self._build_refine_bar(footer)
        self._pages = pages = ttk.Frame(footer)
        pages.pack(anchor="center")
        self._prev_button = ttk.Button(pages, text="← Previous", command=self.prev_page)
        self._prev_button.pack(side="left")
        self._page_label = ttk.Label(pages, text="", font=theme.FONT_NORMAL)
        self._page_label.pack(side="left", padx=theme.GROUP_GAP)
        self._next_button = ttk.Button(pages, text="Next →", command=self.next_page)
        self._next_button.pack(side="left")
        self._update_navigation()

        container = tk.Frame(self._parent, bg=theme.BG)
        container.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(container, bg=theme.BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self._canvas.yview)
        self.grid_frame = tk.Frame(self._canvas, bg=theme.BG, highlightthickness=0)
        self._grid_window = self._canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self._canvas.configure(yscrollcommand=scrollbar.set)
        self._canvas.bind("<Configure>", lambda event: self._arrange(event.width))
        self.grid_frame.bind("<Configure>", lambda event: self.update_scroll_region())
        self._canvas.bind("<ButtonPress-3>", self._show_background_menu)
        self._canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        RubberBandSelector(self._root, self._canvas, lambda: self._layout.visible_tiles(),
                           self.select_paths)
        self._bind_mouse_wheel()

    def _build_refine_bar(self, footer):
        """Shown once a result is rated: how many, Refine results, Clear ratings."""
        bar = ttk.Frame(footer)
        self._ratings_label = ttk.Label(bar, text="")
        self._ratings_label.pack(side="left", padx=(0, theme.GAP))
        ttk.Button(bar, text="Refine results", style=theme.PRIMARY_BUTTON,
                   command=self._hooks.refine).pack(side="left")
        ttk.Button(bar, text="Clear ratings", command=self._clear_ratings).pack(side="left", padx=(theme.GAP, 0))
        return bar

    def _bind_mouse_wheel(self):
        """The wheel scrolls the results; with Ctrl held it zooms the thumbnails."""
        scroll = self._canvas.yview_scroll
        bind = self._canvas.bind_all
        if sys.platform == "darwin":
            bind("<MouseWheel>", lambda e: scroll(int(-e.delta), "units"))
        elif sys.platform.startswith("linux"):
            bind("<Button-4>", lambda e: scroll(-1, "units"))
            bind("<Button-5>", lambda e: scroll(1, "units"))
            bind("<Control-Button-4>", lambda e: self._on_zoom_wheel(e, 1))
            bind("<Control-Button-5>", lambda e: self._on_zoom_wheel(e, -1))
        else:
            bind("<MouseWheel>", lambda e: scroll(int(-e.delta / WINDOWS_WHEEL_DELTA), "units"))
        if not sys.platform.startswith("linux"):
            bind("<Control-MouseWheel>", lambda e: self._on_zoom_wheel(e, 1 if e.delta > 0 else -1))

    def _on_zoom_wheel(self, event, steps):
        """Only in the main window — not in a dialog open over it."""
        try:
            in_main_window = event.widget.winfo_toplevel() is self._root
        except (AttributeError, tk.TclError):   # a widget known to Tcl only, e.g. a drop-down list
            in_main_window = False
        if in_main_window:
            self.zoom(steps)
        return "break"

    def zoom(self, steps):
        """Make the thumbnails `steps` sizes larger (negative: smaller). The page is
        laid out and redrawn sharp at the new size once the wheel stops."""
        levels = settings.THUMBNAIL_ZOOM_LEVELS
        current = min(range(len(levels)), key=lambda i: abs(levels[i] - self.thumbnail_size))
        size = levels[max(0, min(len(levels) - 1, current + steps))]
        if size == self.thumbnail_size:
            return
        self.thumbnail_size = size
        if not self._hooks.is_indexing():
            self._hooks.set_status(f"Thumbnail size: {size} px", theme.STATUS_OK)
        if self._zoom_timer:
            self._root.after_cancel(self._zoom_timer)
        self._zoom_timer = self._root.after(ZOOM_SETTLE_MS, self._zoom_settled)

    def _zoom_settled(self):
        self._zoom_timer = None
        self._hooks.thumbnail_size_changed(self.thumbnail_size)
        self._redraw_page()
        self._arrange()

    def register_drop_target(self, file_type, on_drop):
        """Accept files dragged onto the grid from other programs: on_drop(paths) (needs
        tkinterdnd2). A result dragged out of the grid and let go over it again is ignored —
        a slightly shaky click starts a drag, and it must not start a search."""
        try:
            self._canvas.drop_target_register(file_type)
            self._canvas.dnd_bind("<<Drop>>", lambda event: self._dropped(event, on_drop))
        except Exception:
            pass   # the app works without drag and drop

    def _dropped(self, event, on_drop):
        if not self._dragging_out:
            on_drop(list(self._root.tk.splitlist(event.data)))   # Tk's list: paths with spaces are in braces

    # ── layout ───────────────────────────────────────────────────────────────
    def update_scroll_region(self):
        """Scrollable area: the full canvas width (starting at 0, or the canvas would
        undo the centring offset) and the height of the content."""
        self.grid_frame.update_idletasks()
        content = self._canvas.bbox("all")
        bottom = content[3] if content else 0
        self._canvas.configure(scrollregion=(0, 0, self._canvas.winfo_width(), bottom))

    def _arrange(self, width=None):
        width = width or self._canvas.winfo_width()
        left, used = self._layout.arrange(width)
        self._canvas.coords(self._grid_window, left, 0)
        self._canvas.itemconfig(self._grid_window, width=used)
        self.update_scroll_region()

    # ── search lifecycle ─────────────────────────────────────────────────────
    @property
    def total_found(self):
        return len(self.results)

    def begin(self):
        """A new search starts: the grid is emptied and the search gets the next generation
        number (returned), which makes every earlier search stale. On the Tk thread, before
        the search thread starts — numbered in the search thread, a cancelled search that
        started late could take the number of a newer one and show its results."""
        self.clear()
        self._generation += 1
        self._stopped = False
        self.is_searching = True
        return self._generation

    def is_stale(self, generation):
        return self._stopped or generation != self._generation

    def finish(self):
        self.is_searching = False

    def cancel(self):
        """Abandon the current search and empty the grid."""
        self._generation += 1
        self._stopped = True
        self.clear()
        self.is_searching = False

    def clear(self, keep_results=False):
        """Remove the cards. keep_results=True is a page turn: the result list and
        the selection logic stay, only the widgets go."""
        self.loader.cancel_all()
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        self.grid_frame.configure(height=1)   # placed tiles set the height themselves
        if not keep_results:
            self.results, self._items = [], []
            self._offset = 0
            self.selected.clear()
        self._update_navigation()
        self._show_selection_count()
        gc.collect()   # free the thumbnails' memory now
        self._canvas.yview_moveto(0)

    def show(self, results, generation, selected=(), summary=None):
        """Display the first page of a finished search; `selected` paths start
        selected, `summary` replaces the "Found N results" status text."""
        if self.is_stale(generation):
            return
        self.results = results
        self._summary = summary
        self.selected = set(selected)
        self._show_selection_count()
        is_duplicates = any(result.group for result in results)
        self._layout = self._list_layout if is_duplicates else self._grid_layout
        self._layout.new_results()
        self._items = self._layout.items(results)
        safe_print(f"[SEARCH] Displaying the first page of {len(self._items)} {self._layout.item_noun}")
        self._show_page(0)

    # ── pages ────────────────────────────────────────────────────────────────
    def _page_size(self):
        return max(MIN_PAGE_SIZE, self._hooks.page_size())

    def _page_position(self):
        """(current page, number of pages), both starting at 1."""
        size = self._page_size()
        pages = max(1, math.ceil(len(self._items) / size))
        current = max(1, math.ceil(self._offset / size))
        return current, pages

    def next_page(self):
        self._turn_page(self._offset)

    def prev_page(self):
        current, _pages = self._page_position()
        pages_before_previous = current - 2
        if pages_before_previous >= 0:
            self._turn_page(pages_before_previous * self._page_size())

    def _turn_page(self, start):
        if not self.is_searching:
            self._show_page(start)

    def _show_page(self, start):
        page = self._items[start:start + self._page_size()]
        if not page:
            return
        # The selection stays: files selected on other pages remain selected.
        self.clear(keep_results=True)
        self._page_start, self._offset = start, start + len(page)
        self._stopped = False
        self._arrange()
        self._layout.render(page, self._page_displayed)

    def _redraw_page(self):
        """Show the current page again (new thumbnail size, files removed), keeping
        the scroll position."""
        if self._items:
            self._scroll_after_load = self._canvas.yview()[0]
            last_page_start = (len(self._items) - 1) // self._page_size() * self._page_size()
            self._show_page(min(self._page_start, last_page_start))
        else:
            self.clear(keep_results=True)
            self._page_start = self._offset = 0

    def _page_displayed(self):
        self.is_searching = False
        if not self._hooks.is_indexing():
            self._hooks.set_status(self._summary or f"Found {self.total_found:,} results", theme.STATUS_OK)
        self.update_scroll_region()
        self._update_navigation()
        if self._scroll_after_load is not None:   # a redraw keeps the place
            self._canvas.yview_moveto(self._scroll_after_load)
            self._scroll_after_load = None

    def _update_navigation(self):
        if self._items:
            current, pages = self._page_position()
            self._page_label.config(
                text=f"Page {current} of {pages}  ·  {len(self._items):,} {self._layout.item_noun}")
        else:
            current = 1
            self._page_label.config(text="No results")
        self._prev_button.config(state="normal" if current > 1 else "disabled")
        self._next_button.config(state="normal" if self._offset < len(self._items) else "disabled")

    def _warn_opencv_missing(self):
        if not self._opencv_warned:
            self._opencv_warned = True
            messagebox.showwarning("Missing Dependency", OPENCV_MISSING_MESSAGE)

    # ── preview (Space) ──────────────────────────────────────────────────────
    def focus(self):
        """Keys go to the results (Space previews) instead of the search field."""
        self._canvas.focus_set()

    def preview(self, result=None):
        """The large preview of `result`, or (Space) of the result under the mouse pointer, else
        the first selected one on the page, else the page's first. ← → go through all results."""
        if result is None:
            result = self._result_to_preview()
            if result is None:
                return
        if self._preview is not None and self._preview.is_open:
            self._preview.close()
        index = next(i for i, shown in enumerate(self.results) if shown is result)
        self._preview = PreviewWindow(self._root, self.results, index, PreviewHooks(
            open_result=self._hooks.open_result, show_in_folder=reveal_in_file_manager, text_of=self._hooks.text_of,
            is_selected=lambda path: path in self.selected, set_selected=self._select_path,
            rating_of=self.ratings.of, rate=self.rate))

    def _result_to_preview(self):
        shown = self._layout.visible_tiles()
        if not shown:
            return None
        tile = self._tile_under_pointer() or next((t for t in shown if t.path in self.selected), shown[0])
        return tile.result

    def _tile_under_pointer(self):
        widget = self._root.winfo_containing(*self._root.winfo_pointerxy())
        while widget is not None and not isinstance(widget, ResultTile):
            widget = widget.master
        return widget

    # ── clicks ───────────────────────────────────────────────────────────────
    def _on_card_click(self, card):
        self._scroll_into_view(card)
        reveal_in_file_manager(card.path)

    def _on_card_open(self, card):
        self._hooks.open_result(card.result)

    def _paths_to_drag(self, card):
        """Dragging a selected card drags the whole selection; otherwise just its file."""
        self._dragging_out = True
        return self._paths_for(card)

    def _drag_ended(self):
        # once events still queued (a drop onto the grid among them) have been handled
        self._root.after_idle(self._forget_drag)

    def _forget_drag(self):
        self._dragging_out = False

    def _paths_for(self, card):
        return sorted(self.selected) if card.path in self.selected else [card.path]

    def _scroll_into_view(self, card):
        try:
            self.grid_frame.update_idletasks()
            top = card.winfo_rooty() - self.grid_frame.winfo_rooty()   # rows may sit inside groups
            bottom = top + card.winfo_height()
            content_height = self.grid_frame.winfo_height()
            view_height = self._canvas.winfo_height()
            if content_height <= 0 or view_height <= 0:
                return
            view_top, view_bottom = (fraction * content_height for fraction in self._canvas.yview())
            if bottom > view_bottom:
                self._canvas.yview_moveto((bottom + SCROLL_MARGIN_PX - view_height) / content_height)
            elif top < view_top:
                self._canvas.yview_moveto(max(0.0, (top - SCROLL_MARGIN_PX) / content_height))
        except Exception as error:
            safe_print(f"[SCROLL] Error: {error}")

    # ── ratings (refining the search) ────────────────────────────────────────
    def rate(self, result, mark):
        """▲ LIKE / ▼ DISLIKE a result, or UNRATED to take its rating back."""
        self.ratings.rate(result, mark)
        self._ratings_changed()

    def forget_ratings(self):
        """A new search starts without ratings."""
        self.ratings.clear()
        self._ratings_changed()

    def _clear_ratings(self):
        """Take every rating back, and show the search as it was without them."""
        self.forget_ratings()
        self._hooks.refine()

    def _ratings_changed(self):
        for tile in self._layout.tiles():
            tile.show_rating()
        liked, disliked = self.ratings.counts()
        if liked or disliked:
            self._ratings_label.config(text=f"▲ {liked}   ▼ {disliked}")
            self._refine_bar.pack(side="right", before=self._pages)   # on the line of the page buttons
        else:
            self._refine_bar.pack_forget()

    def _rating_items(self, result):
        """▲ / ▼ for a result of the grid (duplicates are found, not ranked: nothing to refine)."""
        if self._layout is not self._grid_layout:
            return []
        given = self.ratings.of(result)
        return [*(MenuItem((GIVEN_MARK if given == mark else "") + label,
                           lambda mark=mark: self.rate(result, mark.replacing(given)))
                  for mark, label in RATING_LABELS.items()), SEPARATOR]

    # ── selection ────────────────────────────────────────────────────────────
    def _select_path(self, path, select):
        self.select_paths({path}, select)

    def select_paths(self, paths, select):
        """Select or deselect files and update the tick of every tile that shows one
        (all frames of one video share the file)."""
        if select:
            self.selected |= set(paths)
        else:
            self.selected -= set(paths)
        for tile in self._layout.tiles():
            tile.set_selected(tile.path in self.selected)
        self._layout.selection_changed(self.selected)
        self._show_selection_count()

    def _show_selection_count(self):
        count = len(self.selected)
        self._selection_label.config(text=f"{count:,} selected" if count else "")

    def _page_paths(self):
        """Every file of the shown page, also the duplicates of groups not opened."""
        return self._layout.paths(self._items[self._page_start:self._offset])

    def select_page(self):
        self.select_paths(self._page_paths(), True)

    def select_all(self):
        """Every result of the search, on every page."""
        self.select_paths({result.path for result in self.results}, True)

    def deselect_page(self):
        self.select_paths(self._page_paths(), False)

    def deselect_all(self):
        """The whole selection, including files selected on other pages."""
        self.select_paths(set(self.selected), False)

    def remove_files(self, paths):
        """Forget files that no longer exist (moved, deleted) and show the page again
        without them; a duplicates group left with one file disappears."""
        gone = set(paths)
        self.selected -= gone
        self.results = [result for result in self.results if result.path not in gone]
        self._items = self._layout.items(self.results)
        self._show_selection_count()
        self._redraw_page()
        self._update_navigation()

    # ── menus ────────────────────────────────────────────────────────────────
    def _selection_actions(self):
        hooks, any_selected = self._hooks, bool(self.selected)
        return [MenuItem("Copy selected to folder…", hooks.copy_selected, any_selected),
                MenuItem("Move selected to folder…", hooks.move_selected, any_selected),
                MenuItem("Delete selected (Recycle Bin)", hooks.delete_selected, any_selected)]

    def _show_background_menu(self, event):
        popup_menu(self._root, event, [
            *(MenuItem(label, lambda command=command: command(self)) for label, command in SELECTION_COMMANDS),
            SEPARATOR,
            *self._selection_actions(),
        ])

    def _similar_with_text_item(self, result):
        """'Find similar + “at night”' while the search field holds text."""
        text = self._hooks.typed_query()
        if not text:
            return []
        shown = text if len(text) <= MENU_QUERY_CHARS else text[:MENU_QUERY_CHARS] + "…"
        return [MenuItem(f"Find similar + “{shown}”", lambda: self._hooks.find_similar(result, text))]

    def _moments_menu(self, result):
        """'Matching moments ▸' for a video the search matches at more than one moment."""
        moments = matching_moments(result)
        if len(moments) < 2:
            return []
        items = tuple(MenuItem(f"{CURRENT_MOMENT if moment.seconds == result.timestamp else OTHER_MOMENT}"
                               f"Play from {format_moment(moment.seconds)}   ({moment.score:.3f})",
                               lambda moment=moment: self._hooks.open_result(result.at(moment)))
                      for moment in moments)
        return [SubMenu(f"Matching moments ({len(moments)})", items)]

    def _show_card_menu(self, event, card):
        hooks, result, path = self._hooks, card.result, card.path
        selected = path in self.selected
        is_video = result.kind == MediaKind.VIDEO
        files = self._paths_for(card)
        popup_menu(self._root, event, [
            MenuItem(f"Play from {format_moment(result.timestamp)}" if is_video else "Open",
                     lambda: hooks.open_result(result)),
            *self._moments_menu(result),
            MenuItem("Preview  (Space)", lambda: self.preview(result)),
            MenuItem("Show in folder", lambda: reveal_in_file_manager(path)),
            MenuItem("Find similar", lambda: hooks.find_similar(result, "")),
            *self._similar_with_text_item(result),
            SEPARATOR,
            *self._rating_items(result),
            MenuItem(f"Copy {len(files)} files to clipboard" if len(files) > 1 else "Copy file to clipboard",
                     lambda: hooks.copy_files(files)),
            MenuItem("Copy frame as picture" if is_video else "Copy as picture", lambda: hooks.copy_picture(result)),
            SEPARATOR,
            MenuItem("Deselect" if selected else "Select", lambda: self._select_path(path, not selected)),
            SEPARATOR,
            *self._selection_actions(),
        ])


# (label, ResultsView method) of the select / deselect commands, for the toolbar and menus
SELECTION_COMMANDS = (
    ("Select All (current page)", ResultsView.select_page),
    ("Select All", ResultsView.select_all),
    ("Deselect All (current page)", ResultsView.deselect_page),
    ("Deselect All", ResultsView.deselect_all),
)
