"""Space: a large preview of a search result inside the program, with what is known about
it beside the picture. ← / → go through all the results of the search, Enter opens the
file (a video from its moment), Space or Esc closes. A video has its timeline under the
picture: where else in it the search matches (click, or Ctrl + ← →). Pictures are read in the
background, the neighbours ahead of time, so going through results does not wait for the disk."""
import os
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from tkinter import ttk
from typing import Callable

from PIL import Image, ImageTk

from vibekimus.imaging import open_image
from vibekimus.media_kind import MediaKind
from vibekimus.relevance import Rating
from vibekimus.ui import theme
from vibekimus.ui.background import when_done
from vibekimus.ui.formatting import GIVEN_MARK, RATING_LABELS, format_details, format_moment, matching_moments
from vibekimus.ui.video_timeline import TimelineStrip
from vibekimus.ui.windows import new_window
from vibekimus.video_io import read_frame_at

TITLE = "Vibekimus — preview"
INFO_WIDTH = 340
INFO_WRAP = INFO_WIDTH - 4 * theme.PAD     # the panel's padding is 2 × PAD on each side
TEXT_BOX_LINES = 12
LOAD_THREADS = 2
POLL_MS = 30
RESIZE_SETTLE_MS = 80
NEIGHBOURS = 1                 # pictures read ahead on each side
UNREADABLE = "This file cannot be shown (moved, deleted or damaged)."
LOADING = "Loading…"
KEYS_HINT = ("← →  previous / next     Ctrl + ← →  moments of a video     + / −  more / less like this     "
             "Enter  open     Space / Esc  close")
# (mark, button text, keys) of the rating buttons
RATING_BUTTONS = ((Rating.LIKE, RATING_LABELS[Rating.LIKE], ("<plus>", "<equal>", "<KP_Add>")),
                  (Rating.DISLIKE, RATING_LABELS[Rating.DISLIKE], ("<minus>", "<KP_Subtract>")))


@dataclass(frozen=True)
class PreviewHooks:
    open_result: Callable      # (result) — the default viewer, or the player from the moment
    show_in_folder: Callable   # (path)
    text_of: Callable          # (result) -> the text read on the picture ("" when none)
    is_selected: Callable      # (path) -> bool
    set_selected: Callable     # (path, selected)
    rating_of: Callable        # (result) -> relevance.Rating
    rate: Callable             # (result, mark)


def _read_picture(result, max_size):
    """The picture of a result (a video's frame at its moment), at most max_size; None if unreadable."""
    if result.kind == MediaKind.VIDEO:
        picture = read_frame_at(result.path, result.timestamp)
    else:
        picture = open_image(result.path, draft_size=max(max_size))   # at least this on both sides
    if picture is not None:
        picture.thumbnail(max_size, Image.LANCZOS)
    return picture


class PreviewWindow:
    def __init__(self, root, results, index, hooks):
        self._results, self._index, self._hooks = results, index, hooks
        self._max_size = (root.winfo_screenwidth(), root.winfo_screenheight())
        self._pool = ThreadPoolExecutor(max_workers=LOAD_THREADS)
        self._loads = {}               # (result index, seconds into a video) -> Future of its picture
        self._moments = {}             # result index -> the Moment of its video picked on the timeline
        self._picture = self._photo = None
        self._picture_watch = None     # background.FutureWatch of the picture being loaded
        self._resize_timer = None
        self._window = window = new_window(root, TITLE, background=theme.BG, maximized=True, transient=False,
                                           on_close=self.close)
        self._build_info(window)
        picture_area = tk.Frame(window, bg=theme.BG)
        picture_area.pack(side="left", fill="both", expand=True)
        self._timeline = TimelineStrip(picture_area, self._pick_moment)
        self._canvas = tk.Canvas(picture_area, bg=theme.BG, highlightthickness=0)
        self._canvas.pack(side="top", fill="both", expand=True)
        self._canvas.bind("<Configure>", lambda event: self._redraw_soon())
        self._canvas.bind("<Double-Button-1>", lambda event: self._open())
        for keys, action in ((("<space>", "<Escape>"), self.close), (("<Left>", "<Up>"), lambda: self._step(-1)),
                             (("<Right>", "<Down>"), lambda: self._step(1)), (("<Return>",), self._open),
                             (("<Control-Left>",), lambda: self._step_moment(-1)),
                             (("<Control-Right>",), lambda: self._step_moment(1)),
                             (("<Home>",), lambda: self._go(0)),
                             (("<End>",), lambda: self._go(len(self._results) - 1)),
                             *((keys, lambda mark=mark: self._rate(mark)) for mark, _text, keys in RATING_BUTTONS)):
            for key in keys:
                window.bind(key, lambda event, act=action: (act(), "break")[1])
        window.focus_force()
        self._go(index)

    # ── layout ───────────────────────────────────────────────────────────────
    def _build_info(self, window):
        panel = ttk.Frame(window, padding=theme.PAD * 2, width=INFO_WIDTH)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        self._name = ttk.Label(panel, font=theme.FONT_HEADING, wraplength=INFO_WRAP, justify="left")
        self._name.pack(anchor="w")
        self._folder = ttk.Label(panel, style=theme.MUTED_LABEL, wraplength=INFO_WRAP, justify="left")
        self._folder.pack(anchor="w", pady=(theme.GAP, 0))
        self._details = ttk.Label(panel, wraplength=INFO_WRAP, justify="left")
        self._details.pack(anchor="w", pady=(theme.PAD, 0))
        self._position = ttk.Label(panel, style=theme.MUTED_LABEL)
        self._position.pack(anchor="w", pady=(theme.GAP, 0))
        self._selected = tk.BooleanVar()
        ttk.Checkbutton(panel, text="Selected", variable=self._selected, takefocus=False,
                        command=self._selection_changed).pack(anchor="w", pady=(theme.PAD, 0))
        buttons = ttk.Frame(panel)
        buttons.pack(anchor="w", pady=(theme.PAD, 0))
        ttk.Button(buttons, text="Open", style=theme.PRIMARY_BUTTON, takefocus=False,
                   command=self._open).pack(side="left")
        ttk.Button(buttons, text="Show in folder", takefocus=False,
                   command=lambda: self._hooks.show_in_folder(self._result.path)).pack(side="left", padx=(theme.GAP, 0))
        rating = ttk.Frame(panel)
        rating.pack(anchor="w", pady=(theme.PAD, 0))
        self._rating_buttons = {}
        for column, (mark, text, _keys) in enumerate(RATING_BUTTONS):
            button = ttk.Button(rating, text=text, takefocus=False, command=lambda mark=mark: self._rate(mark))
            button.pack(side="left", padx=(theme.GAP if column else 0, 0))
            self._rating_buttons[mark] = (button, text)
        ttk.Label(panel, text=KEYS_HINT, style=theme.MUTED_LABEL, wraplength=INFO_WRAP,
                  justify="left").pack(side="bottom", anchor="w")
        self._text_heading = ttk.Label(panel, text="Text on the picture", font=theme.FONT_BOLD)
        self._text = tk.Text(panel, height=TEXT_BOX_LINES, wrap="word", font=theme.FONT_SMALL, bg=theme.CARD_BG,
                             fg=theme.FG, relief="flat", borderwidth=0, padx=theme.GAP, pady=theme.GAP,
                             takefocus=False)

    # ── moving through the results ───────────────────────────────────────────
    @property
    def _result(self):
        return self._shown(self._index)

    def _shown(self, index):
        """A result as shown: a video at the moment picked on its timeline."""
        moment = self._moments.get(index)
        return self._results[index] if moment is None else self._results[index].at(moment)

    def _load_key(self, index):
        return index, self._shown(index).timestamp

    def _step(self, direction):
        self._go(self._index + direction)

    def _go(self, index):
        if not 0 <= index < len(self._results):
            return
        self._index = index
        self._show_info()
        self._picture = None
        self._draw_message(LOADING)
        self._request_pictures()
        self._wait_for_picture()

    def _request_pictures(self):
        """The current picture and its neighbours; reads that are no longer needed are dropped."""
        wanted = range(max(0, self._index - NEIGHBOURS), min(len(self._results), self._index + NEIGHBOURS + 1))
        wanted_keys = {self._load_key(index) for index in wanted}
        for key in list(self._loads):
            if key not in wanted_keys:
                self._loads.pop(key).cancel()
        for index in wanted:
            key = self._load_key(index)
            if key not in self._loads:
                self._loads[key] = self._pool.submit(_read_picture, self._shown(index), self._max_size)

    def _wait_for_picture(self):
        """The picture of the result shown now (a picture still loading for the one shown before is not)."""
        if self._picture_watch is not None:
            self._picture_watch.cancel()
        self._picture_watch = when_done(self._window, self._loads[self._load_key(self._index)], self._picture_loaded,
                                        POLL_MS)

    def _picture_loaded(self, load):
        self._picture_watch = None
        self._picture = None if load.cancelled() or load.exception() else load.result()
        self._draw()

    # ── drawing ──────────────────────────────────────────────────────────────
    def _show_info(self):
        result = self._result
        self._name.config(text=os.path.basename(result.path))
        self._folder.config(text=os.path.dirname(result.path))
        lines = [format_details(result)]
        if result.kind == MediaKind.VIDEO:
            lines.append(f"▶  frame at {format_moment(result.timestamp)}")
            matching = len(matching_moments(result))
            if matching > 1:
                lines.append(f"Matches at {matching} moments (orange on the timeline)")
        lines.append(f"Score {result.score:.3f}" + ("  ·  copy to keep" if result.keeper else ""))
        self._details.config(text="\n".join(line for line in lines if line))
        self._position.config(text=f"{self._index + 1:,} of {len(self._results):,}")
        self._selected.set(self._hooks.is_selected(result.path))
        self._show_rating()
        self._show_text(self._hooks.text_of(result) if result.kind != MediaKind.VIDEO else "")
        self._show_timeline(result)

    def _show_timeline(self, result):
        strip = self._timeline
        strip.show(result.timeline, result.timestamp)
        if result.timeline is None:
            strip.canvas.pack_forget()
        elif not strip.canvas.winfo_ismapped():
            strip.canvas.pack(side="bottom", fill="x", before=self._canvas)

    def _show_text(self, text):
        self._text.config(state="normal")
        self._text.delete("1.0", "end")
        if text:
            self._text.insert("1.0", text)
            self._text_heading.pack(anchor="w", pady=(theme.PAD * 2, theme.GAP))
            self._text.pack(anchor="w", fill="x")
        else:
            self._text_heading.pack_forget()
            self._text.pack_forget()
        self._text.config(state="disabled")

    def _redraw_soon(self):
        if self._resize_timer is not None:
            self._window.after_cancel(self._resize_timer)
        self._resize_timer = self._window.after(RESIZE_SETTLE_MS, self._draw)

    def _draw(self):
        self._resize_timer = None
        if self._picture is None:
            load = self._loads.get(self._load_key(self._index))
            self._draw_message(LOADING if load is not None and not load.done() else UNREADABLE)
            return
        width, height = self._canvas.winfo_width(), self._canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
        fitted = self._picture.copy()
        fitted.thumbnail((width, height), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(fitted)
        self._canvas.delete("all")
        self._canvas.create_image(width // 2, height // 2, image=self._photo, anchor="center")

    def _draw_message(self, text):
        self._canvas.delete("all")
        self._photo = None
        self._canvas.create_text(self._canvas.winfo_width() // 2, self._canvas.winfo_height() // 2, text=text,
                                 fill=theme.FG_MUTED, font=theme.FONT_NORMAL)

    # ── rating (refining the search) ─────────────────────────────────────────
    def _rate(self, mark):
        """Give the shown result this rating, or take it back when it has it already."""
        result = self._result
        self._hooks.rate(result, mark.replacing(self._hooks.rating_of(result)))
        self._show_rating()

    def _show_rating(self):
        given = self._hooks.rating_of(self._result)
        for mark, (button, text) in self._rating_buttons.items():
            button.config(text=(GIVEN_MARK if given == mark else "") + text)

    # ── moments of a video ───────────────────────────────────────────────────
    def _pick_moment(self, moment):
        self._moments[self._index] = moment
        self._go(self._index)

    def _step_moment(self, direction):
        """The next (1) or previous (-1) moment of the video where the search matches."""
        result = self._result
        if result.timeline is not None:
            moment = result.timeline.next_matching(result.timestamp, direction)
            if moment is not None:
                self._pick_moment(moment)

    # ── actions ──────────────────────────────────────────────────────────────
    def _open(self):
        self._hooks.open_result(self._result)

    def _selection_changed(self):
        self._hooks.set_selected(self._result.path, self._selected.get())

    def close(self):
        if self._picture_watch is not None:
            self._picture_watch.cancel()
        if self._resize_timer is not None:
            self._window.after_cancel(self._resize_timer)
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._window.destroy()

    @property
    def is_open(self):
        return bool(self._window.winfo_exists())
