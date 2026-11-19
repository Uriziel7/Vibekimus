"""Explore: the library's pictures grouped into topics, each a card with a few of its
pictures, its guessed name and size. Show puts a topic's pictures in the results of
the main window; Inside groups a topic into smaller topics; Back goes up again.
Grouping and the pictures of the cards are made in the background."""
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from tkinter import ttk
from typing import Callable

from PIL import Image, ImageOps, ImageTk

from vibekimus import settings
from vibekimus.ui import theme
from vibekimus.ui.background import when_done
from vibekimus.ui.widgets import ScrollableFrame
from vibekimus.ui.windows import new_window

GEOMETRY = "1180x820"
MIN_WIDTH, MIN_HEIGHT = 520, 420
COLLAGE_SIDE = 220                  # a card's picture: COLLAGE_GRID × COLLAGE_GRID pictures of the topic
COLLAGE_GRID = 2
COLLAGE_GAP = 2
CARD_PADDING = 6
CARD_GAP = 10
NAME_WRAP = COLLAGE_SIDE
POLL_MS = 40
PICTURE_THREADS = 4
ALL_PICTURES = "All pictures"
PATH_SEPARATOR = "  ›  "
GROUPING = "Grouping {count} pictures into topics…"
GROUPED = "{topics} topics of {count:,} pictures — Show puts a topic in the results, Inside splits it further"
NOTHING = "No pictures to group (check the filters and exclusions)."
FAILED = "Grouping failed: {error}"


@dataclass(frozen=True)
class ExploreHooks:
    find: Callable        # (within: Topic or None) -> topics, largest first; runs in a worker thread
    show: Callable        # (topic) — its pictures as the main window's results
    picture: Callable     # (key, (width, height)) -> PIL image of an indexed picture; any thread
    pictures: int         # how many pictures the top level groups


class ExploreWindow:
    def __init__(self, parent, title, hooks):
        self._hooks = hooks
        self._levels = []          # (name, topics) from the whole library down to the shown level
        self._collages = {}        # first keys of a topic -> PIL collage (kept while the window is open)
        self._photos = []          # PhotoImages of the shown cards (Tk forgets unreferenced ones)
        self._cards = []           # (frame, canvas, topic)
        self._pending = {}         # Future of a collage being made -> its FutureWatch
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=PICTURE_THREADS)
        self._grouping = ThreadPoolExecutor(max_workers=1)
        self._window = window = new_window(parent, title, geometry=GEOMETRY, min_size=(MIN_WIDTH, MIN_HEIGHT),
                                           transient=False, on_close=self.close, close_on_escape=True)
        window.bind("<BackSpace>", lambda event: self._back())
        self._build_header(window)
        self._page = ScrollableFrame(window)
        self._page.pack(fill="both", expand=True)
        self._page.bind("<Configure>", lambda event: self._place_cards())
        window.bind("<MouseWheel>", self._page.scroll)   # these cards, not the results behind the window
        self._group(None, ALL_PICTURES, hooks.pictures)

    def _build_header(self, window):
        header = ttk.Frame(window, padding=theme.PAD * 2)
        header.pack(side="top", fill="x")
        self._back_button = ttk.Button(header, text="◀  Back", command=self._back, state="disabled")
        self._back_button.pack(side="left")
        self._path = ttk.Label(header, text=ALL_PICTURES, font=theme.FONT_HEADING)
        self._path.pack(side="left", padx=theme.PAD)
        self._status = ttk.Label(window, style=theme.MUTED_LABEL, padding=(theme.PAD * 2, 0))
        self._status.pack(side="top", fill="x")

    @property
    def is_open(self):
        return not self._closed and bool(self._window.winfo_exists())

    def close(self):
        self._closed = True
        for pool in (self._pool, self._grouping):
            pool.shutdown(wait=False, cancel_futures=True)
        self._window.destroy()

    # ── levels ───────────────────────────────────────────────────────────────
    def _group(self, within, name, count):
        """Group the library (within None) or a topic, in the background, then show it."""
        self._status.config(text=GROUPING.format(count=f"{count:,}"), foreground=theme.STATUS_BUSY)
        self._back_button.config(state="disabled")
        when_done(self._window, self._grouping.submit(self._hooks.find, within),
                  lambda future: self._grouped(future, name), POLL_MS)

    def _grouped(self, future, name):
        error = future.exception()
        if error is not None:
            self._status.config(text=FAILED.format(error=error), foreground=theme.STATUS_ERROR)
            self._back_button.config(state="normal" if self._levels else "disabled")
            return
        self._levels.append((name, future.result()))
        self._show_level()

    def _back(self):
        if len(self._levels) > 1 and str(self._back_button.cget("state")) != "disabled":
            self._levels.pop()
            self._show_level()

    def _show_level(self):
        topics = self._levels[-1][1]
        self._path.config(text=PATH_SEPARATOR.join(name for name, _topics in self._levels))
        self._back_button.config(state="normal" if len(self._levels) > 1 else "disabled")
        count = sum(topic.size for topic in topics)
        self._status.config(text=GROUPED.format(topics=len(topics), count=count) if topics else NOTHING,
                            foreground=theme.FG_MUTED)
        self._show_cards(topics)

    # ── cards ────────────────────────────────────────────────────────────────
    def _show_cards(self, topics):
        for future, watch in self._pending.items():
            future.cancel()
            watch.cancel()
        self._pending.clear()
        for frame, _canvas, _topic in self._cards:
            frame.destroy()
        self._cards, self._photos = [], []
        for topic in topics:
            self._cards.append(self._card(topic))
        self._place_cards()
        self._page.scroll_to_top()

    def _card(self, topic):
        frame = tk.Frame(self._page.body, bg=theme.CARD_BG, padx=CARD_PADDING, pady=CARD_PADDING)
        canvas = tk.Canvas(frame, width=COLLAGE_SIDE, height=COLLAGE_SIDE, bg=theme.BG, highlightthickness=0,
                           cursor="hand2")
        canvas.pack()
        canvas.bind("<Button-1>", lambda event: self._hooks.show(topic))
        tk.Label(frame, text=topic.name, bg=theme.CARD_BG, fg=theme.FG, font=theme.FONT_BOLD, wraplength=NAME_WRAP,
                 justify="left", anchor="w").pack(fill="x", pady=(CARD_PADDING, 0))
        size_text = f"{topic.size:,} picture" + ("" if topic.size == 1 else "s")
        tk.Label(frame, text=size_text, bg=theme.CARD_BG, fg=theme.FG_MUTED, font=theme.FONT_SMALL,
                 anchor="w").pack(fill="x")
        buttons = tk.Frame(frame, bg=theme.CARD_BG)
        buttons.pack(fill="x", pady=(CARD_PADDING, 0))
        ttk.Button(buttons, text="Show", style=theme.PRIMARY_BUTTON,
                   command=lambda: self._hooks.show(topic)).pack(side="left")
        can_split = topic.size >= 2 * settings.EXPLORE_MIN_TOPIC_SIZE
        ttk.Button(buttons, text="Inside  ›", state="normal" if can_split else "disabled",
                   command=lambda: self._group(topic, topic.name, topic.size)).pack(side="left", padx=(theme.GAP, 0))
        self._draw_collage(canvas, topic)
        return frame, canvas, topic

    def _place_cards(self):
        width = self._page.winfo_width()
        columns = max(1, (width - 2 * theme.PAD * 2) // (COLLAGE_SIDE + 2 * CARD_PADDING + CARD_GAP))
        for number, (frame, _canvas, _topic) in enumerate(self._cards):
            frame.grid(row=number // columns, column=number % columns, padx=(0, CARD_GAP), pady=(0, CARD_GAP),
                       sticky="n")

    # ── pictures of the cards ────────────────────────────────────────────────
    def _draw_collage(self, canvas, topic):
        keys = topic.keys[:COLLAGE_GRID * COLLAGE_GRID]
        collage = self._collages.get(keys)
        if collage is not None:
            self._put(canvas, collage)
        else:
            future = self._pool.submit(self._make_collage, keys)
            self._pending[future] = when_done(canvas, future, lambda made: self._collage_made(made, canvas, keys),
                                              POLL_MS)

    def _make_collage(self, keys):
        side = (COLLAGE_SIDE - (COLLAGE_GRID - 1) * COLLAGE_GAP) // COLLAGE_GRID
        collage = Image.new("RGB", (COLLAGE_SIDE, COLLAGE_SIDE), theme.BG)
        for number, key in enumerate(keys):
            try:
                picture = self._hooks.picture(key, (side * 2, side * 2)).convert("RGB")
            except Exception:
                continue   # moved or unreadable: that square stays empty
            left, top = number % COLLAGE_GRID * (side + COLLAGE_GAP), number // COLLAGE_GRID * (side + COLLAGE_GAP)
            collage.paste(ImageOps.fit(picture, (side, side), Image.BICUBIC), (left, top))
        return collage

    def _collage_made(self, future, canvas, keys):
        self._pending.pop(future, None)
        if not future.cancelled() and future.exception() is None:
            self._collages[keys] = future.result()
            self._put(canvas, future.result())

    def _put(self, canvas, collage):
        photo = ImageTk.PhotoImage(collage)
        self._photos.append(photo)
        canvas.create_image(0, 0, image=photo, anchor="nw")

