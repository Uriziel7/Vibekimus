"""A video's timeline under its preview: every sampled moment as a bar as tall as its
score, the moments where the search matches in orange, the shown one in blue.
A click on a bar shows that moment."""
import math
import tkinter as tk

from vibekimus.ui import theme
from vibekimus.ui.formatting import format_moment

STRIP_HEIGHT = 72
SIDE_MARGIN = 24
TOP_MARGIN = 18            # room for the label of the moment under the pointer
LABEL_HEIGHT = 16          # room for the times under the track
BAR_WIDTH = 4
CURRENT_BAR_WIDTH = 6
LOWEST_BAR = 4             # the lowest-scoring moment is still visible
PLAYHEAD_WIDTH = 2         # a line through the whole strip at the moment shown
PICK_DISTANCE = 12         # a click this many pixels from a bar picks it
TRACK_WIDTH = 2


class TimelineStrip:
    def __init__(self, parent, on_pick):
        self._on_pick = on_pick            # (Moment)
        self._timeline = None
        self._current = None               # seconds of the moment shown
        self._hovered = None               # Moment under the pointer
        self.canvas = canvas = tk.Canvas(parent, height=STRIP_HEIGHT, bg=theme.BG, highlightthickness=0,
                                         cursor="hand2")
        canvas.bind("<Configure>", lambda event: self._draw())
        canvas.bind("<Motion>", self._on_motion)
        canvas.bind("<Leave>", lambda event: self._hover(None))
        canvas.bind("<Button-1>", self._on_click)

    def show(self, timeline, current_seconds):
        """Draw a video's timeline, or nothing (timeline None: a picture)."""
        self._timeline, self._current, self._hovered = timeline, current_seconds, None
        self._draw()

    # ── geometry ─────────────────────────────────────────────────────────────
    def _x(self, seconds):
        usable = max(self.canvas.winfo_width() - 2 * SIDE_MARGIN, 1)
        return SIDE_MARGIN + usable * min(seconds / self._timeline.length, 1.0)

    def _track_y(self):
        return self.canvas.winfo_height() - LABEL_HEIGHT

    def _bar_height(self, score, lowest, highest):
        tallest = self._track_y() - TOP_MARGIN
        if not math.isfinite(score):
            return 0
        share = (score - lowest) / (highest - lowest) if highest > lowest else 1.0
        return LOWEST_BAR + share * (tallest - LOWEST_BAR)

    def _moment_at(self, x):
        if self._timeline is None:
            return None
        nearest = min(self._timeline.moments, key=lambda moment: abs(self._x(moment.seconds) - x))
        return nearest if abs(self._x(nearest.seconds) - x) <= PICK_DISTANCE else None

    # ── drawing ──────────────────────────────────────────────────────────────
    def _draw(self):
        canvas = self.canvas
        canvas.delete("all")
        timeline = self._timeline
        if timeline is None or canvas.winfo_width() <= 1:
            return
        track = self._track_y()
        canvas.create_line(SIDE_MARGIN, track, canvas.winfo_width() - SIDE_MARGIN, track, fill=theme.BORDER,
                           width=TRACK_WIDTH)
        lowest, highest = timeline.score_range()
        playhead = self._x(self._current)
        canvas.create_line(playhead, TOP_MARGIN, playhead, track, fill=theme.ACCENT_HOVER, width=PLAYHEAD_WIDTH)
        for moment in timeline.moments:
            self._draw_bar(moment, self._bar_height(moment.score, lowest, highest))
        canvas.create_text(SIDE_MARGIN, track + LABEL_HEIGHT / 2, text=format_moment(0), anchor="w",
                           fill=theme.FG_MUTED, font=theme.FONT_SMALL)
        canvas.create_text(canvas.winfo_width() - SIDE_MARGIN, track + LABEL_HEIGHT / 2, anchor="e",
                           text=format_moment(timeline.length), fill=theme.FG_MUTED, font=theme.FONT_SMALL)
        self._draw_hover_label()

    def _draw_bar(self, moment, height):
        x, track = self._x(moment.seconds), self._track_y()
        current = moment.seconds == self._current
        colour = theme.ACCENT_HOVER if current else theme.ORANGE if moment.matches else theme.FG_MUTED
        width = CURRENT_BAR_WIDTH if current else BAR_WIDTH
        self.canvas.create_rectangle(x - width / 2, track - height, x + width / 2, track, fill=colour, width=0)

    def _draw_hover_label(self):
        moment = self._hovered
        if moment is None:
            return
        text = format_moment(moment.seconds)
        if math.isfinite(moment.score):
            text += f"  ·  {moment.score:.3f}"
        x = min(max(self._x(moment.seconds), SIDE_MARGIN), self.canvas.winfo_width() - SIDE_MARGIN)
        self.canvas.create_text(x, TOP_MARGIN / 2, text=text, fill=theme.FG, font=theme.FONT_SMALL)

    # ── pointer ──────────────────────────────────────────────────────────────
    def _on_motion(self, event):
        self._hover(self._moment_at(event.x))

    def _hover(self, moment):
        if moment != self._hovered:
            self._hovered = moment
            self._draw()

    def _on_click(self, event):
        moment = self._moment_at(event.x)
        if moment is not None:
            self._on_pick(moment)
