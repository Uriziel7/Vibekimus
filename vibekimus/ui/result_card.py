"""One search result as a flat, Metro-style tile: the picture fills the whole tile
(scaled and centred, proportions kept), the name and file details sit on a dark band
at its bottom, a video shows its moment in a corner badge. A click selects the tile
(check in the top-right corner, shown on hover, and an accent border), a double click
opens it. ▲ / ▼ beside the check rate the result for refining the search (shown on
hover, and always once rated). The picture can be dragged into other programs."""
import os
import tkinter as tk
from tkinter import font as tkfont

from PIL import Image, ImageOps, ImageTk

from vibekimus.media_kind import MediaKind
from vibekimus.relevance import Rating
from vibekimus.ui import theme
from vibekimus.ui.formatting import format_details, video_badge
from vibekimus.ui.result_tile import ResultTile

TEXT_PADDING = 8          # text and badges from the tile's edges
OVERLAY_HEIGHT = 48       # the dark band behind the text, darkest at the bottom
OVERLAY_DARKNESS = 230    # 0–255: how dark the band gets at the very bottom
OVERLAY_CURVE = 0.45      # < 1: the band darkens quickly, so text is readable on white pictures
LINE_GAP = 1
CHECK_SIZE = 22           # the selection check in the top-right corner
CHECK_MARGIN = 6
SELECTED_BORDER = 3
HOVER_BORDER = 1
BADGE_PADDING = (5, 2)
RATING_GAP = 4            # between the rating buttons and the check
# (mark, canvas tag, symbol, colour when given), from the check leftwards
RATING_BUTTONS = ((Rating.DISLIKE, "rate-down", "▼", theme.STATUS_ERROR),
                  (Rating.LIKE, "rate-up", "▲", theme.STATUS_OK))
REDRAW_DELAY_MS = 60      # while the window is resized, the picture is redrawn once it settles
ELLIPSIS = "…"
PICTURE, MARKS = "picture", "marks"   # canvas tags: what changes with the size / the selection


def _fit_text(font, text, width):
    """The text, cut with an ellipsis to fit `width` pixels."""
    if font.measure(text) <= width:
        return text
    while text and font.measure(text + ELLIPSIS) > width:
        text = text[:-1]
    return text + ELLIPSIS


def _on_dark_band(image, size):
    """The picture scaled and cropped to fill `size`, with the text band darkened."""
    if image.mode != "RGB":   # transparent pictures on the tile's colour
        background = Image.new("RGB", image.size, theme.CARD_BG)
        background.paste(image, mask=image.convert("RGBA").getchannel("A"))
        image = background
    picture = ImageOps.fit(image, size, Image.BICUBIC)
    width, height = size
    band = min(OVERLAY_HEIGHT, height)
    shade = Image.linear_gradient("L").resize((width, band)).point(
        lambda v: round(OVERLAY_DARKNESS * (v / 255) ** OVERLAY_CURVE))
    region = picture.crop((0, height - band, width, height))
    picture.paste(Image.composite(Image.new("RGB", region.size), region, shade), (0, height - band))
    return picture


class ResultCard(ResultTile):
    """Placed by the layout at any size; the picture (set_source, made in the
    background) is fitted to whatever size the tile has."""

    _fonts = None   # (name, details), made once Tk runs

    def __init__(self, parent, result, selected, callbacks, drag_type=None):
        super().__init__(parent, result, selected, callbacks, drag_type, bg=theme.CARD_BG)
        if ResultCard._fonts is None:
            ResultCard._fonts = (tkfont.Font(font=theme.FONT_SMALL_BOLD), tkfont.Font(font=theme.FONT_SMALL))
        self._canvas = canvas = tk.Canvas(self, bg=theme.CARD_BG, highlightthickness=0, bd=0)
        canvas.pack(fill="both", expand=True)
        self._source = None
        self._photo = None
        self._hover = False
        self._redraw_timer = None
        self._make_interactive(canvas, draggable=True)
        self._pressed_rating = None   # the rating button a press began on
        canvas.bind("<ButtonPress-1>", self._note_pressed_rating, add="+")
        canvas.bind("<Enter>", lambda event: self._set_hover(True))
        canvas.bind("<Leave>", lambda event: self._set_hover(False))
        canvas.bind("<Configure>", lambda event: self._redraw_soon())

    @property
    def source(self):
        return self._source

    def set_source(self, image):
        """The picture to show (any size; it is fitted to the tile)."""
        self._source = image
        self._draw()

    def set_selected(self, selected):
        super().set_selected(selected)
        self._draw_marks()

    def show_rating(self):
        self._draw_marks()

    # ── input ────────────────────────────────────────────────────────────────
    def _clicked(self):
        """A click on ▲ / ▼ gives or takes back that rating; anywhere else on the tile it
        selects or deselects it (a double click opens it)."""
        if self._pressed_rating is not None:
            given = self._callbacks.rating_of(self.result)
            self._callbacks.on_rate(self.result, self._pressed_rating.replacing(given))
            return
        self._callbacks.on_toggle(self.path, not self._selected.get())

    def _note_pressed_rating(self, event):
        tags = self._canvas.gettags("current")
        self._pressed_rating = next((mark for mark, tag, _symbol, _colour in RATING_BUTTONS if tag in tags), None)

    def _set_hover(self, hover):
        self._hover = hover
        self._draw_marks()

    # ── drawing ──────────────────────────────────────────────────────────────
    def _redraw_soon(self):
        if self._redraw_timer:
            self.after_cancel(self._redraw_timer)
        self._redraw_timer = self.after(REDRAW_DELAY_MS, self._draw)

    def _draw(self):
        self._redraw_timer = None
        canvas = self._canvas
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 2 or height < 2:
            return
        canvas.delete(PICTURE)
        if self._source is not None:
            self._photo = ImageTk.PhotoImage(_on_dark_band(self._source, (width, height)))
            canvas.create_image(0, 0, image=self._photo, anchor="nw", tags=PICTURE)
        self._draw_text(width, height)
        self._draw_badge()
        self._draw_marks()

    def _draw_text(self, width, height):
        result, canvas = self.result, self._canvas
        name_font, details_font = self._fonts
        room = width - 2 * TEXT_PADDING
        score = f"{result.score:.3f}"
        details = format_details(result)
        bottom = height - TEXT_PADDING
        canvas.create_text(TEXT_PADDING, bottom, anchor="sw", text=_fit_text(details_font, details, room),
                           font=details_font, fill=theme.OVERLAY_FG_MUTED, tags=PICTURE)
        name_bottom = bottom - details_font.metrics("linespace") - LINE_GAP
        score_width = details_font.measure(score) + TEXT_PADDING
        canvas.create_text(TEXT_PADDING, name_bottom, anchor="sw", font=name_font, fill=theme.FG, tags=PICTURE,
                           text=_fit_text(name_font, os.path.basename(self.path), room - score_width))
        canvas.create_text(width - TEXT_PADDING, name_bottom, anchor="se", text=score, font=details_font,
                           fill=theme.OVERLAY_FG_MUTED, tags=PICTURE)

    def _draw_badge(self):
        if self.result.kind != MediaKind.VIDEO:
            return
        canvas = self._canvas
        text = canvas.create_text(TEXT_PADDING + BADGE_PADDING[0], TEXT_PADDING + BADGE_PADDING[1], anchor="nw",
                                  text=video_badge(self.result), font=theme.FONT_TINY_BOLD,
                                  fill=theme.BADGE_FG, tags=PICTURE)
        left, top, right, bottom = canvas.bbox(text)
        box = canvas.create_rectangle(left - BADGE_PADDING[0], top - BADGE_PADDING[1], right + BADGE_PADDING[0],
                                      bottom + BADGE_PADDING[1], fill=theme.ORANGE, width=0, tags=PICTURE)
        canvas.tag_lower(box, text)

    def _draw_marks(self):
        """Border and corner check: accent when selected, a hint while hovered."""
        canvas = self._canvas
        canvas.delete(MARKS)
        width, height = canvas.winfo_width(), canvas.winfo_height()
        selected = self._selected.get()
        if selected or self._hover:
            border = SELECTED_BORDER if selected else HOVER_BORDER
            inset = border / 2
            canvas.create_rectangle(inset, inset, width - inset, height - inset, width=border, tags=MARKS,
                                    outline=theme.ACCENT if selected else theme.FG_MUTED)
            right = width - CHECK_MARGIN
            canvas.create_rectangle(right - CHECK_SIZE, CHECK_MARGIN, right, CHECK_MARGIN + CHECK_SIZE, tags=MARKS,
                                    fill=theme.ACCENT if selected else theme.BG, outline=theme.FG)
            if selected:
                canvas.create_text(right - CHECK_SIZE / 2, CHECK_MARGIN + CHECK_SIZE / 2, text="✓", tags=MARKS,
                                   fill=theme.MENU_ACTIVE_FG, font=theme.FONT_BOLD)
        self._draw_rating(width)

    def _draw_rating(self, width):
        """▲ / ▼ left of the check: both while hovered, otherwise only the one given."""
        canvas, given = self._canvas, self._callbacks.rating_of(self.result)
        right = width - CHECK_MARGIN - CHECK_SIZE - RATING_GAP
        for mark, tag, symbol, colour in RATING_BUTTONS:
            active = given == mark
            if self._hover or active:
                left, middle = right - CHECK_SIZE, CHECK_MARGIN + CHECK_SIZE / 2
                canvas.create_rectangle(left, CHECK_MARGIN, right, CHECK_MARGIN + CHECK_SIZE, tags=(MARKS, tag),
                                        fill=colour if active else theme.BG, outline=theme.FG)
                canvas.create_text((left + right) / 2, middle, text=symbol, font=theme.FONT_SMALL, tags=(MARKS, tag),
                                   fill=theme.BADGE_FG if active else theme.FG)
            right -= CHECK_SIZE + RATING_GAP
