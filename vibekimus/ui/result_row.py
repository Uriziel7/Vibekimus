"""One search result as a compact row of a list: selection box, thumbnail, then the
file's name, folder and details. Used by the duplicates list, where the row of the
copy to keep is followed by the rows of its duplicates."""
import os
import tkinter as tk

from PIL import ImageTk

from vibekimus.media_kind import MediaKind
from vibekimus.ui import theme
from vibekimus.ui.formatting import format_details, format_moment
from vibekimus.ui.result_tile import ResultTile

ROW_PADDING = 4          # around the thumbnail, so rows touch without wasted space
TEXT_GAP = 10            # between the thumbnail and the text
KEEPER_TEXT = "Keep  ·  best copy (largest resolution, then file, then oldest)"
COPY_TEXT = "Duplicate  ·  {score:.3f} alike"


def row_height(thumbnail_size):
    return thumbnail_size + 2 * ROW_PADDING


def _details(result):
    details = format_details(result)
    if result.kind == MediaKind.VIDEO:
        moment = f"t={format_moment(result.timestamp)}"
        return f"{moment}  ·  {details}" if details else moment
    return details


class ResultRow(ResultTile):
    """A row of fixed height, so a list can be laid out before the thumbnails arrive
    (set_thumbnail fills the picture in later)."""

    def __init__(self, parent, result, selected, callbacks, thumbnail_size, background, drag_type=None):
        super().__init__(parent, result, selected, callbacks, drag_type, bg=background,
                         height=row_height(thumbnail_size))
        self.pack_propagate(False)
        self._photo = None
        self._select_box(self, background, text="").pack(side="left", padx=(ROW_PADDING, 0))
        frame = tk.Frame(self, bg=theme.BG, width=thumbnail_size, height=thumbnail_size)
        frame.pack_propagate(False)
        frame.pack(side="left", padx=ROW_PADDING, pady=ROW_PADDING)
        self._picture = tk.Label(frame, bg=theme.BG, cursor="hand2")
        self._picture.pack(expand=True)
        self._make_interactive(frame)
        self._make_interactive(self._picture, draggable=True)
        self._make_interactive(self)
        self._add_text(background)

    def _add_text(self, background):
        result = self.result
        text = tk.Frame(self, bg=background)
        text.pack(side="left", fill="both", expand=True, padx=(TEXT_GAP - ROW_PADDING, 0))
        role, colour = ((KEEPER_TEXT, theme.STATUS_OK) if result.keeper
                        else (COPY_TEXT.format(score=result.score), theme.ORANGE))
        lines = ((os.path.basename(self.path), theme.FONT_BOLD, theme.FG),
                 (os.path.dirname(self.path), theme.FONT_SMALL, theme.FG_MUTED),
                 (_details(result), theme.FONT_SMALL, theme.FG),
                 (role, theme.FONT_SMALL, colour))
        holder = tk.Frame(text, bg=background)
        holder.pack(side="left", anchor="w")   # the lines centred vertically, aligned left
        for line, font, colour in lines:
            label = tk.Label(holder, text=line, font=font, fg=colour, bg=background, anchor="w")
            label.pack(fill="x")
            self._make_interactive(label)
        self._make_interactive(text)
        self._make_interactive(holder)

    def set_thumbnail(self, image):
        self._photo = ImageTk.PhotoImage(image)   # kept here so it is not garbage-collected
        self._picture.config(image=self._photo)
