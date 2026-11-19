"""Windows-Explorer-style rubber-band selection over the results grid."""
import os
import tkinter as tk

from vibekimus.ui import theme

DRAG_THRESHOLD_PX = 5          # below this a press + release is still a click
CTRL_STATE_MASK = 0x4          # event.state bit set while Ctrl is held
# Windows: this near-black colour is declared transparent, so only the drawn
# rectangle of the overlay window is visible.
TRANSPARENT_COLOUR = "#000002"
BAND_STYLE = dict(outline=theme.ACCENT, fill="", width=2, dash=(4, 2))


class RubberBandSelector:
    """Drag on the empty area of the grid to select (or, with Ctrl, deselect) cards.

    Press inside the canvas but not on a card starts tracking; moving more than
    DRAG_THRESHOLD_PX turns it into a drag and draws the rectangle; release selects
    every card the rectangle touches.

    The handlers are bound on the whole application: tkinter's implicit pointer grab
    delivers motion and release to whichever widget received the press, which is
    rarely the grid itself.

    The rectangle is drawn on a transparent top-level window, because canvas items
    are always rendered behind widgets embedded in the canvas (the cards). Where
    that overlay cannot be made it is drawn on the canvas.
    """

    def __init__(self, root, canvas, get_cards, on_select):
        self._root = root
        self._canvas = canvas
        self._get_cards = get_cards   # the tiles that can be seen (cards, or rows of open groups)
        self._on_select = on_select              # on_select(paths, select: bool)
        self._start = (0, 0)
        self._tracking = False   # from press until release
        self._dragging = False   # once the threshold is crossed
        self._canvas_band = None
        self._overlay, self._overlay_canvas, self._overlay_band = self._create_overlay(), None, None
        if self._overlay is not None:
            self._overlay_canvas = tk.Canvas(self._overlay, bg=TRANSPARENT_COLOUR, highlightthickness=0)
            self._overlay_canvas.pack(fill="both", expand=True)

        root.bind_all("<ButtonPress-1>", self._on_press)
        root.bind_all("<B1-Motion>", self._on_drag)
        root.bind_all("<ButtonRelease-1>", self._on_release)

    def _create_overlay(self):
        if os.name != "nt":
            return None
        try:
            overlay = tk.Toplevel(self._root)
            overlay.overrideredirect(True)
            overlay.withdraw()
            overlay.attributes("-transparentcolor", TRANSPARENT_COLOUR)
            return overlay
        except Exception:
            return None

    # ── geometry ─────────────────────────────────────────────────────────────
    def _to_canvas(self, x_root, y_root):
        """Screen coordinates → scroll-adjusted canvas coordinates."""
        canvas = self._canvas
        return (canvas.canvasx(x_root - canvas.winfo_rootx()),
                canvas.canvasy(y_root - canvas.winfo_rooty()))

    @staticmethod
    def _contains(widget, x_root, y_root):
        left, top = widget.winfo_rootx(), widget.winfo_rooty()
        return (left <= x_root <= left + widget.winfo_width()
                and top <= y_root <= top + widget.winfo_height())

    def _over_a_card(self, x_root, y_root):
        for card in self._get_cards():
            try:
                if self._contains(card, x_root, y_root):
                    return True
            except Exception:
                continue
        return False

    # ── events ───────────────────────────────────────────────────────────────
    def _on_press(self, event):
        if not self._contains(self._canvas, event.x_root, event.y_root):
            return
        if self._over_a_card(event.x_root, event.y_root):
            return   # the card handles its own clicks
        self._tracking = True
        self._dragging = False
        self._start = self._to_canvas(event.x_root, event.y_root)
        self._erase_canvas_band()

    def _on_drag(self, event):
        if not self._tracking:
            return
        x, y = self._to_canvas(event.x_root, event.y_root)
        if not self._dragging:
            if abs(x - self._start[0]) < DRAG_THRESHOLD_PX and abs(y - self._start[1]) < DRAG_THRESHOLD_PX:
                return
            self._dragging = True
            self._show_overlay()
        self._draw_band(x, y)

    def _on_release(self, event):
        if not self._tracking:
            return
        was_dragging = self._dragging
        self._tracking = self._dragging = False
        self._erase_canvas_band()
        self._hide_overlay()
        if not was_dragging:
            return   # a plain click — nothing to do
        x, y = self._to_canvas(event.x_root, event.y_root)
        left, right = sorted((self._start[0], x))
        top, bottom = sorted((self._start[1], y))
        if right - left < DRAG_THRESHOLD_PX and bottom - top < DRAG_THRESHOLD_PX:
            return
        deselect = bool(event.state & CTRL_STATE_MASK)
        self._on_select(self._paths_touching(left, top, right, bottom), not deselect)

    def _paths_touching(self, left, top, right, bottom):
        paths = set()
        for card in self._get_cards():
            try:
                # Screen position → canvas-scroll coordinates (tiles may sit inside groups)
                card_left, card_top = self._to_canvas(card.winfo_rootx(), card.winfo_rooty())
                if (card_left < right and card_left + card.winfo_width() > left
                        and card_top < bottom and card_top + card.winfo_height() > top):
                    paths.add(card.path)
            except Exception:
                pass
        return paths

    # ── drawing ──────────────────────────────────────────────────────────────
    def _show_overlay(self):
        """Position the overlay once per drag — doing it on every motion event flickers."""
        if self._overlay is None:
            return
        canvas = self._canvas
        self._overlay.geometry(f"{canvas.winfo_width()}x{canvas.winfo_height()}"
                               f"+{canvas.winfo_rootx()}+{canvas.winfo_rooty()}")
        self._overlay.deiconify()
        self._overlay.lift()

    def _hide_overlay(self):
        if self._overlay is None:
            return
        if self._overlay_band:
            self._overlay_canvas.delete(self._overlay_band)
            self._overlay_band = None
        self._overlay.withdraw()

    def _erase_canvas_band(self):
        if self._canvas_band:
            self._canvas.delete(self._canvas_band)
            self._canvas_band = None

    def _draw_band(self, x, y):
        start_x, start_y = self._start
        if self._overlay is None:
            self._erase_canvas_band()
            self._canvas_band = self._canvas.create_rectangle(start_x, start_y, x, y, **BAND_STYLE)
            return
        # The overlay's top-left corner is the canvas widget's top-left on screen.
        scroll_x, scroll_y = self._canvas.canvasx(0), self._canvas.canvasy(0)
        if self._overlay_band:
            self._overlay_canvas.delete(self._overlay_band)
        self._overlay_band = self._overlay_canvas.create_rectangle(
            start_x - scroll_x, start_y - scroll_y, x - scroll_x, y - scroll_y, **BAND_STYLE)
