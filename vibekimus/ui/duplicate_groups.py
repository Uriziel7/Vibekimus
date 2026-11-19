"""One group of the duplicates list: the copy to keep, and under it a bar that opens
(with an animation) the list of its duplicates."""
import tkinter as tk

from vibekimus.duplicates import space_freed
from vibekimus.ui import theme
from vibekimus.ui.collapsible import Collapsible
from vibekimus.ui.formatting import format_size

COPIES_INDENT = 36        # the duplicates' rows are indented under the kept copy
BAR_PADDING = (8, 3)      # horizontal, vertical padding of the open/close bar
OPEN_ARROW, CLOSED_ARROW = "▼", "▶"


def _bar_text(group, is_open):
    copies = len(group) - 1
    noun = "duplicate" if copies == 1 else "duplicates"
    action = "hide" if is_open else "show"
    return (f"{OPEN_ARROW if is_open else CLOSED_ARROW}  {copies} {noun}  ·  "
            f"{format_size(space_freed(group))} to free  ·  click to {action}")


class DuplicateGroupView(tk.Frame):
    """make_row(parent, result, background) builds a ResultRow; load_thumbnails(rows)
    fills their pictures in; on_resize runs while the group opens or closes;
    on_toggled(group number, is_open) lets the list remember open groups;
    select_paths(paths, select) selects or deselects files."""

    def __init__(self, parent, group, make_row, load_thumbnails, on_resize, on_toggled, select_paths):
        super().__init__(parent, bg=theme.BG)
        self.group = group
        self._make_row = make_row
        self._load_thumbnails = load_thumbnails
        self._on_toggled = on_toggled
        self.keeper_row = make_row(self, group[0], theme.CARD_BG)
        self.keeper_row.pack(fill="x")
        self._build_bar(select_paths)
        self._copies = Collapsible(self, on_step=on_resize, bg=theme.BG)
        self._copies.pack(fill="x")
        self._copy_rows = None   # made when the group is first opened

    def _build_bar(self, select_paths):
        """The open/close text on the left, "Select whole group" on the right."""
        bar = tk.Frame(self, bg=theme.PANEL_BG)
        bar.pack(fill="x")
        self._whole_group = tk.BooleanVar(value=False)
        paths = {result.path for result in self.group}
        tk.Checkbutton(bar, text="Select whole group", variable=self._whole_group, font=theme.FONT_SMALL,
                       bg=theme.PANEL_BG, fg=theme.FG, selectcolor=theme.BG, activebackground=theme.PANEL_BG,
                       command=lambda: select_paths(paths, self._whole_group.get())).pack(
            side="right", padx=BAR_PADDING[0])
        self._bar = tk.Label(bar, text=_bar_text(self.group, False), font=theme.FONT_SMALL, fg=theme.ACCENT_HOVER,
                             bg=theme.PANEL_BG, anchor="w", cursor="hand2",
                             padx=BAR_PADDING[0], pady=BAR_PADDING[1])
        self._bar.pack(side="left", fill="x", expand=True)
        self._bar.bind("<Button-1>", lambda event: self.toggle())

    def show_selection(self, selected):
        """Tick "Select whole group" exactly when every file of the group is selected."""
        self._whole_group.set(all(result.path in selected for result in self.group))

    def rows(self):
        """Every row made so far (the duplicates' rows only once opened)."""
        return [self.keeper_row] + (self._copy_rows or [])

    def open_rows(self):
        """The rows that can be seen: the duplicates' only while the group is open."""
        return [self.keeper_row] + (self._copy_rows if self._copies.is_open else [])

    def toggle(self, animate=True):
        if self._copies.is_open:
            self._copies.close(animate)
        else:
            self._make_copy_rows()
            self._copies.open(animate)
        self._bar.config(text=_bar_text(self.group, self._copies.is_open))
        self._on_toggled(self.group[0].group, self._copies.is_open)

    def _make_copy_rows(self):
        if self._copy_rows is not None:
            return
        self._copy_rows = []
        for result in self.group[1:]:
            row = self._make_row(self._copies.body, result, theme.PANEL_BG)
            row.pack(fill="x", padx=(COPIES_INDENT, 0))
            self._copy_rows.append(row)
        self._load_thumbnails(self._copy_rows)
