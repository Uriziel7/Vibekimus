"""The row of filters: media type, dates, resolution, file size, folder and sorting."""
import os
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, ttk

from vibekimus.filters import ResultFilters, SortOrder, parse_date
from vibekimus.media_types import MEDIA_TYPES
from vibekimus.ui import theme
from vibekimus.ui.flow import FlowFrame
from vibekimus.ui.widgets import ControlGroup
from vibekimus.units import BYTES_PER_KB, BYTES_PER_MB

ANY_TYPE = ("Any type", None)
TYPE_CHOICES = (ANY_TYPE,) + tuple((media_type.label, media_type.key) for media_type in MEDIA_TYPES)
RESOLUTION_CHOICES = (("Any resolution", 0), ("720p or more", 1280 * 720),
                      ("1080p or more", 1920 * 1080), ("4K or more", 3840 * 2160))
SIZE_CHOICES = (("Any file size", 0), ("100 KB or more", 100 * BYTES_PER_KB), ("1 MB or more", BYTES_PER_MB),
                ("5 MB or more", 5 * BYTES_PER_MB), ("20 MB or more", 20 * BYTES_PER_MB))
SORT_CHOICES = (("Best match first", SortOrder.RELEVANCE), ("Newest first", SortOrder.NEWEST),
                ("Oldest first", SortOrder.OLDEST), ("Largest file first", SortOrder.LARGEST_FILE),
                ("Highest resolution first", SortOrder.MOST_PIXELS),
                ("Most space to free first", SortOrder.SPACE_FREED))
DATE_FIELD_WIDTH = 10
FOLDER_NAME_CHARS = 24
ANY_FOLDER = "Any folder"
CLEAR_BUTTON_CHARS = 2


@dataclass(frozen=True)
class FilterChoice:
    filters: ResultFilters
    media_type: str     # None = any
    sort: SortOrder

    @property
    def needs_file_info(self):
        """Whether the choice depends on dates, sizes or dimensions stored in the index."""
        f = self.filters
        return (f.date_from is not None or f.date_to is not None or bool(f.min_pixels or f.min_bytes)
                or self.sort is not SortOrder.RELEVANCE)


class _Choice:
    """A read-only drop-down of (label, value) pairs; on_change runs when the user
    picks an entry."""

    def __init__(self, parent, choices, width, on_change):
        self._choices = dict(choices)
        self._first = choices[0][0]
        self.variable = tk.StringVar(value=self._first)
        box = ttk.Combobox(parent, textvariable=self.variable, values=[label for label, _ in choices],
                           state="readonly", width=width)
        box.pack(side="left", padx=(theme.GAP, 0))
        box.bind("<<ComboboxSelected>>", lambda event: on_change())

    @property
    def value(self):
        return self._choices[self.variable.get()]

    def reset(self):
        self.variable.set(self._first)


def _widest(choices):
    return max(len(label) for label, _ in choices)


class FilterBar:
    """on_change runs whenever the user changes a filter: a drop-down pick, a typed
    date confirmed with Enter or by leaving its field, a folder, or Reset."""

    def __init__(self, parent, on_change):
        self._on_change = on_change
        bar = FlowFrame(parent)
        bar.pack(fill="x", pady=(theme.ROW_GAP, 0))
        self._type = _Choice(bar.add(ControlGroup(bar)), TYPE_CHOICES, _widest(TYPE_CHOICES), on_change)
        dates = bar.add(ControlGroup(bar))
        dates.add(ttk.Label(dates, text="Date"))
        self._date_from = self._date_field(dates)
        dates.add(ttk.Label(dates, text="to"))
        self._date_to = self._date_field(dates)
        self._applied_dates = self._typed_dates()
        sizes = bar.add(ControlGroup(bar))
        self._resolution = _Choice(sizes, RESOLUTION_CHOICES, _widest(RESOLUTION_CHOICES), on_change)
        self._size = _Choice(sizes, SIZE_CHOICES, _widest(SIZE_CHOICES), on_change)
        folder = bar.add(ControlGroup(bar))
        self._folder = None
        self._folder_button = folder.button(ANY_FOLDER, self._choose_folder)
        folder.button("×", self._on_clear_folder_click, width=CLEAR_BUTTON_CHARS)
        ordering = bar.add(ControlGroup(bar))
        self._sort = _Choice(ordering, SORT_CHOICES, _widest(SORT_CHOICES), on_change)
        ordering.button("Reset filters", self._on_reset_click)

    def _date_field(self, group):
        variable = tk.StringVar()
        entry = group.add(ttk.Entry(group, textvariable=variable, width=DATE_FIELD_WIDTH))
        for event in ("<Return>", "<FocusOut>"):
            entry.bind(event, lambda _event: self._on_dates_confirmed())
        return variable

    def _typed_dates(self):
        return self._date_from.get().strip(), self._date_to.get().strip()

    def _on_dates_confirmed(self):
        """Typing alone changes nothing; a date counts once confirmed, and only if it
        differs from the one applied last."""
        if self._typed_dates() != self._applied_dates:
            self._applied_dates = self._typed_dates()
            self._on_change()

    def _choose_folder(self):
        folder = filedialog.askdirectory(title="Only results inside this folder")
        if folder:
            self._folder = os.path.normpath(folder)
            name = os.path.basename(self._folder) or self._folder
            self._folder_button.config(text="In: " + name[:FOLDER_NAME_CHARS])
            self._on_change()

    def _on_clear_folder_click(self):
        if self._folder is not None:
            self._clear_folder()
            self._on_change()

    def _clear_folder(self):
        self._folder = None
        self._folder_button.config(text=ANY_FOLDER)

    def _on_reset_click(self):
        self.reset()
        self._on_change()

    def read(self):
        """The current choice. Raises ValueError (with a message for the user) when a
        typed date cannot be read."""
        filters = ResultFilters(
            date_from=parse_date(self._date_from.get()),
            date_to=parse_date(self._date_to.get(), end=True),
            min_pixels=self._resolution.value,
            min_bytes=self._size.value,
            folder=self._folder,
        )
        return FilterChoice(filters, self._type.value, self._sort.value)

    def reset(self):
        """Back to no filters, without running on_change."""
        for choice in (self._type, self._resolution, self._size, self._sort):
            choice.reset()
        self._date_from.set("")
        self._date_to.set("")
        self._applied_dates = self._typed_dates()
        self._clear_folder()
