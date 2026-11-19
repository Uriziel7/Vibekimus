"""The Settings window: which model is used and how files are indexed (General), and the
fine-tuning of settings.py (Advanced). It can be resized; both pages scroll."""
import math
import tkinter as tk
from dataclasses import replace
from tkinter import messagebox, ttk

from vibekimus.advanced_options import InvalidValue
from vibekimus.decoding import DecodeMode, DecodingOptions
from vibekimus.encoders.catalog import MODELS, model_by_key
from vibekimus.ui import theme
from vibekimus.ui.advanced_settings import AdvancedSettingsPage
from vibekimus.ui.widgets import ScrollableFrame
from vibekimus.ui.windows import new_window
from vibekimus.units import SECONDS_PER_MINUTE

GEOMETRY = "640x760"
MIN_WIDTH, MIN_HEIGHT = 420, 360
MAX_VIDEO_MINUTES = 600          # upper end of the spin box (10 hours)
MINUTES_FIELD_WIDTH = 7
VIDEO_LENGTH_NOTE = ("0 = index videos of any length. Raising the limit later indexes the videos that "
                     "were skipped; lowering it keeps videos that are already in the index.")
INVALID_MINUTES_MESSAGE = "Enter the longest video length in minutes, e.g. 1, 2.5 or 0 for no limit."
INVALID_ADVANCED_MESSAGE = "These advanced settings take other values:\n\n{errors}"


def parse_minutes(text):
    """Seconds for a number of minutes typed by the user (comma or dot), or None."""
    try:
        minutes = float(text.strip().replace(",", "."))
    except ValueError:
        return None
    if minutes < 0 or not math.isfinite(minutes):
        return None
    return round(minutes * SECONDS_PER_MINUTE)


def format_minutes(seconds):
    minutes = seconds / SECONDS_PER_MINUTE
    return f"{minutes:g}"
MODEL_NOTE = ("Each model keeps its own index: after switching, index (Refresh) once with the new "
              "model. The previous model's index stays and is used again when you switch back.")
IMAGE_CHOICES = (
    (DecodeMode.CPU.value, "CPU", "Recommended. Fastest for camera photos: the CPU decodes JPEGs at reduced size."),
    (DecodeMode.GPU.value, "GPU (JPEG)", "The graphics card decodes every JPEG; other formats stay on the CPU."),
    (DecodeMode.HYBRID.value, "CPU + GPU", "Both share the JPEGs. Slightly faster for libraries of mostly PNG files."),
)
VIDEO_CHOICES = (
    (DecodeMode.CPU.value, "CPU", "Software decoding."),
    (DecodeMode.GPU.value, "GPU", "Recommended. The graphics card's video decoder; files it cannot read fall "
                                  "back to the CPU."),
    (DecodeMode.HYBRID.value, "CPU + GPU", "Every other video on each."),
)
VRAM_SAVE_TEXT = "VRAM Save"
VRAM_SAVE_NOTE = ("Keeps SigLIP 2's weights in 16 bits on the graphics card: about half the graphics memory "
                  "(3.2 instead of 6.4 GB while indexing), about 5% slower, results practically the same. "
                  "For graphics cards with little memory, RTX 20 or newer.")
DECODING_NOTE = ("The encoder itself always runs on the GPU. These settings only decide where files are "
                 "decoded before it, and apply from the next indexing run.")


class SettingsDialog:
    """Edits a copy of the preferences and hands the result to on_save."""

    def __init__(self, parent, preferences, on_save):
        self._preferences = preferences
        self._on_save = on_save
        self._window = window = new_window(parent, "Settings", geometry=GEOMETRY, min_size=(MIN_WIDTH, MIN_HEIGHT),
                                           modal=True, close_on_escape=True)

        buttons = ttk.Frame(window, padding=theme.PAD * 2)
        buttons.pack(side="bottom", fill="x")   # packed first: a small window shrinks the pages, not these
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="right")
        ttk.Button(buttons, text="Save", command=self._save, style=theme.PRIMARY_BUTTON).pack(
            side="right", padx=(0, theme.GAP))
        self._notebook = notebook = ttk.Notebook(window)
        notebook.pack(fill="both", expand=True)
        self._pages = [ScrollableFrame(notebook), ScrollableFrame(notebook)]
        notebook.add(self._pages[0], text="General")
        notebook.add(self._pages[1], text="Advanced")
        self._build_general(self._pages[0], preferences)
        self._advanced = AdvancedSettingsPage(self._pages[1], preferences.advanced)
        window.bind("<MouseWheel>", self._scroll)

    def _build_general(self, page, preferences):
        body = page.body
        model_choices = [(spec.key, spec.label, spec.description) for spec in MODELS]
        self._model = self._choice_group(page, "Model", model_choices, preferences.model.key)
        self._note(page, MODEL_NOTE)
        self._vram_save = tk.BooleanVar(value=preferences.vram_save)
        ttk.Checkbutton(body, text=VRAM_SAVE_TEXT, variable=self._vram_save).pack(anchor="w")
        self._note(page, VRAM_SAVE_NOTE)
        self._images = self._choice_group(page, "Image decoding", IMAGE_CHOICES, preferences.decoding.images.value)
        self._videos = self._choice_group(page, "Video decoding", VIDEO_CHOICES, preferences.decoding.videos.value)
        self._note(page, DECODING_NOTE)
        self._video_minutes = self._video_length_field(body, preferences.max_video_seconds)
        self._note(page, VIDEO_LENGTH_NOTE)

    def _scroll(self, event):
        """The mouse wheel scrolls the page shown (not the results behind the window)."""
        self._pages[self._notebook.index("current")].scroll(event)
        return "break"

    @staticmethod
    def _choice_group(page, title, choices, current):
        """Radio buttons for (value, label, description) choices; returns the variable."""
        ttk.Label(page.body, text=title, font=theme.FONT_HEADING).pack(anchor="w", pady=(theme.PAD, theme.GAP))
        variable = tk.StringVar(value=current)
        for value, label, description in choices:
            ttk.Radiobutton(page.body, text=label, value=value, variable=variable).pack(anchor="w")
            page.wrapping_label(page.body, description, indent=theme.GROUP_GAP, style=theme.MUTED_LABEL).pack(
                anchor="w", padx=(theme.GROUP_GAP, 0), pady=(0, theme.GAP))
        return variable

    @staticmethod
    def _video_length_field(parent, seconds):
        ttk.Label(parent, text="Video length", font=theme.FONT_HEADING).pack(anchor="w", pady=(theme.PAD, theme.GAP))
        row = ttk.Frame(parent)
        row.pack(anchor="w")
        ttk.Label(row, text="Skip videos longer than").pack(side="left")
        variable = tk.StringVar(value=format_minutes(seconds))
        ttk.Spinbox(row, from_=0, to=MAX_VIDEO_MINUTES, increment=1, textvariable=variable,
                    width=MINUTES_FIELD_WIDTH).pack(side="left", padx=theme.GAP)
        ttk.Label(row, text="minutes").pack(side="left")
        return variable

    @staticmethod
    def _note(page, text):
        page.wrapping_label(page.body, text, style=theme.MUTED_LABEL).pack(anchor="w", pady=(0, theme.PAD))

    def _save(self):
        max_video_seconds = parse_minutes(self._video_minutes.get())
        if max_video_seconds is None:
            self._notebook.select(0)
            messagebox.showerror("Video length", INVALID_MINUTES_MESSAGE, parent=self._window)
            return
        try:
            advanced = self._advanced.changes()
        except InvalidValue as error:
            self._notebook.select(1)
            messagebox.showerror("Advanced settings", INVALID_ADVANCED_MESSAGE.format(errors=error),
                                 parent=self._window)
            return
        self._on_save(replace(
            self._preferences,
            model=model_by_key(self._model.get(), self._preferences.model),
            decoding=DecodingOptions(images=DecodeMode(self._images.get()), videos=DecodeMode(self._videos.get())),
            max_video_seconds=max_video_seconds,
            vram_save=self._vram_save.get(),
            advanced=advanced,
        ))
        self._window.destroy()
