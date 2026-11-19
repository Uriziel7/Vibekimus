"""Choices the user makes in the program (as opposed to settings.py, edited by hand).
Kept in a small JSON file next to the program.

Each preference is described once, as a field of Preferences: its default, how a saved value
is read back (anything not valid gives the default) and how it is written to the file."""
import json
from dataclasses import dataclass, field, fields
from operator import attrgetter
from typing import Callable

from vibekimus import advanced_options, settings
from vibekimus.decoding import DecodeMode, DecodingOptions
from vibekimus.encoders.catalog import model_by_key
from vibekimus.files import write_atomically
from vibekimus.indexing.options import IndexingOptions
from vibekimus.runlog import safe_print

STORED = "stored"   # the key of a field's metadata: how it is kept in the file


def _as_is(value):
    return value


@dataclass(frozen=True)
class _Stored:
    read: Callable    # (saved value, default) -> the value, or the default when the saved one is not valid
    write: Callable   # value -> what the file keeps


def _preference(default, read, write=_as_is):
    """A field of Preferences; default() gives its value when nothing (valid) is saved."""
    return field(default_factory=default, metadata={STORED: _Stored(read, write)})


def _stored(preference):
    return preference.metadata[STORED]


def _seconds(value, default):
    """A non-negative whole number of seconds, or the default."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return value


def _flag(value, default):
    """A true/false value, or the default."""
    return value if isinstance(value, bool) else default


def _thumbnail_size(value, default):
    """One of the zoom levels, or the default."""
    return value if not isinstance(value, bool) and value in settings.THUMBNAIL_ZOOM_LEVELS else default


def _advanced(value, _default):
    return advanced_options.valid_changes(value if isinstance(value, dict) else {})


def _default_decoding():
    return DecodingOptions(
        images=DecodeMode.parse(settings.DEFAULT_IMAGE_DECODING, DecodeMode.CPU),
        videos=DecodeMode.parse(settings.DEFAULT_VIDEO_DECODING, DecodeMode.CPU),
    )


def _decoding(value, default):
    saved = value if isinstance(value, dict) else {}
    return DecodingOptions(images=DecodeMode.parse(saved.get("images"), default.images),
                           videos=DecodeMode.parse(saved.get("videos"), default.videos))


def _decoding_data(decoding):
    return {"images": decoding.images.value, "videos": decoding.videos.value}


@dataclass
class Preferences:
    decoding: DecodingOptions = _preference(_default_decoding, _decoding, _decoding_data)
    # encoders.catalog.ModelSpec, kept by its key
    model: object = _preference(lambda: model_by_key(settings.DEFAULT_MODEL), model_by_key, attrgetter("key"))
    # Videos longer than this are not indexed; 0 = no limit.
    max_video_seconds: int = _preference(lambda: settings.VIDEO_MAX_DURATION_SECONDS, _seconds)
    # Refresh and scans index videos too (otherwise images only).
    index_videos: bool = _preference(lambda: settings.DEFAULT_INDEX_VIDEOS, _flag)
    # Refresh and scans also read the text on pictures (OCR).
    read_text: bool = _preference(lambda: settings.DEFAULT_READ_TEXT, _flag)
    # Index files are encrypted (key in settings.KEY_FILE).
    encrypt_indexes: bool = _preference(lambda: settings.DEFAULT_ENCRYPT_INDEXES, _flag)
    # Thumbnail size in the results (Ctrl + mouse wheel), one of THUMBNAIL_ZOOM_LEVELS.
    thumbnail_size: int = _preference(lambda: settings.DEFAULT_THUMBNAIL_SIZE, _thumbnail_size)
    # The model's weights in 16 bits on the graphics card (about half the memory).
    vram_save: bool = _preference(lambda: settings.DEFAULT_VRAM_SAVE, _flag)
    # Advanced settings changed from settings.py's values: {settings name: value}.
    advanced: dict = _preference(dict, _advanced)

    def indexing_options(self):
        return IndexingOptions(decoding=self.decoding, max_video_seconds=self.max_video_seconds)

    @classmethod
    def load(cls):
        """The saved preferences, or the defaults where nothing (valid) is saved."""
        defaults = cls()
        try:
            with open(settings.PREFERENCES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return defaults
        except Exception as error:
            safe_print(f"[PREFERENCES] Ignoring unreadable file: {error}")
            return defaults
        if not isinstance(data, dict):
            return defaults
        values = {}
        for preference in fields(cls):
            name = preference.name
            values[name] = _stored(preference).read(data.get(name), getattr(defaults, name))
        return cls(**values)

    def save(self):
        data = {preference.name: _stored(preference).write(getattr(self, preference.name))
                for preference in fields(self)}
        try:   # through a temporary file: a crash while writing must not cost every setting
            write_atomically(settings.PREFERENCES_FILE, json.dumps(data, indent=2).encode("utf-8"))
        except OSError as error:
            safe_print(f"[PREFERENCES] Could not save: {error}")
