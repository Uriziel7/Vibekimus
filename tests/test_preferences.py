"""Preferences saved and loaded, and the advanced settings (settings.py values changed in the
Settings window)."""
import json
from dataclasses import replace

import pytest

from support import raised
from vibekimus import advanced_options as advanced
from vibekimus import settings
from vibekimus.decoding import DecodeMode, DecodingOptions
from vibekimus.encoders.catalog import SIGLIP2_SO400M, model_by_key
from vibekimus.indexing.options import IndexingOptions
from vibekimus.preferences import Preferences
from vibekimus.ui.settings_dialog import format_minutes, parse_minutes

VIDEO_SECONDS = 150


def write_preferences(content):
    with open(settings.PREFERENCES_FILE, "w", encoding="utf-8") as f:
        f.write(content if isinstance(content, str) else json.dumps(content))


@pytest.fixture(autouse=True)
def advanced_defaults():
    """Every test ends with settings.py's own values."""
    yield
    advanced.apply({})


def test_every_preference_is_saved_and_loaded():
    chosen = Preferences(DecodingOptions(images=DecodeMode.HYBRID, videos=DecodeMode.CPU), SIGLIP2_SO400M,
                         VIDEO_SECONDS, index_videos=not settings.DEFAULT_INDEX_VIDEOS,
                         read_text=not settings.DEFAULT_READ_TEXT, thumbnail_size=settings.THUMBNAIL_ZOOM_LEVELS[-1],
                         vram_save=not settings.DEFAULT_VRAM_SAVE)
    chosen.save()
    assert Preferences.load() == chosen
    assert Preferences.load().indexing_options() == IndexingOptions(chosen.decoding, VIDEO_SECONDS)


def test_a_model_no_longer_offered_gives_the_default_model():
    write_preferences({"model": "removed-model"})
    assert Preferences.load().model == model_by_key(settings.DEFAULT_MODEL)


def test_a_negative_video_length_gives_the_default():
    write_preferences({"max_video_seconds": -5})
    assert Preferences.load().max_video_seconds == settings.VIDEO_MAX_DURATION_SECONDS


def test_a_damaged_preferences_file_gives_the_defaults():
    write_preferences("{ not json")
    assert Preferences.load() == Preferences()


def test_an_unknown_value_gives_the_default_for_that_value():
    write_preferences({"decoding": {"images": "quantum"}, "index_videos": "yes", "thumbnail_size": 9999})
    assert Preferences.load() == Preferences()


def test_an_unknown_model_key_falls_back_to_the_default_model():
    assert model_by_key("no-such-model") == SIGLIP2_SO400M


def test_minutes_typed_in_settings():
    typed = ("1", "2,5", " 0.5 ", "0", "-1", "abc", "", "inf", "nan")
    assert [parse_minutes(text) for text in typed] == [60, 150, 30, 0, None, None, None, None, None]
    assert format_minutes(60) == "1" and format_minutes(90) == "1.5"


def test_typed_values_are_read_as_the_option_takes_them_and_wrong_ones_refused():
    workers, weight = advanced.BY_NAME["LOAD_WORKERS"], advanced.BY_NAME["COMBINED_TEXT_WEIGHT"]
    drives = advanced.BY_NAME["STORAGE_MODE"]
    assert workers.parse(" 12 ") == 12 and weight.parse("0,4") == 0.4 and drives.parse("hdd") == "hdd"
    wrong = ((workers, "1.5"), (workers, "0"), (workers, "lots"), (weight, "2"), (weight, "nan"), (drives, "floppy"),
             (advanced.BY_NAME["GPU_SEARCH"], "yes"))
    errors = [raised(lambda o=option, v=value: o.parse(v)) for option, value in wrong]
    assert all(isinstance(error, advanced.InvalidValue) for error in errors), errors


KEPT = {"LOAD_WORKERS": 3, "COMBINED_TEXT_WEIGHT": 0.7}


def test_only_known_valid_values_that_differ_from_settings_py_are_kept():
    saved = {"LOAD_WORKERS": 3, "COMBINED_TEXT_WEIGHT": "0.7", "NO_SUCH_OPTION": 1, "VIDEO_WORKERS": 999,
             "DOUBLE_CLICK_MS": settings.DOUBLE_CLICK_MS}
    assert advanced.valid_changes(saved) == KEPT


def test_changes_are_applied_to_the_running_program_and_taken_back_without_them():
    original_workers = settings.LOAD_WORKERS
    advanced.apply(KEPT)
    assert (settings.LOAD_WORKERS, settings.COMBINED_TEXT_WEIGHT) == (3, 0.7)
    advanced.apply({})
    assert (settings.LOAD_WORKERS, settings.COMBINED_TEXT_WEIGHT) == (original_workers,
                                                                      advanced.DEFAULTS["COMBINED_TEXT_WEIGHT"])


def test_advanced_settings_are_saved_with_the_preferences():
    replace(Preferences(), advanced=KEPT).save()
    assert Preferences.load().advanced == KEPT
