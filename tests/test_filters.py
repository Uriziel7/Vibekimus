"""File details in the index, the filters and sorting of the results, excluded folders."""
import os
import time
from dataclasses import replace

import numpy as np

from support import EVERYTHING, FakeModel, fails, in_order, names, open_library, raised, refresh, run
from vibekimus import settings
from vibekimus.embedding_index import EmbeddingIndex
from vibekimus.encoders.catalog import SIGLIP2_SO400M
from vibekimus.exclusions import Exclusions, ExclusionsNotSaved
from vibekimus.filters import ResultFilters, SortOrder, parse_date
from vibekimus.index_files import ImageIndexFormat, save_index_file
from vibekimus.indexing.controller import IMAGES
from vibekimus.media_info import unknown_info
from vibekimus.scope import AllDrivesScope
from vibekimus.search import SearchEngine, SearchOptions

EVERYTHING_RED = SearchOptions(min_score=EVERYTHING, include_videos=False)


def search_with(library, **changes):
    return SearchEngine(library).by_text(FakeModel(), "red", replace(EVERYTHING_RED, **changes))


def period(start, end=None):
    """ResultFilters of the whole period(s) typed like in the filter bar."""
    return ResultFilters(date_from=parse_date(start), date_to=parse_date(end or start, end=True))


def test_indexing_stores_dimensions_file_size_and_the_exif_date_taken(details):
    beach = dict(zip(details.images.keys, details.images.info))["trip/beach.jpg"]
    assert (int(beach["width"]), int(beach["height"])) == (1200, 800) and beach["bytes"] > 0
    assert time.localtime(beach["taken"])[:3] == (2019, 7, 14)


def test_an_older_index_without_file_details_loads_and_refresh_fills_them_in(details):
    folder = details.scope.data_dir
    save_index_file(details.images.path, details.images.embeddings.copy(),
                    {"version": 1, "kind": "images", "model": SIGLIP2_SO400M.name,
                     "pretrained": SIGLIP2_SO400M.pretrained, "paths": list(details.images.keys)})
    old = open_library(folder)
    old.load_images()
    assert len(old.images.rows_missing_info()) == 4
    run(old, refresh(IMAGES))
    reloaded = open_library(folder)
    reloaded.load_images()
    assert not reloaded.images.rows_missing_info()
    assert int(reloaded.images.info[reloaded.images.keys.index("red_large.png")]["width"]) == 1600


def test_dates_are_read_as_whole_periods_and_nonsense_is_refused():
    assert parse_date("2019") == time.mktime((2019, 1, 1, 0, 0, 0, 0, 0, -1))
    assert parse_date("2019", end=True) == time.mktime((2020, 1, 1, 0, 0, 0, 0, 0, -1))
    assert parse_date("2023-12", end=True) == time.mktime((2024, 1, 1, 0, 0, 0, 0, 0, -1))
    assert parse_date("2023.06.15", end=True) == time.mktime((2023, 6, 16, 0, 0, 0, 0, 0, -1))
    assert parse_date("") is None
    assert fails(lambda: parse_date("June")) and fails(lambda: parse_date("2023-13"))


def test_the_date_filter_uses_the_date_taken_when_known_and_the_modification_date_otherwise(details):
    assert names(search_with(details, filters=period("2019"))) == ["beach.jpg"]
    assert names(search_with(details, filters=period("2020-03"))) == ["green.png"]


def test_the_resolution_filter(details):
    assert names(search_with(details, filters=ResultFilters(min_pixels=1280 * 720))) == ["beach.jpg", "red_large.png"]


def test_the_folder_filter_keeps_only_files_inside_the_folder(details):
    trip = os.path.join(details.scope.data_dir, "trip")
    assert names(search_with(details, filters=ResultFilters(folder=trip))) == ["beach.jpg"]


def test_sorting_oldest_first_and_by_resolution(details):
    assert in_order(search_with(details, sort=SortOrder.OLDEST))[:2] == ["beach.jpg", "green.png"]
    assert in_order(search_with(details, sort=SortOrder.MOST_PIXELS))[0] == "red_large.png"


def test_a_narrow_filter_finds_its_match_when_it_is_not_among_the_best_overall(details, monkeypatch):
    monkeypatch.setattr(settings, "MAX_SEARCH_RESULTS", 1)
    trip = os.path.join(details.scope.data_dir, "trip")
    assert names(search_with(details, filters=ResultFilters(folder=trip))) == ["beach.jpg"]


def test_a_picked_folder_becomes_its_index_path_prefix(tmp_path):
    folder = str(tmp_path)
    scope = open_library(folder).scope
    drive = os.path.splitdrive(folder)[0] + os.sep
    assert scope.folder_prefix(os.path.join(folder, "trip")) == "trip"
    assert scope.folder_prefix(folder) == "", "the library itself"
    assert scope.folder_prefix(os.path.dirname(folder)) is None, "outside it"
    assert AllDrivesScope([drive]).folder_prefix(os.path.join(drive, "Photos", "Old")) == drive[:2] + "/Photos/Old"


def test_an_excluded_folder_covers_its_files_and_subfolders_only():
    rules = Exclusions()
    rules.add_folder("trip")
    assert ([rules.matches(path) for path in ("trip/beach.jpg", "Trip/sub/x.jpg", "holiday/trip/x.jpg", "trip2/x.jpg")]
            == [True, True, False, False])


def test_an_excluded_folder_is_saved_read_back_and_its_files_leave_the_results(details):
    details.exclusions.add_folder("trip")
    reread = Exclusions()
    reread.load(details.scope.data_dir)
    without_trip = search_with(details)
    details.exclusions.remove_folder("trip")
    assert reread.folders() == ["trip"]
    assert "beach.jpg" not in names(without_trip) and "beach.jpg" in names(search_with(details))


def test_an_exclusions_file_that_cannot_be_read_is_not_overwritten(tmp_path):
    rules_file = tmp_path / settings.EXCLUSIONS_FILE
    rules_file.write_text("{not json", encoding="utf-8")
    rules = Exclusions()
    rules.load(str(tmp_path))
    assert isinstance(raised(lambda: rules.add("private")), ExclusionsNotSaved)
    assert rules_file.read_text(encoding="utf-8") == "{not json" and not rules.sorted()


def test_dates_up_to_1970_can_be_filtered_and_an_unknown_date_never_passes(tmp_path):
    sixties, after_seventy = parse_date("1960"), parse_date("1971")
    assert sixties < 0 < after_seventy and fails(lambda: parse_date("1970", end=True)) is None
    rows = EmbeddingIndex(ImageIndexFormat(SIGLIP2_SO400M))
    rows.add([f"p{i}.jpg" for i in range(3)], np.ones((3, 8), np.float32), unknown_info(3))
    rows.flush()
    rows.info["modified"] = [-1, 100_000_000, 2_000_000_000]   # unknown, 1973, 2033
    passed = ResultFilters(date_from=sixties, date_to=parse_date("1980", end=True)).row_mask(rows.view(), None)
    assert passed.tolist() == [False, True, False]
