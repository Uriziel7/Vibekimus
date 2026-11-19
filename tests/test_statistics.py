"""Statistics of the library: files and bytes, by year, format, folder, kind of picture, duplicates."""
import os
import time

from support import FakeModel
from vibekimus import library_stats, media_types
from vibekimus.library_stats import OTHER_FOLDERS, StatisticsCollector
from vibekimus.media_types import MediaType
from vibekimus.search import SearchEngine

COPIES = ("blue_b.png", "red_y.png")


def test_files_and_bytes_by_year_format_folder_and_kind_and_the_duplicates(pairs, monkeypatch):
    monkeypatch.setattr(media_types, "MEDIA_TYPES", (MediaType("reds", "Reds", ("red",)),
                                                     MediaType("blues", "Blues", ("blue",))))
    stats = StatisticsCollector(pairs).collect(lambda view: SearchEngine(pairs).media_types_of(FakeModel(), view),
                                               count_duplicates=True)
    folder = pairs.scope.data_dir
    pictures = [name for name in os.listdir(folder) if name.endswith(".png")]
    all_bytes = sum(os.path.getsize(os.path.join(folder, name)) for name in pictures)
    copies_bytes = sum(os.path.getsize(os.path.join(folder, name)) for name in COPIES)
    assert (stats.images.files, stats.images.bytes, stats.videos.files) == (4, all_bytes, 0)
    assert {t.label: t.files for t in stats.by_year} == {"2015": 1, time.strftime("%Y"): 3}
    assert [(t.label, t.files) for t in stats.by_format] == [("PNG", 4)]
    assert [(t.label, t.files) for t in stats.by_folder] == [(folder, 4)]
    assert {(t.label, t.files) for t in stats.by_type} == {("Reds", 2), ("Blues", 2)}
    assert stats.index_bytes == os.path.getsize(pairs.images.path)
    duplicates = stats.duplicates
    assert duplicates is not None and (duplicates.groups, duplicates.copies, duplicates.bytes) == (2, 2, copies_bytes)


def test_subfolders_excluded_files_and_what_needs_the_model(details):
    trip = os.path.join(details.scope.data_dir, "trip")
    details.exclusions.add_folder("trip")
    try:
        without_trip = StatisticsCollector(details).collect()
    finally:
        details.exclusions.remove_folder("trip")
    everything = StatisticsCollector(details).collect()
    assert {t.label: t.files for t in everything.by_folder}.get(trip) == 1, "a subfolder's files counted under it"
    assert without_trip.images.files == everything.images.files - 1
    assert trip not in {t.label for t in without_trip.by_folder}, "excluded files left out"
    assert everything.by_type == () and everything.duplicates is None, "nothing counted that needs the model"


def test_beyond_the_folders_shown_the_rest_are_added_up(details, monkeypatch):
    monkeypatch.setattr(library_stats, "FOLDERS_SHOWN", 1)
    folders = StatisticsCollector(details).collect().by_folder
    assert len(folders) == 2 and folders[1].label == OTHER_FOLDERS
    assert sum(t.files for t in folders) == len(details.images.keys)
