"""What a library covers and where its index lives: the all-drives scan, opening saved
indexes, indexes of each model, indexes of several folders, the list of saved indexes,
an old folder index moved into the program's data, and the index opened last."""
import json
import os
import shutil

import numpy as np
import pytest
from PIL import Image

from support import (COLOURS, as_key, colour_vec, fails, open_library, raised, rank, rebuild, refresh, run,
                     save_picture, video_files, write_clip)
from vibekimus import saved_indexes as saved_indexes_module
from vibekimus import settings
from vibekimus.encoders.catalog import SIGLIP2_SO400M, ModelSpec
from vibekimus.encryption import write_file_atomically
from vibekimus.index_files import save_index_file
from vibekimus.index_migration import move_folder_index
from vibekimus.indexing.base import PassState
from vibekimus.indexing.controller import IMAGES, VIDEOS
from vibekimus.last_index import LastIndex
from vibekimus.library import IncompatibleIndex, MediaLibrary
from vibekimus.saved_indexes import IndexPlace, SavedIndexes
from vibekimus.scope import AllDrivesScope, FolderScope, MultiFolderScope, same_path, scope_of_index_folder

BIG = (1600, 1200)
# A second model: the program offers one, but every model keeps its own index files.
ANOTHER_MODEL = ModelSpec(key="another-model", label="Another model", description="", family=SIGLIP2_SO400M.family,
                          name="ViT-Another", pretrained="test")


def noisy(path, rgb, size=BIG):
    """A photo with noise, so the JPEG stays above the all-drives minimum file size."""
    noise = np.random.default_rng(1).integers(-40, 40, (size[1], size[0], 3))
    pixels = np.clip(np.array(rgb)[None, None, :] + noise, 0, 255)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8)).save(path, quality=90)


def files_under(*roots):
    return {os.path.join(folder, name) for root in roots for folder, _dirs, files in os.walk(root) for name in files}


@pytest.fixture
def drives(tmp_path):
    """Two pretend drives: photos, system folders, an icon, a video and (where allowed) a link."""
    c, d = str(tmp_path / "driveC"), str(tmp_path / "driveD")
    noisy(os.path.join(c, "Windows", "wallpaper.jpg"), COLOURS["white"])
    noisy(os.path.join(c, "Program Files", "App", "splash.jpg"), COLOURS["white"])
    noisy(os.path.join(c, "Users", "me", "AppData", "Local", "cache.jpg"), COLOURS["white"])
    noisy(os.path.join(c, "Users", "me", "Pictures", "red.jpg"), COLOURS["red"])
    noisy(os.path.join(c, "Users", "me", "Pictures", "Windows", "blue.jpg"), COLOURS["blue"])
    save_picture(os.path.join(c, "Users", "me", "Pictures", "icon.jpg"), COLOURS["green"], (32, 32))
    noisy(os.path.join(d, "photos", "purple.jpg"), COLOURS["purple"])
    write_clip(os.path.join(d, "clip_green.mp4"), COLOURS["green"])
    try:   # on Windows creating a symlink needs Developer Mode or admin rights
        os.symlink(os.path.join(c, "Users"), os.path.join(d, "link_to_users"), target_is_directory=True)
    except OSError:
        pass
    photos = sorted(as_key(path) for path in (os.path.join(c, "Users", "me", "Pictures", "red.jpg"),
                                              os.path.join(c, "Users", "me", "Pictures", "Windows", "blue.jpg"),
                                              os.path.join(d, "photos", "purple.jpg")))
    return c, d, photos


def test_the_all_drives_scan(drives):
    c, d, photos = drives
    before = files_under(c, d)
    library = open_library(AllDrivesScope([c, d]))
    outcome = run(library, refresh(IMAGES), refresh(VIDEOS))
    assert outcome == [(IMAGES, PassState.FINISHED, False), (VIDEOS, PassState.FINISHED, True)], "videos follow"
    assert sorted(library.images.keys) == photos, "photos on both drives; no system folders, icons or links"
    assert video_files(library.videos) == [as_key(d, "clip_green.mp4")]
    assert files_under(c, d) == before, "nothing is written to the scanned drives"
    index_dir = settings.ALL_DRIVES_INDEX_DIR
    assert os.path.dirname(library.images.path) == index_dir and os.path.dirname(library.videos.path) == index_dir
    hit = library.scope.absolute_path(library.images.keys[rank(library.images, colour_vec(COLOURS["purple"]))[0]])
    assert os.path.isfile(hit) and hit.endswith("purple.jpg")


def test_the_all_drives_index_reloads_and_keeps_a_drive_that_is_not_connected(drives):
    c, d, photos = drives
    run(open_library(AllDrivesScope([c, d])), refresh(IMAGES), refresh(VIDEOS))
    library = open_library(AllDrivesScope([c, d]))
    library.load_images()
    library.load_videos()
    assert sorted(library.images.keys) == photos and len(library.videos.keys) == 1
    os.rename(d, d + "_unplugged")
    os.remove(os.path.join(c, "Users", "me", "Pictures", "red.jpg"))
    try:
        run(library, refresh(IMAGES), refresh(VIDEOS))
    finally:
        os.rename(d + "_unplugged", d)
    assert sorted(library.images.keys) == [p for p in photos if not p.endswith("red.jpg")]
    assert len(library.videos.keys) == 1


def test_a_folders_saved_index_opens_as_that_folder(mixed):
    saved = open_library(mixed.scope.data_dir)
    saved.open_saved_index(mixed.images.path)
    saved.load_images()
    assert isinstance(saved.scope, FolderScope) and sorted(saved.images.keys) == sorted(mixed.images.keys)


def test_an_index_of_another_model_or_a_renamed_file_is_refused(mixed, tmp_path):
    other = tmp_path / "other_model"
    other.mkdir()
    foreign = str(other / os.path.basename(mixed.images.path))
    save_index_file(foreign, np.zeros((1, 6), np.float32), {"model": "ViT-L-14", "pretrained": "laion2b",
                                                            "paths": ["a.jpg"]})
    renamed = str(other / "my_copy.npz")
    shutil.copy(mixed.images.path, renamed)
    errors = [fails(lambda p=p: MediaLibrary(SIGLIP2_SO400M).open_saved_index(p)) for p in (foreign, renamed)]
    assert all(errors) and "ViT-L-14" in errors[0]
    assert isinstance(raised(lambda: MediaLibrary(SIGLIP2_SO400M).open_saved_index(foreign)), IncompatibleIndex)


def test_the_all_drives_index_opens_as_the_all_drives_library(mixed):
    os.makedirs(settings.ALL_DRIVES_INDEX_DIR)
    all_drives_file = os.path.join(settings.ALL_DRIVES_INDEX_DIR, os.path.basename(mixed.images.path))
    shutil.copy(mixed.images.path, all_drives_file)
    library = MediaLibrary(SIGLIP2_SO400M)
    library.open_saved_index(all_drives_file)
    assert isinstance(library.scope, AllDrivesScope)


def test_the_siglip_index_keeps_the_file_name_it_had_before_models_became_selectable(mixed):
    assert mixed.images.format.file_name() == ".clip_cache_ViT-SO400M-16-SigLIP2-384_WEBLI.npz"


def test_another_model_uses_other_index_files_and_its_library_says_which_model_to_choose(mixed):
    folder = mixed.scope.data_dir
    other = open_library(folder, ANOTHER_MODEL)
    assert os.path.dirname(other.images.path) == folder and other.images.path != mixed.images.path
    assert not other.has_saved_images()
    error = fails(lambda: other.open_saved_index(mixed.images.path))
    assert error is not None and SIGLIP2_SO400M.label in error and "Settings" in error


def test_switching_the_model_keeps_the_folder_open_and_empties_the_index(mixed):
    switching = open_library(mixed.scope.data_dir)
    switching.load_images()
    switching.use_model(ANOTHER_MODEL)
    assert switching.is_open and not switching.images.keys
    assert switching.images.path == open_library(mixed.scope.data_dir, ANOTHER_MODEL).images.path


@pytest.fixture
def trip_and_party(tmp_path):
    """Two folders of pictures, one with a subfolder."""
    first, second = str(tmp_path / "trip"), str(tmp_path / "party")
    for folder, colours in ((first, ("red", "blue")), (os.path.join(first, "day two"), ("green",)),
                            (second, ("purple",))):
        for name in colours:
            save_picture(os.path.join(folder, f"{name}.jpg"), COLOURS[name])
    keys = sorted([as_key(first, "blue.jpg"), as_key(first, "day two", "green.jpg"), as_key(first, "red.jpg"),
                   as_key(second, "purple.jpg")])
    return first, second, keys


@pytest.fixture
def holiday(trip_and_party):
    """An index of the first folder, with the second added and indexed."""
    first, second, _keys = trip_and_party
    scope = SavedIndexes.create("Holiday: 2024", [first])
    library = open_library(scope)
    run(library, rebuild(IMAGES))
    added = library.add_folder(second)
    inside = library.add_folder(os.path.join(first, "day two"))
    run(library, refresh(IMAGES))
    return library, added, inside


def test_an_index_of_several_folders(trip_and_party, holiday):
    _first, _second, keys = trip_and_party
    library, added, inside = holiday
    assert os.path.dirname(library.scope.data_dir) == settings.INDEXES_DIR, "kept in the program's data"
    assert added and not inside, "a second folder is added, one inside a folder already in it is not"
    assert sorted(library.images.keys) == keys


def test_its_name_and_folders_are_saved_with_it_and_removing_a_folder_drops_its_entries(trip_and_party, holiday):
    first, second, keys = trip_and_party
    library = holiday[0]
    reopened = open_library(library.scope_of_saved_index(library.images.path))
    reopened.load_images()
    assert isinstance(reopened.scope, MultiFolderScope) and reopened.scope.name == "Holiday: 2024"
    assert reopened.scope.scan_roots == [first, second] and sorted(reopened.images.keys) == keys
    reopened.remove_folder(second)
    again = open_library(MultiFolderScope.load(library.scope.data_dir))
    again.load_images()
    assert again.scope.scan_roots == [first]
    assert sorted(again.images.keys) == [key for key in keys if "/party/" not in key]
    assert again.scope.folder_prefix(os.path.join(first, "day two")) == as_key(first, "day two")
    assert again.scope.folder_prefix(second) is None


@pytest.fixture
def old_folder_index(media_library, media_folder, tmp_path):
    """A copy of an indexed folder (pictures, videos) with read text and exclusions."""
    folder = str(tmp_path / "old_folder_index")
    shutil.copytree(media_folder, folder)
    old = open_library(folder)
    old.load_images()
    old.load_videos()
    old.texts.set(old.images.keys[0], "hello there")
    old.texts.save()
    old.exclusions.add("tmp")
    old.exclusions.add_folder("sub dir")
    return old


def test_an_old_folder_index_moves_to_the_programs_data_without_computing_anything_again(old_folder_index):
    old = old_folder_index
    folder = old.scope.data_dir
    before = {name: open(os.path.join(folder, name), "rb").read() for name in os.listdir(folder)
              if name.startswith(".")}
    moved_scope = move_folder_index(folder, "Old library")
    moved = open_library(moved_scope)
    moved.load_images()
    moved.load_videos()
    moved.load_texts()
    old_rows = {old.scope.absolute_path(k): e for k, e in zip(old.images.keys, old.images.embeddings)}
    new_rows = {moved.scope.absolute_path(k): e for k, e in zip(moved.images.keys, moved.images.embeddings)}
    assert moved_scope.scan_roots == [folder]
    assert old_rows.keys() == new_rows.keys() and all(np.array_equal(old_rows[k], new_rows[k]) for k in old_rows)
    assert all(os.path.isabs(key) for key in moved.images.keys)
    assert sorted(moved.videos.files()) == sorted(as_key(folder, f) for f in old.videos.files())
    assert moved.texts.text_of(as_key(folder, old.images.keys[0])) == "hello there"
    assert moved.exclusions.sorted() == ["tmp"] and moved.exclusions.folders() == [as_key(folder, "sub dir")]
    changed = [name for name, data in before.items() if open(os.path.join(folder, name), "rb").read() != data]
    assert not changed, "the old files stay as they were"


def test_the_list_of_saved_indexes(holiday, old_folder_index, mixed, trip_and_party, tmp_path, monkeypatch):
    first, second, keys = trip_and_party
    move_folder_index(old_folder_index.scope.data_dir, "Old library")
    os.makedirs(settings.ALL_DRIVES_INDEX_DIR)
    shutil.copy(mixed.images.path, settings.ALL_DRIVES_INDEX_DIR)
    saved_list = SavedIndexes(SIGLIP2_SO400M)
    saved_list.remember(mixed.scope.data_dir)
    listed = {saved.name: saved for saved in saved_list.list()}
    assert {"Holiday: 2024", "Old library", "All drives", "decode_modes"} <= listed.keys()
    assert listed["Holiday: 2024"].roots == (first, second) and listed["Holiday: 2024"].kind == IndexPlace.COLLECTION
    assert listed["Holiday: 2024"].counts() == (len(keys), 0), "with its folders and what it holds"
    assert listed["decode_modes"].kind == IndexPlace.FOLDER
    assert not SavedIndexes(ANOTHER_MODEL).list(), "another model's list shows none of them"

    monkeypatch.setattr(saved_indexes_module, "detect_drive_roots", lambda: [str(tmp_path)])
    found_now = saved_list.find_on_pc()
    found = {saved.data_dir for saved in saved_list.list() if saved.kind == IndexPlace.FOLDER}
    assert found_now > 0 and {old_folder_index.scope.data_dir, mixed.scope.data_dir} <= found, "Find on this PC"
    program_data = os.path.dirname(settings.INDEXES_DIR)
    assert not any(folder.startswith(program_data) for folder in found), "not the program's own index folders"
    saved_list.forget(mixed.scope.data_dir)
    assert mixed.scope.data_dir not in {saved.data_dir for saved in saved_list.list()}
    assert os.path.exists(mixed.images.path), "a folder index taken off the list keeps its files"


def opened_again(scope):
    LastIndex.remember(scope)
    return LastIndex.scope()


def test_at_the_very_first_start_there_is_no_index_to_open_again():
    assert LastIndex.scope() is None


def test_the_index_open_last_opens_again_as_it_was(holiday, trip_and_party, media_library, media_folder):
    first, second, _keys = trip_and_party
    os.makedirs(settings.ALL_DRIVES_INDEX_DIR)
    shutil.copy(media_library.images.path, settings.ALL_DRIVES_INDEX_DIR)   # the all-drives index has files
    several = opened_again(MultiFolderScope.load(holiday[0].scope.data_dir))
    folder = opened_again(FolderScope(media_folder))
    all_drives = opened_again(AllDrivesScope.saved())
    assert isinstance(several, MultiFolderScope) and not isinstance(several, AllDrivesScope)
    assert several.name == "Holiday: 2024" and several.scan_roots == [first, second]
    assert type(folder) is FolderScope and folder.data_dir == media_folder
    assert isinstance(all_drives, AllDrivesScope)
    not_scanned = SavedIndexes.create("Not scanned yet", [second])
    assert isinstance(opened_again(not_scanned), MultiFolderScope), "also a new index closed before its first scan"


def test_an_index_removed_since_or_a_folder_without_index_files_is_not_opened(trip_and_party, tmp_path):
    gone = SavedIndexes.create("Deleted later", [trip_and_party[1]])
    LastIndex.remember(gone)
    shutil.rmtree(gone.data_dir)
    assert LastIndex.scope() is None
    plain = tmp_path / "plain folder"
    plain.mkdir()
    assert opened_again(FolderScope(str(plain))) is None and os.listdir(plain) == [], "nothing is created in it"


def test_a_damaged_last_index_file_opens_nothing():
    os.makedirs(os.path.dirname(settings.LAST_INDEX_FILE), exist_ok=True)
    with open(settings.LAST_INDEX_FILE, "wb") as f:
        f.write(b"\x00 not json")
    assert LastIndex.scope() is None


def test_only_the_folder_of_the_index_files_is_kept(media_library, media_folder):
    LastIndex.remember(FolderScope(media_folder))
    with open(settings.LAST_INDEX_FILE, "rb") as f:
        assert json.loads(f.read()) == {"data_dir": media_folder}


def test_an_all_drives_index_kept_in_another_folder_opens_as_one(tmp_path):
    drive_root = os.path.splitdrive(str(tmp_path))[0] + os.sep
    elsewhere = tmp_path / "other copy" / "index_all_drives"
    elsewhere.mkdir(parents=True)
    write_file_atomically(str(elsewhere / settings.LIBRARY_FILE),   # an older file: no kind in it
                          json.dumps({"name": AllDrivesScope.NAME, "roots": [drive_root]}).encode("utf-8"))
    old_style = scope_of_index_folder(str(elsewhere))
    rules = old_style.scan_rules(True)
    assert isinstance(old_style, AllDrivesScope) and same_path(old_style.data_dir, str(elsewhere))
    assert rules.prune_system_folders and rules.min_file_bytes > 0 and rules.skip_cloud_only_files
    named_like_it = str(tmp_path / "named all drives")
    MultiFolderScope(AllDrivesScope.NAME, [str(tmp_path / "trip")], named_like_it).save()
    assert type(scope_of_index_folder(named_like_it)) is MultiFolderScope, "a folder index named so is not one"
    new_style = str(tmp_path / "all drives, new style")
    AllDrivesScope([drive_root], new_style).save()
    assert isinstance(scope_of_index_folder(new_style), AllDrivesScope)
