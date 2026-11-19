"""Index files that were tampered with: pickles, objects inside a .npz, paths outside the
indexed folders, files that are not media."""
import os
import pickle

import numpy as np
import pytest

from support import fails, open_library
from vibekimus.encoders.catalog import SIGLIP2_SO400M
from vibekimus.index_files import ImageIndexFormat, VideoIndexFormat, save_index_file
from vibekimus.media_info import unknown_info
from vibekimus.scope import AllDrivesScope, MultiFolderScope, is_safe_rel_path

OUTSIDE_PATHS = ("../../outside.jpg", "/etc/passwd", "C:/Windows/win.ini", "a/../../b.jpg", "\\\\server\\share\\x.jpg")


class Payload:
    """Unpickling it would create the marker file."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return open, (self.marker, "w")


@pytest.fixture
def evil(tmp_path):
    """An empty library, and the file a payload would create."""
    return open_library(tmp_path / "evil"), str(tmp_path / "PWNED")


def test_a_raw_pickle_named_like_an_index_is_refused_and_not_run(evil):
    library, marker = evil
    os.makedirs(library.scope.data_dir, exist_ok=True)
    with open(library.images.path, "wb") as f:
        pickle.dump(Payload(marker), f)
    assert fails(library.load_images) and not library.images.keys
    assert not os.path.exists(marker)


def test_a_npz_with_a_pickled_object_inside_is_refused_and_not_run(evil, tmp_path):
    library, marker = evil
    smuggled = str(tmp_path / "obj.npz")
    with open(smuggled, "wb") as f:
        np.savez(f, emb=np.zeros((1, 6), np.float16), meta=np.array([Payload(marker)], dtype=object),
                 count=np.array(1))
    assert fails(lambda: library.images.load(smuggled, library.scope.validate)) and not library.images.keys
    assert not os.path.exists(marker)


@pytest.mark.parametrize("bad", OUTSIDE_PATHS)
def test_an_index_pointing_outside_the_folder_is_refused(evil, tmp_path, bad):
    library, _marker = evil
    traversal = str(tmp_path / "trav.npz")
    save_index_file(traversal, np.zeros((1, 6), np.float32), {"paths": [bad]})
    assert fails(lambda: library.images.load(traversal, library.scope.validate)) and not library.images.keys


def test_ordinary_relative_paths_are_accepted():
    assert all(is_safe_rel_path(path) for path in ("a.jpg", "sub dir/b.jpg", "x/y/z..jpg", "..hidden/c.png"))


def test_an_index_listing_a_file_that_is_not_media_is_refused(tmp_path):
    image_format, video_format = ImageIndexFormat(SIGLIP2_SO400M), VideoIndexFormat(SIGLIP2_SO400M)
    image_format.write(str(tmp_path / image_format.file_name()), ["cat.jpg", "run_me.bat"], np.ones((2, 6), np.float32),
                       unknown_info(2))
    assert "not of that kind" in (fails(open_library(tmp_path).load_images) or "")
    assert fails(lambda: video_format._refuse_other_files([("clip.MP4", 1.0)])) is None
    assert fails(lambda: video_format._refuse_other_files([("clip.jpg", 1.0)])) is not None


def test_an_index_of_several_folders_refuses_files_outside_them(tmp_path):
    inside = str(tmp_path / "photos")
    several = MultiFolderScope("x", [inside], str(tmp_path / "multi_data"))
    outside = str(tmp_path.parent / "elsewhere" / "a.jpg").replace("\\", "/")
    assert fails(lambda: several.validate([os.path.join(inside, "a", "b.jpg").replace("\\", "/")])) is None
    assert fails(lambda: several.validate([outside])) is not None
    assert fails(lambda: several.validate([inside.replace("\\", "/") + "2/a.jpg"])) is not None   # a look-alike name
    assert fails(lambda: several.validate(path for path in [outside])) is not None
    assert fails(lambda: AllDrivesScope([inside]).validate([outside])) is None   # any drive
