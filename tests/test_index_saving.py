"""Saving the index: autosave pieces after an interruption, saves that shrink the index,
an index whose files are all gone, a save that fails, merging rows without enough memory."""
import os
import shutil

import numpy as np
import pytest

from support import COLOURS, open_library, raised, rebuild, refresh, run, save_picture, video_files, write_clip
from vibekimus import embedding_index as embedding_index_module
from vibekimus import settings
from vibekimus.embedding_index import EmbeddingIndex, IndexSaveError
from vibekimus.encoders.catalog import SIGLIP2_SO400M
from vibekimus.index_files import ImageIndexFormat, index_shards, peek_index_count
from vibekimus.indexing.base import PassState
from vibekimus.indexing.controller import IMAGES, VIDEOS
from vibekimus.media_info import unknown_info

DIMENSIONS = 8


def only_partial_saves(index):
    """ "Power cut": the final full save never happens, only the periodic partial ones."""
    real_save = index.save
    index.save = lambda partial=False: real_save(partial=True) if partial else None


def test_an_interrupted_run_keeps_its_autosave_pieces_and_the_next_full_save_merges_them(tmp_path, monkeypatch):
    folder = str(tmp_path)
    for name in ("red", "green", "blue"):
        save_picture(os.path.join(folder, f"{name}.jpg"), COLOURS[name], (800, 600))
    write_clip(os.path.join(folder, "clip_red.mp4"), COLOURS["red"])
    first = open_library(folder)
    run(first, rebuild(IMAGES), rebuild(VIDEOS))
    base_size = os.path.getsize(first.images.path)
    assert peek_index_count(first.images.path) == 3 and not first.images.shards() and not first.videos.shards()

    save_picture(os.path.join(folder, "purple.jpg"), COLOURS["purple"], (800, 600))
    save_picture(os.path.join(folder, "yellow.jpg"), COLOURS["yellow"], (800, 600))
    write_clip(os.path.join(folder, "clip_purple.mp4"), COLOURS["purple"])
    monkeypatch.setattr(settings, "AUTOSAVE_SECONDS", 0)   # auto-save after every batch
    only_partial_saves(first.images)
    only_partial_saves(first.videos)
    run(first, refresh(IMAGES), refresh(VIDEOS))
    pieces, video_pieces = first.images.shards(), first.videos.shards()
    assert len(pieces) == 1 and peek_index_count(pieces[0]) == 2, "only the 2 new pictures are auto-saved"
    assert peek_index_count(first.images.path) == 3 and os.path.getsize(first.images.path) == base_size
    assert len(video_pieces) == 1 and peek_index_count(video_pieces[0]) == 1, "only the new video's frames"

    after = open_library(folder)
    after.load_images()
    after.load_videos()
    assert sorted(after.images.keys) == ["blue.jpg", "green.jpg", "purple.jpg", "red.jpg", "yellow.jpg"]
    assert after.images.embeddings.shape[0] == 5
    assert video_files(after.videos) == ["clip_purple.mp4", "clip_red.mp4"]
    run(after, refresh(IMAGES))
    assert len(after.images.keys) == 5, "continuing re-encodes nothing"
    after.images.save()
    after.videos.save()
    assert peek_index_count(after.images.path) == 5 and not index_shards(after.images.path)
    assert peek_index_count(after.videos.path) == 2 and not index_shards(after.videos.path)


@pytest.fixture
def shrinking(tmp_path):
    """Four indexed pictures; then three of them deleted and a new one added."""
    folder = str(tmp_path)
    for name in ("red", "green", "blue", "yellow"):
        save_picture(os.path.join(folder, f"{name}.jpg"), COLOURS[name])
    run(open_library(folder), rebuild(IMAGES))
    for name in ("red", "green", "blue"):
        os.remove(os.path.join(folder, f"{name}.jpg"))
    save_picture(os.path.join(folder, "purple.jpg"), COLOURS["purple"])
    return folder


def test_a_refresh_that_deletes_more_than_it_adds_is_saved(shrinking):
    library = open_library(shrinking)
    library.load_images()
    outcome = run(library, refresh(IMAGES))
    on_disk = ImageIndexFormat(SIGLIP2_SO400M).read(library.images.path)[0]
    assert outcome[0][1] is PassState.FINISHED and sorted(on_disk) == ["purple.jpg", "yellow.jpg"]


def test_an_index_whose_every_file_is_gone_is_removed(shrinking):
    for name in ("yellow", "purple"):
        os.remove(os.path.join(shrinking, f"{name}.jpg"))
    library = open_library(shrinking)
    library.load_images()
    run(library, refresh(IMAGES))
    assert not os.path.exists(library.images.path) and not library.images.shards()


def index_of(path, count):
    index = EmbeddingIndex(ImageIndexFormat(SIGLIP2_SO400M))
    index.path = path
    index.add([f"p{i}.jpg" for i in range(count)], np.ones((count, DIMENSIONS), np.float32), unknown_info(count))
    return index


def test_an_index_that_did_not_come_from_a_larger_file_does_not_replace_it(tmp_path):
    path = str(tmp_path / "guarded.npz")
    index_of(path, 5).checkpoint()
    not_loaded = index_of(path, 2)   # e.g. the file failed to load, then 2 files were indexed
    assert isinstance(raised(not_loaded.checkpoint), IndexSaveError)
    assert peek_index_count(path) == 5


def test_a_run_whose_index_cannot_be_saved_ends_as_failed(media_folder, tmp_path, monkeypatch):
    folder = str(tmp_path / "library")
    shutil.copytree(media_folder, folder)
    library = open_library(folder)

    def disk_full(*args, **kwargs):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(library.images.format, "write", disk_full)
    outcome = run(library, rebuild(IMAGES))
    assert outcome and outcome[-1][1] is PassState.FAILED


def test_rows_not_merged_for_lack_of_memory_stay_pending_with_their_keys():
    index = EmbeddingIndex(ImageIndexFormat(SIGLIP2_SO400M))
    index.add(["a.jpg"], np.ones((1, DIMENSIONS), np.float32), unknown_info(1))
    index.flush()
    index.add(["b.jpg"], np.full((1, DIMENSIONS), 2.0, np.float32), unknown_info(1))

    def no_memory(*args, **kwargs):
        raise MemoryError()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(embedding_index_module.np, "concatenate", no_memory)
        assert isinstance(raised(index.flush), MemoryError)
    after = index.view()
    assert after.keys == ["a.jpg", "b.jpg"] and after.embeddings[1, 0] == 2.0
