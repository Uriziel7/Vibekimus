"""Indexing pictures end to end: what lands in the index, reloading it, the decode modes,
and Stop while files take long to decode."""
import os
import time

import numpy as np
import pytest

from support import (COLOURS, HANG_SECONDS, MAX_STOP_SECONDS, FakeGpuModel, colour_vec, open_library, rank, rebuild,
                     run, seconds_to_stop)
from vibekimus import settings
from vibekimus.decoding import DecodeMode, DecodingOptions
from vibekimus.index_files import peek_index_count
from vibekimus.indexing import image_loading
from vibekimus.indexing.controller import IMAGES

READABLE_PICTURES = 7   # six colours and a PNG; broken.jpg is not a picture


def test_the_image_index_holds_every_readable_picture(media_library):
    images = media_library.images
    assert len(images.keys) == READABLE_PICTURES and images.embeddings.shape == (READABLE_PICTURES, 6)


def test_pictures_reach_the_encoder_already_reduced(media_library, media_encoder):
    assert max(max(size) for size in media_encoder.seen_sizes) < 3000, max(media_encoder.seen_sizes)


def test_the_index_is_a_npz_and_no_pickle_is_written(media_library, media_folder):
    assert os.path.exists(media_library.images.path) and media_library.images.path.endswith(".npz")
    assert not any(name.endswith(".pkl") for name in os.listdir(media_folder))


def test_a_corrupt_file_is_logged_as_skipped(media_library, media_folder):
    log = os.path.join(media_folder, settings.SKIPPED_IMAGES_LOG)
    assert os.path.exists(log) and "broken.jpg" in open(log, encoding="utf-8").read()


def test_a_purple_query_ranks_the_purple_picture_first(media_library):
    images = media_library.images
    assert images.keys[rank(images, colour_vec(COLOURS["purple"]))[0]] == "sub dir/purple.jpg"


def test_the_reloaded_image_index_matches(media_library, media_folder):
    reloaded = open_library(media_folder)
    reloaded.load_images()
    images = media_library.images
    assert reloaded.images.keys == images.keys
    assert np.allclose(reloaded.images.embeddings, images.embeddings, atol=1e-3)   # float16 on disk
    assert reloaded.images.embeddings.dtype == np.float32
    assert peek_index_count(images.path) == READABLE_PICTURES


@pytest.mark.parametrize("mode", [DecodeMode.GPU, DecodeMode.HYBRID])
def test_gpu_decode_modes_give_the_cpu_result_and_hand_refused_jpegs_back(mixed, mode):
    want = dict(zip(mixed.images.keys, mixed.images.embeddings))
    library = open_library(mixed.scope.data_dir)
    model = FakeGpuModel()
    run(library, rebuild(IMAGES), encoder=model, decoding=DecodingOptions(images=mode))
    rows = dict(zip(library.images.keys, library.images.embeddings))
    assert rows.keys() == want.keys() and all(np.allclose(rows[key], want[key], atol=1e-3) for key in rows)
    assert model.jpeg_decoder.decoded > 0 and model.jpeg_decoder.refused > 0


def test_gpu_decode_mode_without_a_gpu_decoder_falls_back_to_the_cpu(mixed):
    library = open_library(mixed.scope.data_dir)
    run(library, rebuild(IMAGES), decoding=DecodingOptions(images=DecodeMode.GPU))
    assert len(library.images.keys) == len(mixed.images.keys)


def test_stop_ends_image_indexing_without_waiting_for_files_being_decoded(mixed, monkeypatch):
    monkeypatch.setattr(image_loading, "open_image", lambda path, draft_size=None: time.sleep(HANG_SECONDS))
    assert seconds_to_stop(open_library(mixed.scope.data_dir), IMAGES) < MAX_STOP_SECONDS
