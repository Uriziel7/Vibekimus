"""Indexing videos: what lands in the index, the length limit, the time budget per video,
the frame taken for each moment, and Stop while videos take long to read."""
import os
import time

import cv2
import numpy as np
import pytest

from support import (COLOURS, HANG_SECONDS, MAX_STOP_SECONDS, VIDEO_FPS, colour_vec, open_library, rank, rebuild,
                     refresh, run, seconds_to_stop, video_files, write_clip)
from vibekimus import settings, video_io
from vibekimus.decoding import DecodeMode, DecodingOptions
from vibekimus.indexing.base import PassState
from vibekimus.indexing.controller import VIDEOS
from vibekimus.video_io import NO_TIME_STAMP, VideoFrameSampler, VideoSample

VIDEOS_IN_LIBRARY = 3
GREY = (120, 120, 120)


def test_stderr_is_given_back_after_video_indexing(tmp_path):
    write_clip(str(tmp_path / "clip.mp4"), COLOURS["green"])
    before = os.fstat(2)
    run(open_library(tmp_path), rebuild(VIDEOS))
    after = os.fstat(2)
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)


def test_the_video_index_holds_every_video(media_library):
    videos = media_library.videos
    assert len(videos.files()) == VIDEOS_IN_LIBRARY and len(videos.keys) == videos.embeddings.shape[0]


def test_a_purple_query_ranks_the_purple_clip_first(media_library):
    videos = media_library.videos
    best = rank(videos, colour_vec(COLOURS["purple"]))
    assert best and videos.keys[best[0]][0] == "clip_purple.mp4"


def test_the_reloaded_video_index_matches_and_a_refresh_changes_nothing(media_library, media_folder):
    reloaded = open_library(media_folder)
    reloaded.load_videos()
    videos = media_library.videos
    assert reloaded.videos.keys == videos.keys
    assert np.allclose(reloaded.videos.embeddings, videos.embeddings, atol=1e-3)
    before = list(reloaded.videos.keys)
    assert run(reloaded, refresh(VIDEOS)) == [(VIDEOS, PassState.UP_TO_DATE, True)]
    assert reloaded.videos.keys == before


def test_a_video_over_the_length_limit_is_skipped_not_reported_and_indexed_once_the_limit_goes(tmp_path):
    folder = str(tmp_path)
    write_clip(os.path.join(folder, "short.mp4"), GREY, frames=10, size=(320, 240))   # 1 s
    write_clip(os.path.join(folder, "long.mp4"), GREY, frames=40, size=(320, 240))    # 4 s
    library = open_library(folder)
    run(library, rebuild(VIDEOS), max_video_seconds=2)
    assert video_files(library.videos) == ["short.mp4"]
    assert not os.path.exists(os.path.join(folder, settings.SKIPPED_VIDEOS_LOG))
    run(library, refresh(VIDEOS), max_video_seconds=0)
    assert video_files(library.videos) == ["long.mp4", "short.mp4"]


def test_over_the_time_budget_a_video_keeps_one_frame(tmp_path, monkeypatch):
    write_clip(str(tmp_path / "twelve_seconds.mp4"), (90, 90, 200), frames=120, size=(320, 240))
    normal = open_library(tmp_path)
    run(normal, rebuild(VIDEOS), decoding=DecodingOptions(videos=DecodeMode.GPU))
    monkeypatch.setattr(settings, "VIDEO_TIME_BUDGET_SECONDS", 0)
    slow = open_library(tmp_path)
    run(slow, rebuild(VIDEOS))
    assert len(slow.videos.keys) == 1 and len(normal.videos.keys) == 2


def test_each_sampled_frame_comes_from_its_moment(tmp_path):
    path = str(tmp_path / "seconds.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (320, 240))
    for second in range(30):
        for _ in range(int(VIDEO_FPS)):
            writer.write(np.full((240, 320, 3), 20 + second * 7, np.uint8))
    writer.release()
    sample = VideoFrameSampler(lambda: False).sample(path)
    off_by = [abs((np.asarray(image).mean() - 20) / 7 - moment) for image, moment in sample.frames]
    assert sample.frames and max(off_by) < 1.0, off_by
    assert (sample.width, sample.height) == (320, 240)


FIRST_GREY = 20
KEY_FRAME_INTERVAL = 150


class StandInCapture:
    """A video as OpenCV shows it, frame n grey FIRST_GREY + n. With time stamps a seek decodes from the key frame
    before the moment; without them (many .wmv) OpenCV decodes from the very first frame."""

    def __init__(self, frames, time_stamps):
        self._frames, self._time_stamps = frames, time_stamps
        self.position = self.decoded = 0

    def get(self, prop):
        if prop == cv2.CAP_PROP_PTS:
            return float(self.position) if self._time_stamps else NO_TIME_STAMP
        return {cv2.CAP_PROP_FPS: VIDEO_FPS, cv2.CAP_PROP_POS_FRAMES: float(self.position)}[prop]

    def set(self, prop, milliseconds):
        assert prop == cv2.CAP_PROP_POS_MSEC
        target = int(milliseconds / 1000 * VIDEO_FPS + 0.5)
        self.decoded += target % KEY_FRAME_INTERVAL if self._time_stamps else target
        self.position = target

    def grab(self):
        if self.position >= self._frames:
            return False
        self.position += 1
        self.decoded += 1
        return True

    def read(self):
        frame = np.full((24, 32, 3), FIRST_GREY + self.position, np.uint8)
        return self.grab(), frame

    def release(self):
        pass


@pytest.mark.parametrize("time_stamps", [True, False], ids=["seeking", "no time stamps"])
def test_each_moment_gives_its_frame_and_without_time_stamps_the_video_is_read_forward_once(time_stamps):
    moments = [2.0, 4.0, 6.0]
    capture = StandInCapture(frames=200, time_stamps=time_stamps)
    frames = VideoFrameSampler(lambda: False)._read_frames(capture, moments, lambda moment: False, "clip")
    targets = [int(moment * VIDEO_FPS) for moment in moments]
    assert [int(np.asarray(image).mean()) - FIRST_GREY for image, _ in frames] == targets
    assert capture.decoded == (sum(target % KEY_FRAME_INTERVAL + 1 for target in targets) if time_stamps
                               else targets[-1] + 1), "each seek from the first frame was decoding it all again"


@pytest.mark.parametrize("time_stamps", [True, False], ids=["seeking", "no time stamps"])
def test_a_thumbnail_frame_of_a_file_without_time_stamps_is_read_with_pyav(time_stamps, monkeypatch):
    by_time = []
    monkeypatch.setattr(video_io, "open_video", lambda cv2, path, hardware: (StandInCapture(200, time_stamps), True))
    monkeypatch.setattr(video_io, "_frame_by_time",
                        lambda path, moment: by_time.append(moment) or np.full((24, 32, 3), 1, np.uint8))
    grey = np.asarray(video_io.read_frame_at("clip.wmv", 4.0)).mean()
    assert (by_time, grey) == (([], FIRST_GREY + 4.0 * VIDEO_FPS) if time_stamps else ([4.0], 1))


def test_pyav_gives_the_frame_opencv_gives_for_the_same_moment(tmp_path):
    pytest.importorskip("av")
    path = str(tmp_path / "frames.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (64, 48))
    for number in range(40):
        writer.write(np.full((48, 64, 3), FIRST_GREY + 5 * number, np.uint8))
    writer.release()
    for moment in (0.0, 0.7, 1.5, 3.2):
        capture = cv2.VideoCapture(path)
        capture.set(cv2.CAP_PROP_POS_MSEC, moment * 1000)
        opencv = capture.read()[1].mean()
        capture.release()
        assert abs(video_io._frame_by_time(path, moment).mean() - opencv) < 1, moment


def test_stop_ends_video_indexing_without_waiting_for_videos_being_read(tmp_path, monkeypatch):
    write_clip(str(tmp_path / "clip.mp4"), COLOURS["green"])
    monkeypatch.setattr(VideoFrameSampler, "sample",
                        lambda self, path, is_indexed=None: (time.sleep(HANG_SECONDS), VideoSample())[1])
    assert seconds_to_stop(open_library(tmp_path), VIDEOS) < MAX_STOP_SECONDS
