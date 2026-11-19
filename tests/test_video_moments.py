"""Every moment of a video where the search matches."""
from dataclasses import replace

import numpy as np
import pytest

from support import COLOURS, EVERYTHING, FakeModel, colour_vec, open_library
from vibekimus.media_info import unknown_info
from vibekimus.search import SearchEngine, SearchOptions
from vibekimus.ui.formatting import video_badge

FRAMES = [(("a.mp4", 5.0), "red"), (("a.mp4", 10.0), "blue"), (("a.mp4", 15.0), "red"),
          (("b.mp4", 5.0), "green"), (("a.mp4", 20.0), "purple")]
CLOSE = 0.9


@pytest.fixture
def engine(tmp_path):
    library = open_library(tmp_path)
    library.videos.add([key for key, _colour in FRAMES], np.stack([colour_vec(COLOURS[c]) for _key, c in FRAMES]),
                       unknown_info(len(FRAMES)))
    return SearchEngine(library)


def search(engine, min_score=CLOSE, **options):
    return engine.by_text(FakeModel(), "red", SearchOptions(min_score=min_score, include_images=False, **options))


def test_a_video_result_knows_every_moment_of_its_video_and_which_match(engine):
    found = search(engine)
    timeline = found[0].timeline
    assert len(found) == 2 and timeline is found[1].timeline
    assert [m.seconds for m in timeline.moments] == [5.0, 10.0, 15.0, 20.0]
    assert [m.seconds for m in timeline.matching] == [5.0, 15.0] and timeline.length == 25.0


def test_one_result_per_video_says_how_many_moments_match(engine):
    best_only = search(engine, best_frame_per_video=True)
    assert len(best_only) == 1 and len(best_only[0].timeline.matching) == 2
    assert video_badge(best_only[0]).endswith("2 moments")
    assert video_badge(replace(best_only[0], timeline=None)) == "▶ 0:05"


def test_a_moment_far_below_the_videos_best_does_not_match_even_among_the_results(engine):
    every_frame = search(engine, min_score=EVERYTHING)
    whole = next(r for r in every_frame if r.path.endswith("a.mp4")).timeline
    assert len(every_frame) == 5 and [m.seconds for m in whole.matching] == [5.0, 15.0]


def test_stepping_between_matching_moments_and_showing_the_result_at_another(engine):
    found = search(engine)
    timeline = found[0].timeline
    at_15 = timeline.next_matching(5.0, 1)
    assert at_15.seconds == 15.0 and timeline.next_matching(15.0, 1) is None
    assert timeline.next_matching(15.0, -1).seconds == 5.0
    assert found[0].at(at_15).timestamp == 15.0 and found[0].at(at_15).path == found[0].path
