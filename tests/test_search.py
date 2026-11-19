"""Searching: text, unwanted words, top-K, the media type filter, find similar, a picture
and text together, refining with ratings, and searching while indexing adds files."""
import os
from dataclasses import replace

import numpy as np
import pytest

from support import COLOURS, EVERYTHING, FakeModel, colour_vec, fails, in_order, names, open_library, raised
from vibekimus import media_types, settings
from vibekimus.duplicates import DuplicateFinder
from vibekimus.explore import TopicFinder, TopicNamer
from vibekimus.filters import ResultFilters
from vibekimus.library_stats import StatisticsCollector
from vibekimus.media_info import unknown_info
from vibekimus.media_kind import MediaKind
from vibekimus.media_types import MediaType
from vibekimus.query_scoring import standardized
from vibekimus.relevance import Feedback, Rating, Ratings
from vibekimus.search import NotInIndex, SearchEngine, SearchOptions, SearchResult
from vibekimus.similarity import top_indices

IMAGES_ONLY = SearchOptions(min_score=EVERYTHING, include_videos=False)
COLOUR_TYPES = (MediaType("reds", "Reds", ("red",)), MediaType("greens", "Greens", ("green",)),
                MediaType("blues", "Blues", ("blue",)))


def place(results, name):
    return in_order(results).index(name) + 1


def test_text_search_returns_real_files_best_first_images_and_videos_together(media_library):
    hits = SearchEngine(media_library).by_text(FakeModel(), "purple", SearchOptions())
    assert os.path.isfile(hits[0].path)
    assert set(in_order(hits[:2])) == {"purple.jpg", "clip_purple.mp4"}


def test_an_unwanted_word_pushes_its_matches_down(media_library):
    engine = SearchEngine(media_library)

    def place_of_yellow(query):
        return place(engine.by_text(FakeModel(), query, IMAGES_ONLY), "yellow.jpg")
    assert place_of_yellow("white -yellow") > place_of_yellow("white")


def test_top_k_equals_a_brute_force_top_k():
    scores = np.random.default_rng(0).standard_normal(200_000).astype(np.float32)
    assert set(top_indices(scores, 0.5, 300).tolist()) == set(np.argsort(-scores)[:300].tolist())


def test_the_threshold_still_applies_when_fewer_than_k_pass():
    scores = np.random.default_rng(0).standard_normal(200_000).astype(np.float32)
    assert set(top_indices(scores, 4.0, 300).tolist()) == set(np.flatnonzero(scores >= 4.0).tolist())


def test_the_type_filter_keeps_only_pictures_of_that_type(details, monkeypatch):
    monkeypatch.setattr(media_types, "MEDIA_TYPES", COLOUR_TYPES)
    greens = SearchEngine(details).by_text(FakeModel(), "red", SearchOptions(min_score=EVERYTHING,
                                                                             media_type="greens"))
    assert names(greens) == ["green.png"]


def test_find_similar_puts_the_same_picture_and_its_look_alikes_first(details):
    engine = SearchEngine(details)
    red_small = next(r for r in engine.by_text(FakeModel(), "red", IMAGES_ONLY) if r.path.endswith("red_small.png"))
    similar = engine.by_result(FakeModel(), red_small, IMAGES_ONLY)
    assert set(in_order(similar[:2])) == {"red_small.png", "red_large.png"} and similar[0].score > 0.99


def test_find_similar_on_a_file_no_longer_indexed_says_so(details):
    gone = SearchResult(1.0, "x", MediaKind.IMAGE, key="no/such/file.png")
    assert isinstance(raised(lambda: SearchEngine(details).by_result(FakeModel(), gone, IMAGES_ONLY)), NotInIndex)


def test_scores_are_standardised():
    z = standardized(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert abs(float(z.mean())) < 1e-6 and abs(float(z.std()) - 1) < 1e-6
    assert not standardized(np.ones(3, dtype=np.float32)).any(), "equal scores give zeros"


@pytest.fixture
def red_x(pairs):
    return next(r for r in SearchEngine(pairs).by_text(FakeModel(), "red", SearchOptions(min_score=EVERYTHING))
                if r.path.endswith("red_x.png"))


STRICT = SearchOptions(min_score=0.99)


def test_picture_and_text_the_weight_moves_the_ranking_from_the_picture_to_the_text(pairs, red_x, monkeypatch):
    engine = SearchEngine(pairs)
    monkeypatch.setattr(settings, "COMBINED_TEXT_WEIGHT", 0.0)
    picture_only = engine.by_result(FakeModel(), red_x, STRICT, text="blue")
    monkeypatch.setattr(settings, "COMBINED_TEXT_WEIGHT", 1.0)
    text_only = engine.by_result(FakeModel(), red_x, STRICT, text="blue")
    assert in_order(picture_only)[0].startswith("red") and in_order(text_only)[0].startswith("blue")


def test_picture_and_text_min_score_does_not_cut_mixed_scores_but_filters_do(pairs, red_x, monkeypatch):
    engine = SearchEngine(pairs)
    monkeypatch.setattr(settings, "COMBINED_TEXT_WEIGHT", 0.5)
    mixed = engine.by_result(FakeModel(), red_x, STRICT, text="blue")
    filtered = engine.by_result(FakeModel(), red_x, replace(STRICT, filters=ResultFilters(min_pixels=600 * 400)),
                                text="blue")
    alone = engine.by_result(FakeModel(), red_x, STRICT)
    assert len(mixed) == 4 and len(alone) == 2 and names(filtered) == ["blue_a.png", "red_x.png"]


@pytest.fixture
def plain_red(media_library):
    return SearchEngine(media_library).by_text(FakeModel(), "red", IMAGES_ONLY)


def by_name(results):
    return {os.path.basename(r.path): r for r in results}


def test_ratings_are_given_taken_back_counted_and_handed_to_the_search(plain_red):
    found = by_name(plain_red)
    ratings = Ratings()
    ratings.rate(found["purple.jpg"], Rating.LIKE)
    ratings.rate(found["red.jpg"], Rating.DISLIKE)
    ratings.rate(found["white.jpg"], Rating.LIKE)
    ratings.rate(found["white.jpg"], Rating.UNRATED)
    assert ratings.counts() == (1, 1)
    assert ratings.of(found["purple.jpg"]) == Rating.LIKE and ratings.of(found["white.jpg"]) == Rating.UNRATED
    assert ratings.feedback() == Feedback((("image", "sub dir/purple.jpg"),), (("image", "sub dir/red.jpg"),))


def test_refining_moves_the_liked_result_up_and_the_disliked_one_down(media_library, plain_red):
    found = by_name(plain_red)
    ratings = Ratings()
    ratings.rate(found["purple.jpg"], Rating.LIKE)
    ratings.rate(found["red.jpg"], Rating.DISLIKE)
    refined = SearchEngine(media_library).by_text(FakeModel(), "red", replace(IMAGES_ONLY, feedback=ratings.feedback()))
    assert place(refined, "purple.jpg") < place(plain_red, "purple.jpg")
    assert place(plain_red, "red.jpg") == 1 and place(refined, "red.jpg") > 1
    assert len(refined) == len(plain_red)


def test_a_like_alone_works_and_ratings_of_files_gone_change_nothing(media_library, plain_red):
    engine = SearchEngine(media_library)
    only_liked = Ratings()
    only_liked.rate(by_name(plain_red)["yellow.jpg"], Rating.LIKE)
    toward_yellow = engine.by_text(FakeModel(), "red", replace(IMAGES_ONLY, feedback=only_liked.feedback()))
    gone = Feedback((("image", "no/such/file.jpg"),), ())
    unchanged = engine.by_text(FakeModel(), "red", replace(IMAGES_ONLY, feedback=gone))
    assert place(toward_yellow, "yellow.jpg") < place(plain_red, "yellow.jpg")
    assert [r.path for r in unchanged] == [r.path for r in plain_red]


def test_a_liked_video_frame_refines_a_search_too(media_library):
    engine = SearchEngine(media_library)
    purple_frame = next(r for r in engine.by_text(FakeModel(), "purple", SearchOptions(min_score=EVERYTHING))
                        if r.kind == MediaKind.VIDEO)
    liked = Ratings()
    liked.rate(purple_frame, Rating.LIKE)
    with_frame = engine.by_text(FakeModel(), "green", replace(IMAGES_ONLY, feedback=liked.feedback()))
    plain_green = engine.by_text(FakeModel(), "green", IMAGES_ONLY)
    assert place(with_frame, "purple.jpg") < place(plain_green, "purple.jpg")


def test_searching_duplicates_explore_and_statistics_while_indexing_adds_files(tmp_path):
    race = open_library(tmp_path / "race")
    race.images.add([f"a/{i}.png" for i in range(6)],
                    np.stack([colour_vec(COLOURS[c]) for c in ("red", "red", "green", "blue", "red", "green")]),
                    unknown_info(6))
    race.images.flush()
    race.texts.set("a/1.png", "free beer")
    flush = race.images.flush

    def indexing_adds_meanwhile():   # what an indexing run does between a reader's steps: keys first
        flush()
        race.images.add([f"a/new{len(race.images.keys)}.png"], colour_vec(COLOURS["red"])[None], unknown_info(1))
    race.images.flush = indexing_adds_meanwhile
    engine = SearchEngine(race)
    in_folder_a = ResultFilters(folder=str(tmp_path / "race" / "a"))
    outcomes = [fails(lambda: engine.by_text(FakeModel(), '"free beer" red', SearchOptions(min_score=EVERYTHING))),
                fails(lambda: engine.by_text(FakeModel(), "red", SearchOptions(min_score=EVERYTHING,
                                                                               filters=in_folder_a))),
                fails(lambda: DuplicateFinder(race).find(0.999)),
                fails(lambda: TopicFinder(race, TopicNamer()).find(FakeModel())),
                fails(lambda: StatisticsCollector(race).collect(
                    lambda view: engine.media_types_of(FakeModel(), view), count_duplicates=True))]
    assert outcomes == [None] * 5
