"""Explore: the pictures grouped into topics, named after the closest words."""
import numpy as np
import pytest

from support import COLOURS, EVERYTHING, FakeModel, colour_vec, open_library
from vibekimus import explore, settings
from vibekimus.explore import TopicFinder, TopicNamer, spherical_kmeans
from vibekimus.media_info import unknown_info
from vibekimus.relevance import Rating, Ratings
from vibekimus.search import SearchEngine, SearchOptions

GROUPS = {"red": 30, "green": 20, "blue": 12}
NOISE = 0.05


@pytest.fixture
def explored(tmp_path, monkeypatch):
    """A library of three colour groups (with some noise), three topics of at least four pictures, four words."""
    library = open_library(tmp_path)
    rng = np.random.default_rng(7)
    keys, vectors = [], []
    for colour, count in GROUPS.items():
        for number in range(count):
            vector = colour_vec(COLOURS[colour]) + rng.normal(0, NOISE, 6).astype(np.float32)
            keys.append(f"{colour}/{number:02d}.png")
            vectors.append(vector / np.linalg.norm(vector))
    library.images.add(keys, np.stack(vectors), unknown_info(len(keys)))
    monkeypatch.setattr(settings, "EXPLORE_TOPICS", 3)
    monkeypatch.setattr(settings, "EXPLORE_MIN_TOPIC_SIZE", 4)
    monkeypatch.setattr(explore, "TOPIC_WORDS", ("red", "green", "blue", "purple"))
    return library, keys, vectors


def folders_of(topic):
    return {key.split("/")[0] for key in topic.keys}


def test_pictures_are_grouped_into_topics_largest_first_each_named_after_its_closest_words(explored):
    library = explored[0]
    finder = TopicFinder(library, TopicNamer())
    topics = finder.find(FakeModel())
    assert [t.size for t in topics] == [30, 20, 12]
    assert [folders_of(t) for t in topics] == [{"red"}, {"green"}, {"blue"}]
    assert [t.name.split(" · ")[0] for t in topics] == ["red", "green", "blue"]
    assert [t.keys for t in finder.find(FakeModel())] == [t.keys for t in topics], "the same topics every time"


def test_a_topics_pictures_are_shown_closest_to_its_centre_first_and_ratings_refine_it(explored):
    library = explored[0]
    red = TopicFinder(library, TopicNamer()).find(FakeModel())[0]
    engine = SearchEngine(library)
    shown = engine.by_topic(FakeModel(), red, SearchOptions(min_score=EVERYTHING))
    assert len(shown) == 30 and {r.key for r in shown} == set(red.keys), "nothing else"
    assert [r.score for r in shown] == sorted((r.score for r in shown), reverse=True)
    ratings = Ratings()
    ratings.rate(shown[-1], Rating.LIKE)
    ratings.rate(shown[10], Rating.DISLIKE)
    refined = [r.key for r in engine.by_topic(FakeModel(), red, SearchOptions(min_score=EVERYTHING,
                                                                               feedback=ratings.feedback()))]
    assert len(refined) == 30, "only its own pictures"
    assert refined.index(shown[-1].key) < len(shown) - 1 and refined.index(shown[10].key) > 10


def test_only_pictures_passing_the_filters_without_excluded_ones(explored):
    library, keys, _vectors = explored
    allowed = np.array([not key.startswith("blue/") for key in keys])
    library.exclusions.add("green/0")
    topics = TopicFinder(library, TopicNamer()).find(FakeModel(), allowed=allowed)
    assert sum(t.size for t in topics) == 30 + 10 and not any(folders_of(t) & {"blue"} for t in topics)


def test_inside_a_topic_is_grouped_again_into_topics_of_its_own_pictures(explored, monkeypatch):
    library = explored[0]
    finder = TopicFinder(library, TopicNamer())
    red = finder.find(FakeModel())[0]
    monkeypatch.setattr(settings, "EXPLORE_TOPICS", 2)
    inside = finder.find(FakeModel(), within=red)
    assert len(inside) == 2 and sum(t.size for t in inside) == 30 and all(folders_of(t) == {"red"} for t in inside)


def test_grouping_never_fails_on_identical_pictures(explored):
    vectors = explored[2]
    centres, labels = spherical_kmeans(np.stack([vectors[0]] * 5), 3)
    assert centres.shape == (3, 6) and np.isfinite(centres).all() and len(labels) == 5


def test_explore_groups_in_ram_when_the_graphics_card_has_no_room(explored, monkeypatch):
    library = explored[0]
    real_kmeans, devices = explore.spherical_kmeans, []

    def out_of_memory_on_the_card(vectors, count, device=None, iterations=None):
        devices.append(device)
        if device is not None:
            raise RuntimeError("CUDA out of memory")
        return real_kmeans(vectors, count, None, iterations)
    monkeypatch.setattr(explore, "spherical_kmeans", out_of_memory_on_the_card)
    assert TopicFinder(library, TopicNamer()).find(FakeModel(), device="cuda")
    assert devices == ["cuda", None]
