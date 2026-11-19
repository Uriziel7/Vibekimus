"""Duplicates: groups with the copy to keep, the filters, sorting groups, no chaining."""
import os

import numpy as np

from support import FakeModel, names
from vibekimus import media_types
from vibekimus.duplicates import DuplicateFinder, group_rows, similar_pairs, space_freed, split_groups
from vibekimus.filters import ResultFilters, SortOrder
from vibekimus.media_types import MediaType
from vibekimus.search import SearchEngine, SearchOptions

SAME = 0.999


def keepers(report):
    return [os.path.basename(r.path) for r in report.results if r.keeper]


def removed(report):
    return {os.path.basename(path) for path in report.to_remove}


def test_duplicates_are_grouped_and_the_highest_resolution_copy_is_kept(details):
    report = DuplicateFinder(details).find(threshold=SAME)
    assert report.groups == 1 and keepers(report) == ["red_large.png"] and removed(report) == {"red_small.png"}


def duplicates_with(library, filters):
    allowed = SearchEngine(library).allowed_rows(FakeModel(), library.images, SearchOptions(filters=filters))
    return DuplicateFinder(library).find(SAME, allowed=allowed)


def test_a_group_is_shown_whole_when_one_copy_passes_the_filters(details):
    large_bytes = os.path.getsize(os.path.join(details.scope.data_dir, "red_large.png"))
    for filters in (ResultFilters(min_bytes=large_bytes), ResultFilters(min_pixels=1280 * 720)):
        report = duplicates_with(details, filters)
        assert names(report.results) == ["red_large.png", "red_small.png"] and removed(report) == {"red_small.png"}


def test_a_group_is_left_out_when_no_copy_passes_the_filters(details):
    report = duplicates_with(details, ResultFilters(folder=os.path.join(details.scope.data_dir, "trip")))
    assert report.groups == 0 and not report.results


def test_groups_follow_the_sort_order_of_their_kept_copy(pairs):
    by_pixels = DuplicateFinder(pairs).find(SAME, sort=SortOrder.MOST_PIXELS)
    by_age = DuplicateFinder(pairs).find(SAME, sort=SortOrder.OLDEST)
    assert keepers(by_pixels) == ["red_x.png", "blue_a.png"] and keepers(by_age) == ["blue_a.png", "red_x.png"]
    assert [r.group for r in by_age.results] == [1, 1, 2, 2], "numbered in that order"


def test_most_space_to_free_orders_groups_by_the_bytes_of_their_copies(pairs):
    report = DuplicateFinder(pairs).find(SAME, sort=SortOrder.SPACE_FREED)
    groups = split_groups(report.results)
    freed = [space_freed(group) for group in groups]
    copy_bytes = [r.bytes for r in report.results if not r.keeper]
    assert len(groups) == 2 and freed == sorted(freed, reverse=True) and freed[0] > 0
    assert sorted(freed) == sorted(copy_bytes)
    assert all(group[0].keeper for group in groups) and sum(map(len, groups)) == len(report.results)


def test_duplicates_respect_the_type_filter(pairs, monkeypatch):
    monkeypatch.setattr(media_types, "MEDIA_TYPES", (MediaType("reds", "Reds", ("red",)),
                                                     MediaType("blues", "Blues", ("blue",))))
    allowed = SearchEngine(pairs).allowed_rows(FakeModel(), pairs.images, SearchOptions(media_type="blues"))
    only_blue = DuplicateFinder(pairs).find(SAME, allowed=allowed)
    assert keepers(only_blue) == ["blue_a.png"] and only_blue.groups == 1


def test_a_chain_of_look_alikes_is_not_glued_into_one_group():
    chain = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]   # each picture like the next one only
    assert max(len(group) for group in group_rows(chain)) == 3, "members are like the group's centre"


def test_near_duplicate_pairs_equal_a_brute_force_comparison():
    vectors = np.random.default_rng(3).standard_normal((700, 8)).astype(np.float32)
    vectors[350] = vectors[10] + 0.001
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    brute = {(i, j) for i in range(700) for j in range(i + 1, 700) if vectors[i] @ vectors[j] >= 0.9}
    assert {tuple(pair) for pair in similar_pairs(vectors, 0.9).tolist()} == brute
