"""Finding duplicate and near-duplicate images with the embeddings already in the index.

Every image is compared with every other one (in blocks, on the GPU when there is
one); images above a similarity threshold to a group's centre form the group. In
each group the best copy (highest resolution, then largest file, then oldest) is the
one to keep.
"""
from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np

from vibekimus.filters import SortOrder
from vibekimus.media_info import best_date
from vibekimus.media_kind import MediaKind
from vibekimus.runlog import safe_print
from vibekimus.search import SearchResult

BLOCK_ROWS = 4096   # rows compared with the whole index per step
SAME_FILE_SCORE = 1.0   # the copy to keep, compared with itself


def similar_pairs(embeddings, threshold, device=None, should_stop=lambda: False):
    """(row, row) pairs with cosine similarity >= threshold, each pair once."""
    if device is not None:
        try:
            return _similar_pairs_torch(embeddings, threshold, device, should_stop)
        except Exception as error:
            safe_print(f"[DUPLICATES] GPU comparison unavailable ({error}), using CPU")
    return _similar_pairs_numpy(embeddings, threshold, should_stop)


def _similar_pairs_torch(embeddings, threshold, device, should_stop):
    import torch
    everything = torch.from_numpy(np.ascontiguousarray(embeddings, dtype=np.float32)).to(device).half()
    rows = everything.shape[0]
    pairs = []
    for start in range(0, rows, BLOCK_ROWS):
        if should_stop():
            break
        block = everything[start:start + BLOCK_ROWS]
        similar = (block @ everything.T).float() >= threshold
        # keep j > i only: each pair once, and never an image with itself
        similar &= torch.arange(rows, device=device)[None, :] > torch.arange(
            start, start + block.shape[0], device=device)[:, None]
        found = torch.nonzero(similar).cpu().numpy()
        found[:, 0] += start
        pairs.append(found)
    return np.concatenate(pairs) if pairs else np.empty((0, 2), dtype=np.int64)


def _similar_pairs_numpy(embeddings, threshold, should_stop):
    rows = embeddings.shape[0]
    pairs = []
    for start in range(0, rows, BLOCK_ROWS):
        if should_stop():
            break
        block = embeddings[start:start + BLOCK_ROWS]
        similar = block @ embeddings.T >= threshold
        similar &= np.arange(rows)[None, :] > np.arange(start, start + block.shape[0])[:, None]
        found = np.argwhere(similar)
        found[:, 0] += start
        pairs.append(found)
    return np.concatenate(pairs) if pairs else np.empty((0, 2), dtype=np.int64)


def group_rows(pairs):
    """Groups of rows, largest first. Every member is directly similar to its group's
    centre — pairs are not chained (A~B and B~C does not put A with C), which would
    glue long runs of slightly different images into one huge group. The image with
    the most look-alikes becomes a centre first."""
    neighbours = defaultdict(set)
    for first, second in pairs:
        neighbours[int(first)].add(int(second))
        neighbours[int(second)].add(int(first))
    assigned, groups = set(), []
    for centre in sorted(neighbours, key=lambda row: len(neighbours[row]), reverse=True):
        if centre in assigned:
            continue
        members = [centre] + sorted(neighbours[centre] - assigned)
        if len(members) > 1:
            assigned.update(members)
            groups.append(sorted(members))
    return sorted(groups, key=len, reverse=True)


def keeper_of(rows, info):
    """The copy worth keeping: most pixels, then the largest file, then the oldest."""
    dates = best_date(info[rows])

    def quality(position):
        record = info[rows[position]]
        pixels = int(record["width"]) * int(record["height"])
        date = dates[position] if dates[position] > 0 else float("inf")
        return pixels, int(record["bytes"]), -date

    return rows[max(range(len(rows)), key=quality)]


def space_freed(group):
    """Bytes that deleting every copy of a group (all but its keeper) frees."""
    return sum(result.bytes for result in group if not result.keeper and result.bytes > 0)


def split_groups(results):
    """The groups of a duplicates result list (consecutive results with the same
    group number, the keeper first)."""
    groups = []
    for result in results:
        if groups and groups[-1][0].group == result.group:
            groups[-1].append(result)
        else:
            groups.append([result])
    return groups


@dataclass
class DuplicateReport:
    results: list          # SearchResults, group by group, the keeper first in each
    to_remove: set         # paths of every copy except the keepers
    groups: int


class DuplicateFinder:
    """Finds duplicate images in a library's image index."""

    def __init__(self, library):
        self._library = library

    def find(self, threshold, device=None, should_stop=lambda: False, allowed=None, sort=SortOrder.RELEVANCE):
        """allowed: boolean array of the image rows passing the search filters, None = all.
        A group is shown whole when at least one of its copies passes — the copies
        themselves may be smaller, older or elsewhere. sort orders the groups by their
        keeper; RELEVANCE = largest group first."""
        index = self._library.images.view()
        if not index.keys:
            return DuplicateReport([], set(), 0)
        rows = np.flatnonzero(index.kept(self._library.exclusions))
        pairs = rows[similar_pairs(index.embeddings[rows], threshold, device, should_stop)]
        groups = [self._group_results(index, members) for members in group_rows(pairs)
                  if allowed is None or allowed[members].any()]
        return self._report(self._ordered(groups, sort))

    def _group_results(self, index, rows):
        """SearchResults of one group, the keeper first."""
        to_abs = self._library.scope.absolute_path
        date_of = dict(zip(rows, best_date(index.info[rows])))
        keeper = keeper_of(rows, index.info)
        group = []
        for row in [keeper] + [row for row in rows if row != keeper]:
            score = SAME_FILE_SCORE if row == keeper else index.embeddings[row] @ index.embeddings[keeper]
            key = index.keys[row]
            group.append(SearchResult.of_row(score, to_abs(key), MediaKind.IMAGE, key, index.info[row], date_of[row],
                                             keeper=row == keeper))
        return group

    @staticmethod
    def _ordered(groups, sort):
        """Groups by the space deleting their copies frees, or in the order of their
        keepers under any other sort."""
        if sort is SortOrder.SPACE_FREED:
            return sorted(groups, key=space_freed, reverse=True)
        position = {id(group[0]): group for group in groups}
        return [position[id(keeper)] for keeper in sort.apply([group[0] for group in groups])]

    @staticmethod
    def _report(groups):
        results, to_remove = [], set()
        for number, group in enumerate(groups, start=1):
            results += [replace(result, group=number) for result in group]
            to_remove.update(result.path for result in group if not result.keeper)
        return DuplicateReport(results, to_remove, len(groups))
