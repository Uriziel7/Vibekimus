"""Explore: the pictures of the library grouped into topics without typing anything.

The embeddings already in the index are grouped with spherical k-means (each picture
goes to the topic whose average direction it is closest to); every topic is named
after the two words of topic_words that set it apart from the other topics most. A
topic can be grouped again into smaller topics. Nothing is read from disk and nothing
is saved."""
from dataclasses import dataclass, field

import numpy as np

from vibekimus import settings
from vibekimus.identity_cache import IdentityCache
from vibekimus.runlog import safe_print
from vibekimus.topic_words import TOPIC_WORDS
from vibekimus.vectors import unit_rows

SEED = 0                    # the same library gives the same topics every time
STABLE_SHARE = 0.001        # grouping stops once fewer pictures than this share change topic
NAME_WORDS = 2
NAME_SEPARATOR = " · "
WORD_VECTORS = "word vectors"


@dataclass(frozen=True)
class Topic:
    keys: tuple             # image index keys, closest to the topic's centre first
    centre: np.ndarray = field(repr=False)
    name: str = ""

    @property
    def size(self):
        return len(self.keys)


def topic_count(pictures):
    """How many topics `pictures` are grouped into: settings.EXPLORE_TOPICS, fewer when
    a topic would get less than settings.EXPLORE_MIN_TOPIC_SIZE pictures on average."""
    return max(1, min(settings.EXPLORE_TOPICS, pictures // max(1, settings.EXPLORE_MIN_TOPIC_SIZE)))


def spherical_kmeans(vectors, count, device=None, iterations=None):
    """(centres (count, D), labels (N,)) of unit vectors grouped by cosine similarity.
    Starts from k-means++ picks; a topic left empty takes the picture worst served."""
    import torch
    iterations = iterations or settings.EXPLORE_ITERATIONS
    data = torch.as_tensor(np.ascontiguousarray(vectors, dtype=np.float32), device=device or "cpu")
    generator = torch.Generator(device=data.device).manual_seed(SEED)
    centres = _plus_plus(data, count, generator)
    labels = None
    for _round in range(iterations):
        similarity = data @ centres.T
        best, new_labels = similarity.max(dim=1)
        changed = data.shape[0] if labels is None else int((new_labels != labels).sum())
        labels = new_labels
        centres = _centres(data, labels, count, best)
        if changed <= STABLE_SHARE * data.shape[0]:
            break
    return centres.cpu().numpy(), labels.cpu().numpy()


def _plus_plus(data, count, generator):
    """k-means++: each next centre drawn with probability growing with its distance to
    the centres chosen so far."""
    import torch
    first = torch.randint(data.shape[0], (1,), generator=generator, device=data.device)
    chosen = [first]
    nearest = 1 - (data @ data[first].T).squeeze(1)
    for _ in range(1, count):
        weights = nearest.clamp(min=0) + torch.finfo(data.dtype).tiny
        pick = torch.multinomial(weights, 1, generator=generator)
        chosen.append(pick)
        nearest = torch.minimum(nearest, 1 - (data @ data[pick].T).squeeze(1))
    return data[torch.cat(chosen)]


def _centres(data, labels, count, best):
    import torch
    sums = torch.zeros((count, data.shape[1]), device=data.device, dtype=data.dtype)
    sums.index_add_(0, labels, data)
    empty = (torch.bincount(labels, minlength=count) == 0).nonzero().flatten()
    worst = torch.argsort(best)[:len(empty)]
    sums[empty] = data[worst]
    return torch.nn.functional.normalize(sums, dim=1)


class TopicNamer:
    """Names topics after words of TOPIC_WORDS (encoded once per model): those whose
    similarity to the topic's centre is highest compared with the other topics shown
    with it. Words merely closest to each centre repeat ("people", "a screenshot"):
    on the user's library 19 different names for 36 topics, 10 for 36 topics inside
    one; compared with the others, 34 and 35."""

    def __init__(self):
        self._words = IdentityCache()   # WORD_VECTORS -> unit vectors of TOPIC_WORDS by an encoder

    def names(self, encoder, centres):
        """One name per centre (an array of unit vectors, one row per topic)."""
        similarity = np.asarray(centres) @ self._vectors(encoder).T
        apart = similarity - similarity.mean(axis=0) if len(similarity) > 1 else similarity
        return [NAME_SEPARATOR.join(TOPIC_WORDS[i] for i in np.argsort(-row)[:NAME_WORDS]) for row in apart]

    def _vectors(self, encoder):
        vectors = self._words.get(WORD_VECTORS, encoder)
        if vectors is None:
            vectors = self._words.put(WORD_VECTORS, unit_rows(encoder.encode_text(list(TOPIC_WORDS))), encoder)
        return vectors


class TopicFinder:
    """Groups the pictures of a library (or of one topic) into topics."""

    def __init__(self, library, namer=None):
        self._library = library
        self._namer = namer or TopicNamer()

    def find(self, encoder, allowed=None, within=None, device=None):
        """Topics, largest first. allowed: boolean array of the image rows to use (the
        filter bar; pictures indexed since do not count), None = all; within: a Topic to group
        further. Excluded files are left out."""
        index = self._library.images.view()
        rows = self._rows(index, allowed, within)
        if not len(rows):
            return []
        vectors = unit_rows(index.embeddings[rows].astype(np.float32))
        centres, labels = self._grouped(vectors, topic_count(len(rows)), device)
        groups = [(centre, np.flatnonzero(labels == number)) for number, centre in enumerate(centres)]
        groups = [(centre, members) for centre, members in groups if len(members)]
        names = self._namer.names(encoder, [centre for centre, _members in groups])
        topics = [Topic(tuple(index.keys[rows[i]] for i in members[np.argsort(-(vectors[members] @ centre))]),
                        centre, name) for (centre, members), name in zip(groups, names)]
        return sorted(topics, key=lambda topic: -topic.size)

    @staticmethod
    def _grouped(vectors, count, device):
        """On the graphics card, or in RAM when it has no room (a large library on a small
        card: the pictures take about 4.6 GB per million there)."""
        try:
            return spherical_kmeans(vectors, count, device)
        except RuntimeError as error:   # torch's out-of-memory error is one
            if device is None:
                raise
            safe_print(f"[EXPLORE] Grouping on the graphics card failed ({error}); grouping in RAM")
            return spherical_kmeans(vectors, count, None)

    def _rows(self, index, allowed, within):
        rows = index.kept(self._library.exclusions)
        if allowed is not None:   # rows indexed since the filters were read do not pass
            known = min(len(allowed), len(rows))
            rows[:known] &= allowed[:known]
            rows[known:] = False
        if within is not None:
            members = set(within.keys)
            rows &= np.fromiter((key in members for key in index.keys), dtype=bool, count=len(index.keys))
        return np.flatnonzero(rows)
