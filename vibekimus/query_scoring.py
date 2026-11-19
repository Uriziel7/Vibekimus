"""What a search compares the index with, and how each row is scored.

A text query scores rows by their similarity to the wanted words minus the unwanted
ones; an example (a picture, or a result already in the index) by its similarity to
that picture. A picture *and* text together ("this, but at night") cannot simply be
added: picture-to-picture similarities are several times larger than text-to-picture
ones (SigLIP: about 0.5–1.0 against 0.05–0.2), so the picture would drown the text.
Each part is therefore first standardised over the whole index — how far a row
stands out from the average row, in standard deviations — and only then mixed."""
from abc import ABC, abstractmethod

import numpy as np


def standardized(scores):
    """Scores as standard deviations from their mean (all zeros when they are equal).
    Rows left out (-inf) stay left out and do not count."""
    finite = np.isfinite(scores)
    if not finite.all():
        result = np.full_like(scores, -np.inf)
        result[finite] = standardized(scores[finite])
        return result
    spread = float(scores.std()) if scores.size else 0.0
    return (scores - scores.mean()) / spread if spread > 0 else np.zeros_like(scores)


class Query(ABC):
    # Whether the Min. score slider applies: it is a similarity, which mixed scores are not.
    uses_min_score = True

    @abstractmethod
    def scores(self, similarity, name, embeddings, device):
        """(N,) float32 scores of the index rows; similarity is a SimilarityEngine."""


class TextQuery(Query):
    """Wanted terms (a vector, or None) minus unwanted ones (a vector, or None)."""

    def __init__(self, wanted, unwanted):
        self._wanted, self._unwanted = wanted, unwanted

    def scores(self, similarity, name, embeddings, device):
        total = np.zeros(embeddings.shape[0], dtype=np.float32)
        if self._wanted is not None:
            total += similarity.scores(name, embeddings, self._wanted, device)
        if self._unwanted is not None:
            total -= similarity.scores(name, embeddings, self._unwanted, device)
        return total


class ExampleQuery(Query):
    """Rows like one picture's embedding."""

    def __init__(self, vector):
        self._vector = vector

    def scores(self, similarity, name, embeddings, device):
        return similarity.scores(name, embeddings, self._vector, device)


class CombinedQuery(Query):
    """A picture and text together: both standardised over the index, then mixed —
    text_weight for the text, the rest for the picture."""

    uses_min_score = False

    def __init__(self, example, text, text_weight):
        self._example, self._text, self._text_weight = example, text, text_weight

    def scores(self, similarity, name, embeddings, device):
        picture = standardized(self._example.scores(similarity, name, embeddings, device))
        text = standardized(self._text.scores(similarity, name, embeddings, device))
        return (1 - self._text_weight) * picture + self._text_weight * text


class AmongRows(Query):
    """Another query, with only some rows of one index qualifying (an Explore topic)."""

    def __init__(self, base, name, rows):
        """rows: boolean array over the rows of index `name` (rows added later do not qualify)."""
        self._base, self._name, self._rows = base, name, rows
        self.uses_min_score = base.uses_min_score

    def scores(self, similarity, name, embeddings, device):
        qualifying = np.zeros(embeddings.shape[0], dtype=bool)
        if name == self._name:
            known = min(len(self._rows), embeddings.shape[0])
            qualifying[:known] = self._rows[:known]
        if not qualifying.any():
            return np.full(embeddings.shape[0], -np.inf, dtype=np.float32)
        return np.where(qualifying, self._base.scores(similarity, name, embeddings, device), -np.inf)


class RefinedQuery(Query):
    """A search refined by ratings: its own scores, plus the similarity to the average of
    the results rated "more like this", minus the similarity to the average of those
    rated "less like this" — each part standardised over the index first, as above.
    liked / disliked: ExampleQuery or None."""

    uses_min_score = False

    def __init__(self, base, liked, disliked, liked_weight, disliked_weight):
        self._base, self._liked, self._disliked = base, liked, disliked
        self._liked_weight, self._disliked_weight = liked_weight, disliked_weight

    def scores(self, similarity, name, embeddings, device):
        base = self._base.scores(similarity, name, embeddings, device)
        qualifying = np.isfinite(base)   # every part is standardised over the same rows

        def part(example):
            return standardized(np.where(qualifying, example.scores(similarity, name, embeddings, device), -np.inf))

        total = standardized(base)
        with np.errstate(invalid="ignore"):   # -inf minus -inf (rows left out) is NaN, replaced below
            if self._liked is not None:
                total += self._liked_weight * part(self._liked)
            if self._disliked is not None:
                total -= self._disliked_weight * part(self._disliked)
        return np.where(qualifying, total, -np.inf)
