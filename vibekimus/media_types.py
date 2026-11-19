"""Telling photos, memes, screenshots, drawings and documents apart, using the
embeddings that are already in the index (no extra indexing).

Each type is described by a few short texts; an image belongs to the type whose
averaged text embedding it is closest to. Computed once per index and model.
"""
from dataclasses import dataclass

import numpy as np

from vibekimus.identity_cache import IdentityCache
from vibekimus.vectors import unit

TYPE_VECTORS = "type vectors"


@dataclass(frozen=True)
class MediaType:
    key: str
    label: str
    prompts: tuple


MEDIA_TYPES = (
    MediaType("photo", "Photos", (
        "a photo", "a photograph taken with a camera", "a smartphone photo of a real scene",
        "a candid photo of people", "a landscape photograph")),
    MediaType("meme", "Memes", (
        "an internet meme", "a meme with a funny caption", "a reaction image with text on top and bottom",
        "a funny picture with impact font text")),
    MediaType("screenshot", "Screenshots", (
        "a screenshot of a computer screen", "a screenshot of a phone app", "a screenshot of a website",
        "a screenshot of a video game", "a screenshot of a chat conversation")),
    MediaType("drawing", "Anime & drawings", (
        "an anime illustration", "a cartoon drawing", "manga art", "digital art", "a hand-drawn sketch")),
    MediaType("document", "Documents", (
        "a scanned document", "a page of printed text", "a receipt", "a form", "a slide from a presentation")),
)


def media_type_by_key(key):
    return next((media_type for media_type in MEDIA_TYPES if media_type.key == key), None)


class MediaTypeClassifier:
    """Labels every row of an index with the position of its type in MEDIA_TYPES."""

    def __init__(self, similarity):
        self._similarity = similarity
        self._type_vectors = IdentityCache()   # TYPE_VECTORS -> vectors of an encoder
        self._labels = IdentityCache()         # index name -> labels of (its embeddings, the type vectors)

    def invalidate(self):
        self._type_vectors.clear()
        self._labels.clear()

    def mask(self, name, embeddings, encoder, type_key):
        """Boolean array: which rows are of the given type."""
        wanted = next(i for i, media_type in enumerate(MEDIA_TYPES) if media_type.key == type_key)
        return self.labels(name, embeddings, encoder) == wanted

    def labels(self, name, embeddings, encoder):
        """Per row: the position of its type in MEDIA_TYPES."""
        vectors = self._vectors_for(encoder)
        labels = self._labels.get(name, embeddings, vectors)
        if labels is not None:
            return labels
        device = encoder.search_device
        scores = np.stack([self._similarity.scores(name, embeddings, vector, device) for vector in vectors])
        return self._labels.put(name, scores.argmax(axis=0), embeddings, vectors)

    def _vectors_for(self, encoder):
        """One unit vector per type: the average of its prompts' embeddings."""
        vectors = self._type_vectors.get(TYPE_VECTORS, encoder)
        if vectors is not None:
            return vectors
        means = [encoder.encode_text(list(media_type.prompts)).mean(axis=0) for media_type in MEDIA_TYPES]
        return self._type_vectors.put(TYPE_VECTORS, [unit(mean) for mean in means], encoder)
