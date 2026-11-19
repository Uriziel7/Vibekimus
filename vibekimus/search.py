"""Searching the library by text, by example image, or by an already indexed result."""
import re
from dataclasses import dataclass, field, replace

import numpy as np

from vibekimus import settings
from vibekimus.filters import ResultFilters, SortOrder
from vibekimus.media_info import UNKNOWN, best_date
from vibekimus.media_kind import MediaKind
from vibekimus.media_types import MediaTypeClassifier
from vibekimus.query_scoring import AmongRows, CombinedQuery, ExampleQuery, RefinedQuery, TextQuery
from vibekimus.relevance import Feedback
from vibekimus.runlog import safe_print
from vibekimus.similarity import SimilarityEngine, top_indices
from vibekimus.text_matching import TextMatch
from vibekimus.video_moments import timelines_of

# -"quoted phrase", -word, "quoted phrase", or plain word
QUERY_TOKEN_PATTERN = r'(-?"[^"]+"|[-\w]+)'
NEGATION_PREFIX = "-"
QUOTE = '"'
# Excluded folders are filtered after ranking, so fetch this many times more when any are set.
EXCLUSION_OVERFETCH = 4


@dataclass(frozen=True)
class SearchResult:
    score: float
    path: str               # absolute path of the image or video file
    kind: MediaKind
    timestamp: float = 0.0  # seconds into the video; unused for images
    key: object = None      # the row's key in its index (for "find similar")
    date: float = UNKNOWN   # taken or modified, seconds since 1970
    bytes: int = UNKNOWN
    width: int = UNKNOWN
    height: int = UNKNOWN
    group: int = 0          # duplicate group number (0 = not part of a duplicate search)
    keeper: bool = False    # the copy to keep in its duplicate group
    timeline: object = None  # a video's VideoTimeline: every sampled moment, which ones match

    @classmethod
    def of_row(cls, score, path, kind, key, record, date, timestamp=0.0, **fields):
        """The result of one index row; its file's size and dimensions come from the row's
        media_info record, `date` from media_info.best_date."""
        return cls(float(score), path, kind, timestamp, key, float(date), int(record["bytes"]),
                   int(record["width"]), int(record["height"]), **fields)

    @property
    def pixels(self):
        return self.width * self.height if self.width > 0 and self.height > 0 else UNKNOWN

    def at(self, moment):
        """This video result at another of its moments (a video_moments.Moment)."""
        return replace(self, timestamp=moment.seconds, score=moment.score, key=moment.key)


@dataclass(frozen=True)
class SearchOptions:
    min_score: float = settings.DEFAULT_MIN_SCORE
    include_images: bool = True
    include_videos: bool = True
    best_frame_per_video: bool = False
    filters: ResultFilters = field(default_factory=ResultFilters)
    media_type: str = None      # key of a media_types.MediaType, None = any
    sort: SortOrder = SortOrder.RELEVANCE
    feedback: Feedback = Feedback()   # ratings that refine the search (relevance.py)


class NoSearchTerms(ValueError):
    """The query has nothing to search for (empty, or exclusions only)."""


class NotInIndex(LookupError):
    """The result to search from is no longer in the index."""


def parse_query(query):
    """Split a query into (positive_terms, negative_terms).

    A term is excluded with a leading minus: -word or -"multi word phrase".
    """
    positive, negative = [], []
    for token in re.findall(QUERY_TOKEN_PATTERN, query):
        is_negative = token.startswith(NEGATION_PREFIX)
        term = token[len(NEGATION_PREFIX):].strip('"') if is_negative else token.strip('"')
        if term:
            (negative if is_negative else positive).append(term)
    return positive, negative


def quoted_and_loose(query):
    """The wanted (not excluded) terms of a query: ("quoted phrases", [loose words])."""
    phrases, loose = [], []
    for token in re.findall(QUERY_TOKEN_PATTERN, query):
        if not token.startswith(NEGATION_PREFIX) and token.strip(QUOTE):
            (phrases if token.startswith(QUOTE) else loose).append(token.strip(QUOTE))
    return phrases, loose


def best_frame_per_video(results):
    """Keep only the best-scoring frame of each video; images are kept as they are."""
    kept = [r for r in results if r.kind != MediaKind.VIDEO]
    best = {}
    for result in results:
        if result.kind == MediaKind.VIDEO and (result.path not in best or result.score > best[result.path].score):
            best[result.path] = result
    return kept + list(best.values())


class SearchEngine:
    """Ranks the library's images and video frames against a query."""

    def __init__(self, library, similarity=None):
        self._library = library
        self._similarity = similarity or SimilarityEngine()
        self._types = MediaTypeClassifier(self._similarity)

    def invalidate(self):
        """Forget cached copies of the indexes (call when the library or model changes)."""
        self._similarity.invalidate()
        self._types.invalidate()

    def by_text(self, encoder, query, options, is_cancelled=lambda: False):
        """Best matches for a text query. Returns None if cancelled.
        Raises NoSearchTerms if the query has no positive term.

        Once the text on the pictures has been read: a "quoted phrase" must appear on
        the picture, and pictures showing the loose words rank higher."""
        if not parse_query(query)[0]:
            raise NoSearchTerms(query)
        text = self._text_query(encoder, query, is_cancelled)
        return None if text is None else self._rank(encoder, text, options, self._text_on_pictures(query))

    def _text_on_pictures(self, query):
        """The TextMatch of a query, or None while no text has been read."""
        texts = self._library.texts
        return TextMatch(texts, *quoted_and_loose(query)) if texts.has_data else None

    def by_image(self, encoder, image, options, text=""):
        """Best matches for an example PIL image — with `text`, for the image and the
        text together ("this, but at night")."""
        example = ExampleQuery(encoder.encode_images([image])[0])
        return self._rank(encoder, self._with_text(encoder, example, text), options)

    def by_result(self, encoder, result, options, text=""):
        """Images and frames that look like an earlier result (and match `text`, if
        given), using its stored embedding: nothing is decoded. Raises NotInIndex."""
        index = self._library.images if result.kind == MediaKind.IMAGE else self._library.videos
        embedding = index.embedding_of(result.key)
        if embedding is None:
            raise NotInIndex(result.path)
        return self._rank(encoder, self._with_text(encoder, ExampleQuery(embedding), text), options)

    def by_topic(self, encoder, topic, options):
        """The pictures of an Explore topic (explore.Topic), closest to its centre first;
        filters, sorting and ratings apply as to any search."""
        images, members = self._library.images.view(), set(topic.keys)
        rows = np.fromiter((key in members for key in images.keys), dtype=bool, count=len(images.keys))
        query = AmongRows(ExampleQuery(topic.centre), images.format.kind, rows)
        return self._rank(encoder, query, replace(options, include_images=True, include_videos=False))

    def _with_text(self, encoder, example, text):
        """The example alone, or mixed with the text when it has any term (also only
        excluded ones: "this, -people")."""
        positive, negative = parse_query(text)
        if not positive and not negative:
            return example
        return CombinedQuery(example, self._text_query(encoder, text), settings.COMBINED_TEXT_WEIGHT)

    @staticmethod
    def _text_query(encoder, query, is_cancelled=lambda: False):
        """The TextQuery of a query, or None if cancelled while encoding."""
        positive, negative = parse_query(query)
        safe_print(f"[SEARCH] Positive: {positive}, Negative: {negative}")
        wanted = encoder.encode_text([" ".join(positive)]) if positive else None
        if is_cancelled():
            return None
        unwanted = encoder.encode_text([" ".join(negative)]) if negative else None
        if is_cancelled():
            return None
        return TextQuery(wanted, unwanted)

    def _refined(self, query, feedback):
        """The query moved toward the liked results and away from the disliked ones (those
        still in the index); unchanged without ratings."""
        liked, disliked = self._average_of(feedback.liked), self._average_of(feedback.disliked)
        if liked is None and disliked is None:
            return query
        return RefinedQuery(query, liked, disliked, settings.LIKED_WEIGHT, settings.DISLIKED_WEIGHT)

    def _average_of(self, rated):
        """ExampleQuery of the average embedding of rated results, or None."""
        indexes = {MediaKind.IMAGE: self._library.images, MediaKind.VIDEO: self._library.videos}
        vectors = [indexes[kind].embedding_of(key) for kind, key in rated]
        vectors = [vector for vector in vectors if vector is not None]
        return ExampleQuery(np.mean(np.asarray(vectors, dtype=np.float32), axis=0)) if vectors else None

    def _rank(self, encoder, query, options, text_match=None):
        query = self._refined(query, options.feedback)
        library = self._library
        to_abs = library.scope.absolute_path
        results = []
        if options.include_images:
            results += self._matches(encoder, library.images, query, options, text_match,
                                     lambda key: (to_abs(key), MediaKind.IMAGE, 0.0))
        if options.include_videos:
            results += self._matches(encoder, library.videos, query, options, text_match,
                                     lambda key: (to_abs(key[0]), MediaKind.VIDEO, key[1]), with_timelines)
        if options.best_frame_per_video:
            results = best_frame_per_video(results)
        results.sort(key=lambda r: r.score, reverse=True)
        return options.sort.apply(results[:settings.MAX_SEARCH_RESULTS])

    def _matches(self, encoder, index, query, options, text_match, describe, finish=None):
        """Only the best matches are turned into Python objects — building a list of
        every row above the threshold is what makes large libraries slow. Filters are
        applied before choosing the best, so a narrow filter still fills the page.
        finish(index, scores, results) -> results adds what needs the scores of all rows."""
        index = index.view()   # rows added by an indexing run that is still going count too
        embeddings, keys = index.embeddings, index.keys
        if embeddings is None or not keys:
            return []
        scores = query.scores(self._similarity, index.format.kind, embeddings, encoder.search_device)
        allowed = self.allowed_rows(encoder, index, options)
        if text_match is not None:
            scores = self._raise_shown_words(scores, text_match.word_share(index))
            allowed = _both(allowed, text_match.required_rows(index))
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
        exclusions = self._library.exclusions
        limit = settings.MAX_SEARCH_RESULTS * (EXCLUSION_OVERFETCH if exclusions else 1)
        file_of = index.format.file_of
        info, dates = index.info, best_date(index.info)
        # Mixed scores are no similarity, so the slider does not apply; rows left out
        # by the filters (-inf) still never qualify.
        min_score = options.min_score if query.uses_min_score else np.finfo(np.float32).min
        results = []
        for row in top_indices(scores, min_score, limit):
            key = keys[row]
            if exclusions.matches(file_of(key)):
                continue
            path, kind, timestamp = describe(key)
            results.append(SearchResult.of_row(scores[row], path, kind, key, info[row], dates[row], timestamp))
        return finish(index, scores, results) if finish is not None else results

    @staticmethod
    def _raise_shown_words(scores, word_share):
        """Scores plus a bonus for the share of the typed words seen on each picture.
        The bonus is counted in the scores' own spread, so it weighs the same with
        every model."""
        if word_share is None:
            return scores
        return scores + settings.OCR_WORD_BONUS * float(scores.std()) * word_share

    def media_types_of(self, encoder, index):
        """Per row of an index view: the position of its kind of picture in media_types.MEDIA_TYPES."""
        return self._types.labels(index.format.kind, index.view().embeddings, encoder)

    def allowed_rows(self, encoder, index, options):
        """Rows of an index passing the file filters and the media type, as a boolean
        array, or None for all. The encoder is needed only for a media type."""
        index = index.view()
        allowed = options.filters.row_mask(index, self._library.scope)
        if options.media_type:
            allowed = _both(allowed, self._types.mask(index.format.kind, index.embeddings, encoder,
                                                      options.media_type))
        return allowed


def with_timelines(index, scores, results):
    """Video results, each with the timeline of its video: where else in it the search matches."""
    timelines = timelines_of(index, scores, results)
    file_of = index.format.file_of
    return [replace(result, timeline=timelines[file_of(result.key)]) for result in results]


def _both(first, second):
    """Rows allowed by both masks (None = all rows)."""
    if first is None:
        return second
    return first if second is None else first & second
