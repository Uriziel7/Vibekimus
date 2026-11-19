"""Rating results while searching: ▲ "more like this" and ▼ "less like this". Refining
the search then moves it toward the liked results and away from the disliked ones
(relevance feedback, as in Rocchio's method). Results without a rating do not count."""
from dataclasses import dataclass
from enum import Enum


class Rating(Enum):
    LIKE = 1
    DISLIKE = -1
    UNRATED = 0

    def replacing(self, given):
        """What a result rated `given` gets when this rating is chosen for it: chosen
        again, the rating is taken back."""
        return Rating.UNRATED if given == self else self


@dataclass(frozen=True)
class Feedback:
    """The ratings a search is refined with: (kind, key) of each rated result."""
    liked: tuple = ()
    disliked: tuple = ()

    def __bool__(self):
        return bool(self.liked or self.disliked)


class Ratings:
    """The ratings given to the results of the shown search. A result is one picture or
    one frame of a video (its kind and key in the index)."""

    def __init__(self):
        self._marks = {}   # (kind, key) -> LIKE or DISLIKE

    @staticmethod
    def _id(result):
        return result.kind, result.key

    def of(self, result):
        return self._marks.get(self._id(result), Rating.UNRATED)

    def rate(self, result, mark):
        """Rate a result LIKE or DISLIKE; UNRATED takes its rating away."""
        if mark == Rating.UNRATED:
            self._marks.pop(self._id(result), None)
        else:
            self._marks[self._id(result)] = mark

    def clear(self):
        self._marks.clear()

    def counts(self):
        """(liked, disliked)"""
        marks = list(self._marks.values())
        return marks.count(Rating.LIKE), marks.count(Rating.DISLIKE)

    def feedback(self):
        return Feedback(tuple(i for i, mark in self._marks.items() if mark == Rating.LIKE),
                        tuple(i for i, mark in self._marks.items() if mark == Rating.DISLIKE))

    def __bool__(self):
        return bool(self._marks)
