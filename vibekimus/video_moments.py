"""Where in a video a search matches: every sampled moment of a video that a search
found, with its score, and which of those moments match. A moment matches when it is
among the search's results and its score is close to the video's best moment, measured
in the spread of all frames' scores (so the same for every model)."""
from dataclasses import dataclass

import numpy as np

from vibekimus import settings


@dataclass(frozen=True)
class Moment:
    seconds: float
    score: float      # -inf for a frame the filters leave out
    matches: bool     # among the results of the search
    key: object       # the frame's key in the video index


@dataclass(frozen=True)
class VideoTimeline:
    moments: tuple    # every sampled frame of the video, in time order

    @property
    def matching(self):
        return tuple(moment for moment in self.moments if moment.matches)

    @property
    def length(self):
        """About how long the video is: frames are sampled at equal steps beginning one
        step in (video_io.sample_times), so the end is one step after the last."""
        first, last = self.moments[0].seconds, self.moments[-1].seconds
        return max(last + first, np.finfo(np.float32).eps)

    def score_range(self):
        """(lowest, highest) score of the frames the filters let through; equal when
        there is only one."""
        scores = [moment.score for moment in self.moments if np.isfinite(moment.score)]
        return (min(scores), max(scores)) if scores else (0.0, 0.0)

    def next_matching(self, seconds, direction):
        """The matching moment after (direction 1) or before (-1) `seconds`, or None."""
        later = [m for m in self.matching if (m.seconds - seconds) * direction > 0]
        if not later:
            return None
        return min(later, key=lambda moment: abs(moment.seconds - seconds))


def timelines_of(index, scores, results):
    """{video file: VideoTimeline} for the videos among `results` (SearchResults of
    the video index), from the scores of every row of the index."""
    file_of, seconds_of = index.format.file_of, index.format.moment_of
    found = {result.key: result.score for result in results}
    finite = scores[np.isfinite(scores)]
    margin = settings.MOMENT_MATCH_SPREAD * (float(finite.std()) if finite.size else 0.0)
    timelines = {}
    for file, rows in index.rows_of_files({file_of(result.key) for result in results}).items():
        frames = sorted((index.keys[row], float(scores[row])) for row in rows)
        best = max(found[key] for key, _score in frames if key in found)
        timelines[file] = VideoTimeline(tuple(Moment(seconds_of(key), score, key in found and score >= best - margin,
                                                     key) for key, score in frames))
    return timelines
