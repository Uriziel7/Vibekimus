"""The user's choices that shape an indexing run (set in the Settings window)."""
from dataclasses import dataclass, field

from vibekimus import settings
from vibekimus.decoding import DecodingOptions


@dataclass(frozen=True)
class IndexingOptions:
    decoding: DecodingOptions = field(default_factory=DecodingOptions)
    # Videos longer than this are not indexed; 0 = no limit.
    max_video_seconds: int = field(default_factory=lambda: settings.VIDEO_MAX_DURATION_SECONDS)
