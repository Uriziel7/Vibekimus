"""Turning numbers into the short texts shown in the window."""
from datetime import datetime

from vibekimus.relevance import Rating
from vibekimus.units import BYTES_PER_GB, BYTES_PER_KB, BYTES_PER_MB, SECONDS_PER_MINUTE

SIZE_UNITS = ((BYTES_PER_GB, "GB", 2), (BYTES_PER_MB, "MB", 1), (BYTES_PER_KB, "KB", 0))
DATE_FORMAT = "%Y-%m-%d"
DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
RATING_LABELS = {Rating.LIKE: "▲  More like this", Rating.DISLIKE: "▼  Less like this"}
GIVEN_MARK = "✓ "   # before the label of the rating a result has


def format_size(size_bytes):
    """File size in the largest unit that fits; the smallest unit is used below that."""
    *larger_units, smallest_unit = SIZE_UNITS
    fitting = [unit for unit in larger_units if size_bytes >= unit[0]]
    unit_bytes, name, decimals = fitting[0] if fitting else smallest_unit
    return f"{size_bytes / unit_bytes:.{decimals}f} {name}"


def format_date(seconds, with_time=False):
    return datetime.fromtimestamp(seconds).strftime(DATE_TIME_FORMAT if with_time else DATE_FORMAT)


def format_moment(seconds):
    """m:ss of a moment in a video."""
    minutes, rest = divmod(int(seconds), SECONDS_PER_MINUTE)
    return f"{minutes}:{rest:02d}"


def matching_moments(result):
    """The moments of a video result where the search matches (empty for a picture)."""
    return result.timeline.matching if result.timeline is not None else ()


def video_badge(result):
    """'▶ 1:23' on a video's tile, '▶ 1:23 · 4 moments' when the search matches it at
    more than one moment."""
    badge = f"▶ {format_moment(result.timestamp)}"
    count = len(matching_moments(result))
    return f"{badge}  ·  {count} moments" if count > 1 else badge


def format_details(result):
    """'2023-06-15 · 1920×1080 · 2.3 MB' with whatever is known about a result's file."""
    parts = []
    if result.date > 0:
        parts.append(format_date(result.date))
    if result.width > 0 and result.height > 0:
        parts.append(f"{result.width}×{result.height}")
    if result.bytes >= 0:
        parts.append(format_size(result.bytes))
    return "  ·  ".join(parts)
