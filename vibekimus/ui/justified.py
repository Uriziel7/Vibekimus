"""Justified rows: pictures laid out in rows that fill the full width exactly, each
keeping its own proportions (as in modern photo galleries). Pure geometry, no Tk."""
from dataclasses import dataclass

MIN_ASPECT = 0.5    # narrower pictures (tall screenshots) get a slot of this width : height…
MAX_ASPECT = 2.5    # …wider ones (panoramas) of this one; the picture is cropped to fit
DEFAULT_ASPECT = 1.0


@dataclass(frozen=True)
class Slot:
    x: int
    y: int
    width: int
    height: int


def clamp_aspect(width, height):
    """width : height of a picture, kept within MIN_ASPECT..MAX_ASPECT (DEFAULT_ASPECT
    when unknown)."""
    if width <= 0 or height <= 0:
        return DEFAULT_ASPECT
    return min(MAX_ASPECT, max(MIN_ASPECT, width / height))


def _fitted_height(row, available_width, gap):
    """Height at which a row of these aspects is exactly available_width wide."""
    return (available_width - gap * (len(row) - 1)) / sum(row)


def justified_rows(aspects, available_width, row_height, gap):
    """Slots for pictures of the given aspects, in order. Pictures are added to a row
    until it reaches available_width; the row then ends before or after the last one
    added, whichever brings its height closer to row_height, and is scaled to fill the
    width exactly. The last, unfilled row keeps row_height and is aligned left.
    Returns ([Slot per picture], total height)."""
    slots, row, y = [], [], 0
    for aspect in aspects:
        row.append(aspect)
        if sum(row) * row_height + gap * (len(row) - 1) < available_width:
            continue
        carried = []
        if len(row) > 1:
            with_last = _fitted_height(row, available_width, gap)
            without_last = _fitted_height(row[:-1], available_width, gap)
            if abs(without_last - row_height) < abs(with_last - row_height):
                row, carried = row[:-1], row[-1:]
        height = _fitted_height(row, available_width, gap)
        slots += _place_row(row, y, height, gap, available_width)
        y += round(height) + gap
        row = carried
    if row:
        slots += _place_row(row, y, row_height, gap, None)
        y += row_height + gap
    return slots, max(0, y - gap)


def _place_row(aspects, y, height, gap, width):
    """Slots of one row; with `width` the last slot ends exactly there (rounding)."""
    slots, x = [], 0.0
    for position, aspect in enumerate(aspects):
        is_last = position == len(aspects) - 1
        right = width if (is_last and width is not None) else x + aspect * height
        slots.append(Slot(round(x), y, max(1, round(right) - round(x)), max(1, round(height))))
        x = right + gap
    return slots
