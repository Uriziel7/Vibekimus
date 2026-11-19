"""Facts about a media file that results can be filtered and sorted by: when it was
taken or last modified, its size on disk and its pixel dimensions.

Every index row has one record of INFO_DTYPE. Values that are not known (older
indexes, files whose header cannot be read) are UNKNOWN.
"""
import io
import os
import time
from datetime import datetime

import numpy as np

from vibekimus.files import long_path
from vibekimus.imaging import is_raw, open_picture

UNKNOWN = -1
INFO_DTYPE = np.dtype([
    ("modified", "<f8"),   # file modification time, seconds since 1970
    ("taken", "<f8"),      # EXIF date the photo was taken, seconds since 1970
    ("bytes", "<i8"),      # file size
    ("width", "<i4"),
    ("height", "<i4"),
])
EXIF_IFD_POINTER = 0x8769
EXIF_DATE_TAKEN = 36867      # DateTimeOriginal
EXIF_DATE_TIME = 306         # DateTime (when the file was last edited in camera / software)
EXIF_DATE_FORMAT = "%Y:%m:%d %H:%M:%S"
EPOCH = datetime(1970, 1, 1)


def local_timestamp(moment):
    """Seconds since 1970 of a local date and time (a naive datetime). Windows cannot
    convert dates before 1970 (datetime.timestamp() raises OSError there); those are
    counted from 1970 with the zone's standard offset (summer time aside)."""
    try:
        return moment.timestamp()
    except (OSError, OverflowError, ValueError):
        return (moment - EPOCH).total_seconds() + time.timezone


def unknown_info(rows):
    """`rows` records with every value UNKNOWN."""
    info = np.empty(rows, dtype=INFO_DTYPE)
    for name in INFO_DTYPE.names:
        info[name] = UNKNOWN
    return info


def as_info_array(records):
    """A list of record tuples (or an info array) as an INFO_DTYPE array."""
    return np.array(records, dtype=INFO_DTYPE) if len(records) else unknown_info(0)


def _file_stat(path):
    try:
        stat = os.stat(long_path(path))
        return stat.st_mtime, stat.st_size
    except OSError:
        return UNKNOWN, UNKNOWN


def _exif_date(image):
    """When the photo was taken, from EXIF, as seconds since 1970 (local time)."""
    try:
        exif = image.getexif()
        text = exif.get_ifd(EXIF_IFD_POINTER).get(EXIF_DATE_TAKEN) or exif.get(EXIF_DATE_TIME)
        if not text:
            return UNKNOWN
        return time.mktime(datetime.strptime(str(text).strip("\x00 "), EXIF_DATE_FORMAT).timetuple())
    except Exception:
        return UNKNOWN


def image_info(path, data=None):
    """Info record of an image file. Reads only the file header (no decoding);
    `data` may hold the file's bytes when they have been read already."""
    modified, size = _file_stat(path)
    width = height = taken = UNKNOWN
    if not is_raw(path):
        try:
            source = io.BytesIO(data) if data is not None else long_path(path)
            with open_picture(source, path) as image:
                width, height = image.size
                taken = _exif_date(image)
        except Exception:
            pass
    return modified, taken, size, width, height


def video_info(path, width=UNKNOWN, height=UNKNOWN):
    """Info record of a video file; the dimensions come from the decoder."""
    modified, size = _file_stat(path)
    return modified, UNKNOWN, size, width, height


def best_date(info):
    """Per record: the date taken when known, otherwise the modification date."""
    return np.where(info["taken"] > 0, info["taken"], info["modified"])


def is_missing(info):
    """Per record: True where nothing is known yet (rows from an older index)."""
    return info["modified"] == UNKNOWN
