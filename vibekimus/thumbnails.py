"""Making the small preview picture of a search result."""
from vibekimus.files import long_path
from vibekimus.imaging import decode_raw, is_raw, open_picture, upright
from vibekimus.media_kind import MediaKind
from vibekimus.video_io import read_frame_at

# A JPEG is decoded at a reduced size at least this many times the thumbnail's longest side
# (what PIL's own thumbnail() aims at), so the final resize still has detail to work with.
DRAFT_MARGIN = 2


class VideoSupportMissing(RuntimeError):
    """OpenCV is not installed, so video frames cannot be shown."""


class ThumbnailUnavailable(RuntimeError):
    """No preview could be made for this result."""


def _image_preview(path, size):
    if is_raw(path):
        return decode_raw(path)   # PIL cannot decode camera RAW files itself
    with open(long_path(path), "rb") as f:
        image = open_picture(f, path)
        if image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
        else:
            # Before decoding: a 24 MP JPEG then decodes at 1/2 – 1/8 of its size, about
            # twice as fast. The square box keeps it large enough also once EXIF turns it.
            side = DRAFT_MARGIN * max(size)
            image.draft("RGB", (side, side))
        image.load()
    return upright(image)


def _video_preview(path, timestamp):
    try:
        frame = read_frame_at(path, timestamp)
    except ImportError as error:
        raise VideoSupportMissing() from error
    if frame is None:
        raise ThumbnailUnavailable(path)
    return frame


def make_thumbnail(result, size):
    """A PIL image no larger than `size` for a SearchResult. Raises on failure."""
    if result.kind == MediaKind.VIDEO:
        image = _video_preview(result.path, result.timestamp)
    else:
        image = _image_preview(result.path, size)
    image.thumbnail(size)
    return image
