"""Opening image files of any supported kind as RGB PIL images."""
import io
import os
import warnings

from PIL import Image, ImageOps

from vibekimus import settings
from vibekimus.files import long_path
from vibekimus.runlog import safe_print
from vibekimus.storage import DRIVES

# Allow legitimately large images (scanned maps, panoramas) but keep a ceiling so a
# decompression bomb cannot exhaust RAM; open_image() then reports the file as skipped.
Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

EXTENSION_FORMATS = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG",
                     "webp": "WEBP", "bmp": "BMP", "gif": "GIF"}
# Image.reduce() works on these modes; anything else is converted to RGB(A) first.
REDUCIBLE_MODES = ("RGB", "RGBA", "L", "LA", "I", "F")
# Shrinking by a whole factor only pays off from 2x on.
MIN_REDUCE_FACTOR = 2
# Transparent images are flattened onto white — closer to the model's training data than black.
TRANSPARENCY_BACKGROUND = (255, 255, 255)


def is_raw(path):
    return path.lower().endswith(settings.RAW_EXTS)


def read_bytes(path):
    """The whole file; on a hard disk one file at a time (storage.DRIVES)."""
    with DRIVES.reading(path):
        with open(long_path(path), "rb") as f:
            return f.read()


def decode_raw(path):
    """Develop a camera RAW file. Raises ImportError when rawpy is not installed."""
    import rawpy
    with rawpy.imread(io.BytesIO(read_bytes(path))) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
    return Image.fromarray(rgb)


def open_picture(source, path):
    """A PIL image of a picture file, not decoded yet; source: the file's path or its bytes
    as a file object, path: its name. Only the format its extension names is tried, then
    the plain raster formats of settings.IMAGE_SNIFF_FORMATS — never any other reader PIL
    has (EPS, for one, would start Ghostscript on the file)."""
    extension = os.path.splitext(path)[1].lower().lstrip(".")
    expected = EXTENSION_FORMATS.get(extension)
    try:
        return Image.open(source, formats=[expected] if expected else settings.IMAGE_SNIFF_FORMATS)
    except Image.UnidentifiedImageError:
        # Extension lies about the content (very common for downloads: a PNG or
        # WEBP saved as .jpg). Let PIL recognise it from the bytes instead.
        if hasattr(source, "seek"):
            source.seek(0)
        return Image.open(source, formats=settings.IMAGE_SNIFF_FORMATS)


def _decode_standard(path, draft_size):
    img = open_picture(io.BytesIO(read_bytes(path)), path)
    if draft_size and img.format == "JPEG":
        # Decode directly at a reduced size: the model only sees a few hundred pixels,
        # so decoding a 24 MP photo at full size is wasted time.
        try:
            img.draft("RGB", (draft_size, draft_size))
        except Exception:
            pass
    img.load()
    img = upright(img)
    return _reduce_to(img, draft_size) if draft_size else img


def _reduce_to(img, min_side):
    """Shrink a large image by a whole factor (fast box filter) while both sides stay
    at least min_side. Converting and resizing a 3 MP screenshot costs more than
    decoding it; doing that work on a smaller image saves about a third of the time.
    JPEGs mostly arrive small already, thanks to draft()."""
    factor = min(img.size) // min_side
    if factor < MIN_REDUCE_FACTOR:
        return img
    if img.mode not in REDUCIBLE_MODES:
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    return img.reduce(factor)


def upright(img):
    """Apply the EXIF "rotate me" tag — phone photos are stored sideways."""
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img


def to_rgb(img):
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, TRANSPARENCY_BACKGROUND)
        background.paste(img, mask=img.getchannel("A"))
        return background
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def open_image(path, draft_size=None):
    """Open any supported image as RGB, or return None (and say why) if it cannot be read.

    draft_size: when set, the image may come back smaller than the file, but still at
    least draft_size on both sides (JPEGs are decoded at reduced size, large images
    of other formats are shrunk right after decoding).
    """
    name = os.path.basename(path)
    try:
        img = decode_raw(path) if is_raw(path) else _decode_standard(path, draft_size)
    except ImportError:
        safe_print(f"[RAW] rawpy not installed, skipping {name}. Install with: pip install rawpy")
        return None
    except MemoryError:
        safe_print(f"[IMAGE] Skipping {name}: image too large for available RAM")
        return None
    except Exception as e:
        safe_print(f"[IMAGE] Failed to open {name}: {e}")
        return None
    return to_rgb(img)
