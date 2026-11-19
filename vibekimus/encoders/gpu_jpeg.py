"""Decoding JPEG files on an NVIDIA GPU (nvJPEG through torchvision)."""
import io
from enum import IntEnum

from PIL import Image

EXIF_ORIENTATION_TAG = 0x0112
JPEG_ONLY = ("JPEG",)   # only JPEGs are handed to the GPU decoder
HEIGHT_AXIS, WIDTH_AXIS = 1, 2   # of a CHW tensor
CLOCKWISE, COUNTER_CLOCKWISE = -1, 1   # torch.rot90 quarter turns


class Orientation(IntEnum):
    """The EXIF orientation tag: how the stored picture must be turned to stand upright."""
    UPRIGHT = 1
    MIRRORED = 2
    UPSIDE_DOWN = 3
    MIRRORED_UPSIDE_DOWN = 4
    TRANSPOSED = 5       # mirrored along the top-left to bottom-right diagonal
    TURNED_LEFT = 6      # turned 90° clockwise to stand upright
    TRANSVERSE = 7       # mirrored along the other diagonal
    TURNED_RIGHT = 8     # turned 90° counter-clockwise to stand upright


def exif_orientation(data):
    """The EXIF orientation of a JPEG, read from its header only (no decoding)."""
    try:
        tag = Image.open(io.BytesIO(data), formats=JPEG_ONLY).getexif().get(EXIF_ORIENTATION_TAG)
        return Orientation(tag)
    except Exception:   # no tag, an unknown value, or a header PIL cannot read
        return Orientation.UPRIGHT


def apply_orientation(image, orientation):
    """Turn a CHW tensor upright like PIL's ImageOps.exif_transpose. nvJPEG does
    not apply the EXIF tag itself, so phone photos would otherwise lie on their side."""
    import torch
    axes = (HEIGHT_AXIS, WIDTH_AXIS)
    if orientation == Orientation.MIRRORED:
        return image.flip(WIDTH_AXIS)
    if orientation == Orientation.UPSIDE_DOWN:
        return image.flip(axes)
    if orientation == Orientation.MIRRORED_UPSIDE_DOWN:
        return image.flip(HEIGHT_AXIS)
    if orientation == Orientation.TRANSPOSED:
        return image.transpose(*axes)
    if orientation == Orientation.TURNED_LEFT:
        return torch.rot90(image, CLOCKWISE, axes)
    if orientation == Orientation.TRANSVERSE:
        return image.transpose(*axes).flip(axes)
    if orientation == Orientation.TURNED_RIGHT:
        return torch.rot90(image, COUNTER_CLOCKWISE, axes)
    return image


class GpuJpegDecoder:
    """Decodes a batch of JPEG files on the GPU and applies the model's geometry
    (resize / crop) there, producing the same uint8 tensors the CPU path makes."""

    def __init__(self, device, tensor_geometry):
        self._device = device
        self._geometry = tensor_geometry

    def decode(self, blobs):
        """blobs: file contents. Returns one tensor per blob, or None where the GPU
        could not decode it (e.g. CMYK JPEGs) — those must go to the CPU."""
        import torch
        from torchvision.io import ImageReadMode, decode_jpeg
        data = [torch.frombuffer(bytearray(blob), dtype=torch.uint8) for blob in blobs]
        try:
            images = decode_jpeg(data, mode=ImageReadMode.RGB, device=self._device)
        except Exception:
            images = [self._decode_one(decode_jpeg, ImageReadMode.RGB, item) for item in data]
        return [None if image is None else self._finish(image, blob) for image, blob in zip(images, blobs)]

    def _decode_one(self, decode_jpeg, mode, item):
        try:
            return decode_jpeg(item, mode=mode, device=self._device)
        except Exception:
            return None

    def _finish(self, image, blob):
        image = apply_orientation(image, exif_orientation(blob))
        for step in self._geometry:
            image = step(image)
        return image.contiguous()
