"""GPU JPEG decoding against the CPU path, and duplicate pairs on the GPU. Needs an NVIDIA GPU;
skipped otherwise."""
import io

import numpy as np
import pytest
from PIL import Image, ImageOps

from vibekimus.duplicates import similar_pairs

torch = pytest.importorskip("torch")
transforms = pytest.importorskip("torchvision.transforms")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")

MODEL_SIZE = 384
EXIF_ORIENTATION_TAG = 0x0112
# Mean absolute difference (0-255 scale) allowed between CPU and GPU pixels: JPEG
# decoders and resamplers differ slightly in rounding, never in content.
MAX_MEAN_PIXEL_DIFFERENCE = 3.0


def asymmetric_photo(orientation):
    """A JPEG whose four corners differ, tagged with the given EXIF orientation."""
    image = Image.new("RGB", (640, 400), (240, 240, 240))
    pixels = image.load()
    for x in range(0, 200):
        for y in range(0, 120):
            pixels[x, y] = (230, 20, 20)            # red block top-left
    for x in range(520, 640):
        for y in range(300, 400):
            pixels[x, y] = (20, 20, 230)            # blue block bottom-right
    exif = image.getexif()
    exif[EXIF_ORIENTATION_TAG] = orientation
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, exif=exif)
    return buffer.getvalue()


def cpu_reference(blob, geometry):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(blob)).convert("RGB"))
    for step in geometry:
        image = step(image)
    return np.asarray(image, dtype=np.float32)


def as_hwc(tensor):
    return tensor.permute(1, 2, 0).float().cpu().numpy()


@pytest.fixture(scope="module")
def geometry():
    return [transforms.Resize((MODEL_SIZE, MODEL_SIZE), interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True)]


@pytest.fixture(scope="module")
def decoder(geometry):
    from vibekimus.encoders.gpu_jpeg import GpuJpegDecoder
    return GpuJpegDecoder(torch.device("cuda"), geometry)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_the_exif_orientation_is_applied_like_pil_does(orientation):
    from vibekimus.encoders.gpu_jpeg import GpuJpegDecoder
    blob = asymmetric_photo(orientation)
    want = cpu_reference(blob, [])
    got = GpuJpegDecoder(torch.device("cuda"), []).decode([blob])[0]   # no geometry: the exact output size
    assert got is not None and as_hwc(got).shape == want.shape
    assert float(np.abs(as_hwc(got) - want).mean()) < MAX_MEAN_PIXEL_DIFFERENCE


def test_a_batch_comes_out_at_the_model_size_close_to_the_cpu_path(decoder, geometry):
    blobs = [asymmetric_photo(orientation) for orientation in (1, 6)]
    tensors = decoder.decode(blobs)
    assert all(tuple(t.shape) == (3, MODEL_SIZE, MODEL_SIZE) and t.dtype == torch.uint8 for t in tensors)
    differences = [float(np.abs(as_hwc(t) - cpu_reference(b, geometry)).mean()) for t, b in zip(tensors, blobs)]
    assert max(differences) < MAX_MEAN_PIXEL_DIFFERENCE, differences


def test_files_the_gpu_cannot_decode_come_back_as_none_and_the_rest_still_decode(decoder):
    cmyk = io.BytesIO()
    Image.new("CMYK", (300, 200), (0, 80, 80, 0)).save(cmyk, format="JPEG")
    mixed = decoder.decode([asymmetric_photo(1), cmyk.getvalue(), b"not a jpeg"])
    assert mixed[0] is not None and mixed[1] is None and mixed[2] is None


def test_duplicate_pairs_found_on_the_gpu_equal_the_cpu_result():
    rng = np.random.default_rng(5)
    vectors = rng.standard_normal((5000, 64)).astype(np.float32)
    for copy, original in ((4999, 7), (2500, 1234), (100, 4321)):
        vectors[copy] = vectors[original] + 0.01 * rng.standard_normal(64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    on_gpu = {tuple(pair) for pair in similar_pairs(vectors, 0.95, device=torch.device("cuda")).tolist()}
    on_cpu = {tuple(pair) for pair in similar_pairs(vectors, 0.95).tolist()}
    assert on_gpu == on_cpu and len(on_cpu) == 3
