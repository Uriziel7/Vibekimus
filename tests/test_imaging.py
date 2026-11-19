"""Opening pictures: reduced-size decoding, EXIF rotation, extensions that do not match the
content, formats that would run external programs, thumbnails."""
import os

from PIL import Image

from support import COLOURS, fails, save_picture
from vibekimus.imaging import open_image, open_picture
from vibekimus.media_kind import MediaKind
from vibekimus.search import SearchResult
from vibekimus.thumbnails import make_thumbnail

EXIF_ORIENTATION = 0x0112
SIDEWAYS = 6   # "rotate 90 degrees to display"
EPS_CONTENT = b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nshowpage\n"


def test_draft_decode_shrinks_a_big_jpeg_but_not_below_the_model_size(tmp_path):
    path = save_picture(str(tmp_path / "red.jpg"), COLOURS["red"], (3000, 2000), quality=90)
    full, small = open_image(path), open_image(path, draft_size=384)
    assert full.size == (3000, 2000)
    assert small.size[0] < 3000 and min(small.size) >= 384, small.size


def test_draft_decode_leaves_a_png_untouched(tmp_path):
    path = save_picture(str(tmp_path / "dark.png"), (20, 20, 20), (640, 480))
    assert open_image(path, draft_size=384).size == (640, 480)


def test_a_large_png_is_reduced_early_but_stays_at_least_the_model_size(tmp_path):
    path = str(tmp_path / "screenshot.png")
    Image.new("RGBA", (3000, 2000), (10, 200, 10, 255)).save(path)
    assert open_image(path, draft_size=384).size == (600, 400)   # 2000 // 384 = 5: reduced 5x


def test_a_photo_stored_sideways_is_turned_upright(tmp_path):
    path = str(tmp_path / "sideways.jpg")
    image = Image.new("RGB", (400, 200), (10, 120, 200))
    exif = image.getexif()
    exif[EXIF_ORIENTATION] = SIDEWAYS
    image.save(path, exif=exif)
    assert open_image(path).size == (200, 400)


def test_a_picture_whose_extension_lies_about_its_format_still_opens(tmp_path):
    folder = tmp_path / "misnamed"
    for name, kind in (("png_as.jpg", "PNG"), ("webp_as.jpg", "WEBP"), ("bmp_as.jpg", "BMP"), ("jpeg_as.png", "JPEG")):
        save_picture(str(folder / name), (200, 40, 40), (500, 400), format=kind)
    opened = {name: open_image(str(folder / name), draft_size=384) for name in sorted(os.listdir(folder))}
    assert all(image is not None and image.size == (500, 400) for image in opened.values()), opened


def test_content_sniffing_does_not_reach_formats_that_run_programs(tmp_path):
    eps = tmp_path / "eps_as.jpg"
    eps.write_bytes(EPS_CONTENT)
    assert open_image(str(eps)) is None


def test_an_eps_named_jpg_is_not_opened_by_any_reader_pil_has(tmp_path):
    eps = tmp_path / "drawing.jpg"
    eps.write_bytes(EPS_CONTENT)
    assert fails(lambda: open_picture(str(eps), str(eps))) is not None
    assert fails(lambda: make_thumbnail(SearchResult(0.0, str(eps), MediaKind.IMAGE), (100, 100))) is not None


def test_a_thumbnail_decoded_at_reduced_size_is_upright_and_full_size(tmp_path):
    path = str(tmp_path / "sideways.jpg")
    exif = Image.Exif()
    exif[EXIF_ORIENTATION] = SIDEWAYS
    Image.new("RGB", (3000, 2000), COLOURS["red"]).save(path, exif=exif)
    assert make_thumbnail(SearchResult(0.0, path, MediaKind.IMAGE), (200, 200)).size == (133, 200)
