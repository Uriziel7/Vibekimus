"""Fixtures of every test: the program's own files kept out of the user's data, the stand-in
encoder, and libraries of test pictures (made once, for the tests that only read them)."""
import os
import time

import pytest
from PIL import Image

from support import COLOURS, FakeModel, open_library, rebuild, run, save_picture, write_clip
from vibekimus import hardware, settings
from vibekimus.indexing.controller import IMAGES, VIDEOS

# The program's files (settings name -> file or folder name), each test with its own.
PROGRAM_FILES = {"LOG_FILE": "test_run_log.txt", "PREFERENCES_FILE": "prefs.json",
                 "ALL_DRIVES_INDEX_DIR": "index_all_drives", "INDEXES_DIR": "indexes",
                 "KNOWN_INDEXES_FILE": "known_indexes.json", "LAST_INDEX_FILE": "last_index.json",
                 "KEY_FILE": "key.json"}


def _keep_program_files_in(folder, patch):
    for name, file in PROGRAM_FILES.items():
        patch.setattr(settings, name, os.path.join(folder, file))


@pytest.fixture(scope="session", autouse=True)
def program_files_of_the_session(tmp_path_factory):
    """Also for the fixtures made once per session: nothing reaches the real log or data folder,
    and the graphics card is not asked (smallest batch sizes)."""
    with pytest.MonkeyPatch.context() as patch:
        _keep_program_files_in(str(tmp_path_factory.mktemp("session_app")), patch)
        patch.setattr(hardware, "system_vram_bytes", lambda: None)
        yield


@pytest.fixture(autouse=True)
def program_files(tmp_path, monkeypatch):
    """The program's data folder of this test (saved indexes, last index, preferences, key…)."""
    folder = str(tmp_path / "app")
    os.makedirs(folder)
    _keep_program_files_in(folder, monkeypatch)
    return folder


@pytest.fixture(scope="session")
def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def media_folder(tmp_path_factory):
    """Big JPEGs (reduced-size decoding), a PNG, a broken file and three 3 s videos."""
    folder = str(tmp_path_factory.mktemp("library"))
    for name, rgb in COLOURS.items():
        save_picture(os.path.join(folder, "sub dir", f"{name}.jpg"), rgb, (3000, 2000), quality=90)
    save_picture(os.path.join(folder, "dark.png"), (20, 20, 20), (640, 480))
    with open(os.path.join(folder, "broken.jpg"), "wb") as f:
        f.write(b"not an image at all")
    for name in ("purple", "green", "yellow"):
        write_clip(os.path.join(folder, f"clip_{name}.mp4"), COLOURS[name], size=(1280, 720))
    return folder


@pytest.fixture(scope="session")
def media_encoder():
    """The encoder that indexed media_library (it remembers the picture sizes it was given)."""
    return FakeModel()


@pytest.fixture(scope="session")
def media_library(media_folder, media_encoder):
    """media_folder with its pictures and videos indexed. Only for tests that do not change it."""
    library = open_library(media_folder)
    run(library, rebuild(IMAGES), encoder=media_encoder)
    run(library, rebuild(VIDEOS), encoder=media_encoder)
    return library


def make_details(folder):
    """Pictures with known details: an EXIF date, sizes, a folder, an old modification date."""
    dated = Image.new("RGB", (1200, 800), COLOURS["blue"])
    exif = dated.getexif()
    exif.get_ifd(0x8769)[36867] = "2019:07:14 12:30:00"
    os.makedirs(os.path.join(folder, "trip"))
    dated.save(os.path.join(folder, "trip", "beach.jpg"), exif=exif)
    save_picture(os.path.join(folder, "red_small.png"), COLOURS["red"], (400, 300))
    save_picture(os.path.join(folder, "red_large.png"), COLOURS["red"], (1600, 1200))
    green = save_picture(os.path.join(folder, "green.png"), COLOURS["green"], (640, 480))
    old_time = time.mktime((2020, 3, 1, 10, 0, 0, 0, 0, -1))
    os.utime(green, (old_time, old_time))
    return folder


@pytest.fixture
def details(tmp_path):
    """A folder of make_details() with its pictures indexed."""
    library = open_library(make_details(str(tmp_path / "details")))
    run(library, rebuild(IMAGES))
    return library


@pytest.fixture
def pairs(tmp_path):
    """Two pairs of duplicates (blue, red), each a large and a small copy; the large blue is old."""
    folder = str(tmp_path / "dupe_pairs")
    for name, colour, size in (("blue_a.png", "blue", (800, 600)), ("blue_b.png", "blue", (400, 300)),
                               ("red_x.png", "red", (1000, 800)), ("red_y.png", "red", (500, 400))):
        save_picture(os.path.join(folder, name), COLOURS[colour], size)
    old_blue = time.mktime((2015, 5, 1, 10, 0, 0, 0, 0, -1))
    os.utime(os.path.join(folder, "blue_a.png"), (old_blue, old_blue))
    library = open_library(folder)
    run(library, rebuild(IMAGES))
    return library


def make_mixed(folder, count=30):
    """Photos of every colour, every fourth a PNG, the rest JPEGs."""
    for number in range(count):
        colour = list(COLOURS.values())[number % len(COLOURS)]
        extension = ".png" if number % 4 == 0 else ".jpg"
        save_picture(os.path.join(folder, f"photo_{number:02d}{extension}"), colour, (640, 480))
    return folder


@pytest.fixture
def mixed(tmp_path):
    """A folder of make_mixed() with its pictures indexed (the reference: decoded on the CPU)."""
    library = open_library(make_mixed(str(tmp_path / "decode_modes")))
    run(library, rebuild(IMAGES))
    return library
