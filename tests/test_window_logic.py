"""What the window computes without showing anything: toolbar wrapping, justified rows of
results, clipboard payloads, the video player, the console output."""
import sys

import pytest
from PIL import Image

from support import COLOURS, save_picture
from vibekimus import output_log, settings
from vibekimus.clipboard import BITMAP_FILE_HEADER_BYTES, dib_payload, drop_files_payload, pasted_picture
from vibekimus.ui.flow import flow_layout
from vibekimus.ui.justified import MAX_ASPECT, clamp_aspect, justified_rows
from vibekimus.video_player import clock, find_player

DROPFILES_BYTES = 20
BITMAPINFOHEADER_BYTES = 40
ASPECTS = [1.5, 0.75, 1.0, 1.333, 2.0, 0.5, 1.0]


def test_toolbar_groups_wrap_to_a_new_row_when_the_window_is_too_narrow():
    positions, height = flow_layout([(100, 30), (100, 30), (100, 30)], available_width=250, gap_x=10, gap_y=5)
    assert positions == [(0, 0), (110, 0), (0, 35)] and height == 65


def test_a_group_wider_than_the_window_still_gets_placed():
    assert flow_layout([(400, 20)], available_width=100, gap_x=10, gap_y=5) == ([(0, 0)], 20)


@pytest.fixture
def justified():
    slots, total = justified_rows(ASPECTS, available_width=1000, row_height=200, gap=4)
    rows = {}
    for slot in slots:
        rows.setdefault(slot.y, []).append(slot)
    return slots, total, rows


def test_every_full_row_fills_the_width_close_to_the_height_asked_and_nothing_overlaps(justified):
    slots, total, rows = justified
    full_rows = list(rows.values())[:-1]
    assert full_rows and all(row[-1].x + row[-1].width == 1000 and row[0].x == 0 for row in full_rows)
    assert all(150 <= slot.height <= 270 for slot in slots)
    assert all(b.x >= a.x + a.width for row in rows.values() for a, b in zip(row, row[1:]))
    assert total == max(slot.y + slot.height for slot in slots)


def test_each_picture_keeps_its_proportions_within_a_pixel_of_rounding(justified):
    slots, _total, rows = justified
    last_row = list(rows)[-1]
    assert all(abs(slot.width / slot.height - aspect) < 2 / slot.height + 0.02
               for slot, aspect in zip(slots, ASPECTS) if slot.y != last_row)


def test_unknown_sizes_get_a_square_slot_and_panoramas_a_limited_one():
    assert clamp_aspect(-1, -1) == 1.0 and clamp_aspect(8000, 1000) == MAX_ASPECT
    assert clamp_aspect(1920, 1080) == 1920 / 1080


def test_files_for_the_clipboard_dropfiles_header_wide_paths_double_terminator():
    payload = drop_files_payload([r"C:\a b.jpg", r"D:\ż.png"])
    assert payload[:4] == DROPFILES_BYTES.to_bytes(4, "little")
    assert payload[DROPFILES_BYTES:].decode("utf-16-le") == "C:\\a b.jpg\0D:\\ż.png\0\0"


def test_pictures_for_the_clipboard_a_device_independent_bitmap():
    dib = dib_payload(Image.new("RGBA", (3, 2), (10, 20, 30, 0)))
    assert int.from_bytes(dib[:4], "little") == BITMAPINFOHEADER_BYTES and int.from_bytes(dib[4:8], "little") == 3
    assert BITMAP_FILE_HEADER_BYTES == 14


def test_a_pasted_picture_copied_pixels_or_the_first_picture_among_copied_files(tmp_path):
    picture = save_picture(str(tmp_path / "beach.jpg"), COLOURS["blue"])
    text = tmp_path / "notes.txt"
    text.write_text("")
    copied = pasted_picture(grab=lambda: Image.new("RGBA", (5, 4), (200, 10, 10, 255))).image
    from_files = pasted_picture(grab=lambda: [str(text), picture])
    assert copied.mode == "RGB" and copied.size == (5, 4)
    assert from_files.path == picture and from_files.image is None, "a file is read later, by path"


def test_nothing_is_pasted_from_text_other_files_or_a_busy_clipboard(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("")

    def busy():
        raise OSError("clipboard busy")
    assert pasted_picture(grab=lambda: None) is None and pasted_picture(grab=lambda: [str(text)]) is None
    assert pasted_picture(grab=busy) is None


def test_the_player_chosen_in_settings_starts_at_the_moment(tmp_path, monkeypatch):
    vlc = tmp_path / "players" / "vlc.exe"
    vlc.parent.mkdir()
    vlc.write_bytes(b"")
    monkeypatch.setattr(settings, "VIDEO_PLAYER", str(vlc))
    spec, program = find_player(".mp4")
    assert program == str(vlc) and spec.arguments("v.mp4", 75.5) == ["--start-time=75.500", "v.mp4"]
    assert clock(3725.5) == "01:02:05.500"


def test_printed_text_is_kept_line_by_line_and_the_unfinished_line_apart():
    log = output_log.OutputLog(max_lines=3)
    log.write("first line\nsecond")
    log.write(" half\r\nthird\n")
    assert log.since(0) == (["first line", "second half", "third"], "", 3)


def test_a_carriage_return_rewrites_the_line_like_a_progress_bar():
    log = output_log.OutputLog(max_lines=3)
    log.write("Downloading  10%\rDownloading  55%")
    _lines, partial, seen = log.since(0)
    log.write("\rDownloading 100%\n")
    assert partial == "Downloading  55%" and log.since(seen)[:2] == (["Downloading 100%"], "")


def test_only_the_last_lines_are_kept_and_the_line_numbers_go_on_counting():
    log = output_log.OutputLog(max_lines=3)
    log.write("one\ntwo\nthree\nfour\nfive\nsix\n")
    lines, _partial, total = log.since(0)
    assert lines == ["four", "five", "six"] and total == 6 and log.since(total - 1)[0] == ["six"]


def test_stdout_and_stderr_are_copied_to_the_console_window_once(monkeypatch):
    monkeypatch.setattr(sys, "stdout", sys.stdout)   # put back afterwards
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    output_log.capture()
    output_log.capture()   # a second call (launcher, then the program) must not copy twice
    _lines, _partial, before = output_log.OUTPUT.since(0)
    print("seen in the console window")
    sys.stderr.write("an error too\n")
    captured = output_log.OUTPUT.since(before)[0]
    wrapped_once = not isinstance(sys.stdout._original, output_log._Tee)
    assert captured == ["seen in the console window", "an error too"] and wrapped_once
