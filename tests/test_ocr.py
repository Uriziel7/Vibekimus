"""Reading text on pictures with the real OCR model (PP-OCRv6 small through RapidOCR).

Skipped when rapidocr or the model is not installed. The network is blocked for these
tests: the model must load from the local cache only."""
import os
import socket
import threading
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from vibekimus import settings
from vibekimus.text_matching import normalize

STOP_AFTER_SECONDS = 3
POOL_PROCESSES, POOL_ENGINES = 2, 2
RESULT_WAIT_SECONDS = 60     # a worker process starts and loads its models first
STOPPED_WITHIN_SECONDS = 5
RETIRE_PROCESSES, RETIRE_ROUNDS = 3, 20
ENGINES_AT_ONCE = 4


def _no_network(*args, **kwargs):
    raise RuntimeError("network access attempted")


@pytest.fixture(scope="module", autouse=True)
def offline():
    """No network while these tests run; skipped without the OCR model."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", _no_network)
        patch.setattr(socket, "create_connection", _no_network)
        from vibekimus.text_reading import TextReaderUnavailable, check_available
        try:
            check_available()
        except TextReaderUnavailable as error:
            pytest.skip(str(error))
        yield


def font(size, bold=False):
    return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)


def polish_sign():
    image = Image.new("RGB", (900, 300), (250, 250, 245))
    draw = ImageDraw.Draw(image)
    draw.text((40, 60), "Zażółć gęślą jaźń", fill=(20, 20, 20), font=font(64))
    draw.text((40, 170), "Darmowe piwo w piątek o 18:30", fill=(150, 20, 20), font=font(44))
    return image


def meme():
    noise = (np.random.default_rng(1).random((500, 700, 3)) * 255).astype(np.uint8)
    image = Image.fromarray(noise).filter(ImageFilter.GaussianBlur(6))
    draw = ImageDraw.Draw(image)
    for y, text in ((30, "WHEN THE CODE WORKS"), (400, "ON THE FIRST TRY")):
        draw.text((40, y), text, fill="white", font=font(54, bold=True), stroke_width=4, stroke_fill="black")
    return image


def screenshot():
    image = Image.new("RGB", (1280, 720), (32, 34, 37))
    draw = ImageDraw.Draw(image)
    for row, line in enumerate(["Kasia: kod rabatowy to ZIMA2026", "Order #A7731 shipped via DHL Express"]):
        draw.text((30, 40 + row * 40), line, fill=(220, 221, 222), font=font(18))
    return image


@pytest.fixture(scope="module")
def reader():
    from vibekimus.text_reading import TextReader
    return TextReader(threads=os.cpu_count() or 1)


def test_polish_text_with_every_diacritic(reader):
    sign = normalize(reader.read(polish_sign()))
    assert " zazolc gesla jazn " in sign and " piatek " in sign and "18 30" in sign, sign


def test_meme_text_on_a_busy_background_white_with_a_black_outline(reader):
    assert " when the code works " in normalize(reader.read(meme()))


def test_small_screenshot_text_with_codes_and_numbers(reader):
    shot = normalize(reader.read(screenshot()))
    assert "zima2026" in shot and "a7731" in shot, shot


def test_a_picture_without_text_gives_no_text(reader):
    assert reader.read(Image.new("RGB", (800, 600), (200, 220, 240))) == ""


@pytest.fixture(scope="module")
def on_the_card(reader):
    """Engines on the graphics card (DirectML) and the text the CPU reads from the same pictures."""
    from vibekimus.text_reading import GPU_NAME, TextReader, reading_device
    if reading_device() != GPU_NAME:
        pytest.skip("no DirectML here")
    pictures = [polish_sign(), meme(), screenshot()] * 4
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(settings, "OCR_ON_GPU", False)
        on_cpu = TextReader(threads=os.cpu_count() or 1)
        expected = [on_cpu.read(picture) for picture in pictures]
    readers = [TextReader(threads=2) for _ in range(ENGINES_AT_ONCE)]
    return readers, on_cpu, pictures, expected


def test_reading_happens_on_the_graphics_card(reader, on_the_card):
    on_cpu = on_the_card[1]
    assert reader.on_gpu and not on_cpu.on_gpu


def test_four_engines_reading_at_once_on_the_card_give_the_same_text_as_the_cpu(on_the_card):
    readers, _on_cpu, pictures, expected = on_the_card
    found = [None] * len(pictures)

    def work(engine, indices):
        for index in indices:
            found[index] = engine.read(pictures[index])
    threads = [threading.Thread(target=work, args=(engine, range(n, len(pictures), len(readers))))
               for n, engine in enumerate(readers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert found == expected, [i for i, (a, b) in enumerate(zip(found, expected)) if a != b]


def test_the_engines_share_one_pair_of_models_on_the_card(on_the_card):
    readers = on_the_card[0]
    first = readers[0]._engine
    shared = [engine._engine.text_det.session.session is first.text_det.session.session
              and engine._engine.text_rec.session.session is first.text_rec.session.session for engine in readers]
    assert all(shared), "one pair per engine grew past 14 GB"


def test_stop_while_engines_read_on_the_card_closes_each_in_its_own_thread(reader, on_the_card):
    """As the text pass stops: each worker closes its engine after its current picture while the
    others still read (freeing a model during another's run crashed the program)."""
    from vibekimus.text_reading import _shared_gpu_models
    readers, pictures = on_the_card[0] + [reader], on_the_card[2]
    stop = threading.Event()
    rounds = [0] * len(readers)

    def work(number, engine):
        try:
            while not stop.is_set():
                engine.read(pictures[rounds[number] % len(pictures)])
                rounds[number] += 1
        finally:
            engine.close()
    threads = [threading.Thread(target=work, args=pair) for pair in enumerate(readers)]
    for thread in threads:
        thread.start()
    stop.wait(STOP_AFTER_SECONDS)
    stop.set()
    for thread in threads:
        thread.join()
    assert _shared_gpu_models._parts is None and all(rounds), rounds


@pytest.fixture(scope="module")
def pictures_on_disk(tmp_path_factory):
    """Three pictures in a folder and the text this process reads on each."""
    from vibekimus.text_reading import TextReader
    folder = str(tmp_path_factory.mktemp("ocr"))
    pictures = {"sign.png": polish_sign(), "meme.png": meme(), "shot.png": screenshot()}
    for name, picture in pictures.items():
        picture.save(os.path.join(folder, name))
    here = TextReader(threads=os.cpu_count() or 1)
    expected = {name: here.read(Image.open(os.path.join(folder, name))) for name in pictures}
    here.close()
    return folder, list(pictures), expected


def jobs(folder, names, times):
    return [(name, os.path.join(folder, name)) for name in names] * times


@pytest.fixture
def new_computer(tmp_path, monkeypatch):
    """The program's data in an empty folder, for this process and the worker processes it starts
    (they read it from the environment); the models stay where they are."""
    from huggingface_hub.constants import HF_HOME
    data = str(tmp_path / "data")
    monkeypatch.setenv("VIBEKIMUS_DATA_DIR", data)
    monkeypatch.setenv("HF_HOME", HF_HOME)
    monkeypatch.setattr(settings, "MODELS_DIR", os.path.join(data, "models"))


def test_worker_processes_on_a_new_computer_read_the_same_text(pictures_on_disk, new_computer):
    """No character list yet: the processes must not all write it at once."""
    from vibekimus.indexing.text_reading_pool import ProcessReadingPool
    from vibekimus.text_reading import check_available
    folder, names, expected = pictures_on_disk
    check_available()   # as the text pass does before its processes start
    pool = ProcessReadingPool(POOL_PROCESSES, POOL_ENGINES, threads=2)
    pool.start(jobs(folder, names, 2))
    found = [pool.next_result(RESULT_WAIT_SECONDS) for _ in range(len(names) * 2)]
    pool.stop()
    assert all(item is not None and item[2] is None and item[1] == expected[item[0]] for item in found), found


def test_stop_ends_the_worker_processes_at_once_in_the_middle_of_reading(pictures_on_disk):
    from vibekimus.indexing.text_reading_pool import ProcessReadingPool
    folder, names, _expected = pictures_on_disk
    busy = ProcessReadingPool(POOL_PROCESSES, POOL_ENGINES, threads=2)
    busy.start(jobs(folder, names, 50))
    first = busy.next_result(RESULT_WAIT_SECONDS)
    busy.stop()
    deadline = time.time() + STOPPED_WITHIN_SECONDS
    while not busy.finished() and time.time() < deadline:
        busy.next_result(0.1)   # what was already sent is taken; the processes are gone
    assert first is not None and busy.finished(), first


def test_a_retired_process_finishes_its_pictures_one_added_joins_and_nothing_is_lost(pictures_on_disk):
    """The graphics card running out of memory: one process finishes its pictures and ends, the
    others read the rest; then memory free again: one more joins."""
    from vibekimus.indexing.text_reading_pool import ProcessReadingPool
    folder, names, expected = pictures_on_disk
    work = jobs(folder, names, RETIRE_ROUNDS)
    pool = ProcessReadingPool(RETIRE_PROCESSES, POOL_ENGINES, threads=2)
    pool.start(work)
    found = [pool.next_result(RESULT_WAIT_SECONDS)]
    retired, again = pool.retire_one(), pool.retire_one()
    while len(found) < len(work) // 2 and found[-1] is not None:
        found.append(pool.next_result(RESULT_WAIT_SECONDS))
    left = pool.running()
    pool.add_one()
    grown = pool.running()
    while len(found) < len(work) and found[-1] is not None:
        found.append(pool.next_result(RESULT_WAIT_SECONDS))
    pool.stop()
    assert retired and not again, "only one finishes at a time"
    assert left == RETIRE_PROCESSES - 1 and grown == RETIRE_PROCESSES
    assert len(found) == len(work) and all(item is not None and item[1] == expected[item[0]] for item in found)
