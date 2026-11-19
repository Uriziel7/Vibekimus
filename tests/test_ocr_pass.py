"""The text pass (OCR) with a stand-in reader that "reads" a text chosen by the picture's
colour: what is read, how many readers run, searching the text, the text file."""
import os
import threading
import time

import numpy as np
import pytest

from support import (EVERYTHING, GB, FakeModel, FakeTextReader, fails, in_order, names, open_library, rebuild,
                     refresh, run)
from vibekimus import settings, text_reading
from vibekimus.embedding_index import EmbeddingIndex
from vibekimus.encoders.catalog import SIGLIP2_SO400M
from vibekimus.index_files import ImageIndexFormat
from vibekimus.indexing.base import PassState
from vibekimus.indexing.controller import TEXT
from vibekimus.indexing.text_pass import TextIndexer
from vibekimus.indexing.text_reading_pool import ThreadReadingPool
from vibekimus.media_info import unknown_info
from vibekimus.ocr_capacity import OcrProcessBudget
from vibekimus.search import SearchEngine, SearchOptions
from vibekimus.text_index import TextIndex
from vibekimus.text_matching import TextMatch, normalize
from vibekimus.text_reading import CHARACTER_FILE, TextReaderUnavailable, write_character_file

PICTURES = 4
WITH_TEXT = 3            # green has none
ENGINES_CLOSE_WITHIN = 5
CPU_THREADS = 12
CHARACTERS = 18708       # as long as the real list
RACE_ATTEMPTS, RACING_ENGINES = 30, 8


@pytest.fixture
def stand_in_ocr(monkeypatch):
    """The text pass with stand-in readers in threads; returns the (processes, engines, threads) asked for."""
    asked = []

    def stand_in_pool(processes, engines, threads):
        asked.append((processes, engines, threads))
        return ThreadReadingPool(FakeTextReader, engines, threads)
    monkeypatch.setattr(TextIndexer, "make_pool", staticmethod(stand_in_pool))
    monkeypatch.setattr(TextIndexer, "check_reader", staticmethod(lambda: None))
    monkeypatch.setattr(TextIndexer, "make_budget", staticmethod(lambda on_gpu: OcrProcessBudget(False)))
    monkeypatch.setattr(FakeTextReader, "made", 0)
    monkeypatch.setattr(FakeTextReader, "closed", 0)
    return asked


@pytest.fixture
def read(details, stand_in_ocr):
    """details with the text on its pictures read."""
    run(details, refresh(TEXT))
    return details


def test_text_is_compared_without_case_diacritics_or_punctuation():
    assert normalize("Zażółć GĘŚLĄ, jaźń!") == " zazolc gesla jazn " and normalize("ŁÓDŹ") == " lodz "


def test_the_text_pass_reads_every_picture_closes_its_engines_and_reads_nothing_again(details, stand_in_ocr):
    outcome = run(details, refresh(TEXT))
    texts = details.texts
    assert outcome[-1][1] is PassState.FINISHED
    assert len(texts) == PICTURES and texts.with_text() == WITH_TEXT, "empty text where there is none"
    assert texts.text_of("trip/beach.jpg") == "Zażółć gęślą jaźń"
    deadline = time.time() + ENGINES_CLOSE_WITHIN
    while FakeTextReader.closed < FakeTextReader.made and time.time() < deadline:
        time.sleep(0.01)
    assert 0 < FakeTextReader.made == FakeTextReader.closed, "each engine closed (its graphics memory goes back)"
    assert run(details, refresh(TEXT))[-1][1] is PassState.UP_TO_DATE


def test_the_text_pass_starts_the_processes_the_budget_allows(details, stand_in_ocr, monkeypatch):
    monkeypatch.setattr(TextIndexer, "make_budget", staticmethod(
        lambda on_gpu: OcrProcessBudget(True, free_vram=lambda: 30 * GB, cpu_threads=CPU_THREADS)))
    outcome = run(details, refresh(TEXT))
    processes, _engines, threads = stand_in_ocr[-1]
    assert outcome[-1][1] is PassState.FINISHED and processes == 4
    assert threads == max(1, CPU_THREADS // (4 * settings.OCR_WORKERS)), "CPU threads shared by the most that may run"


class SqueezedPool(ThreadReadingPool):
    """Stand-in engines counted as processes; every retire_one() lets one go."""

    def __init__(self, processes, engines, threads):
        super().__init__(FakeTextReader, engines, threads)
        self.started_with = self.left = processes
        self.history = []

    def running(self):
        return self.left

    def retire_one(self):
        if self.left <= 1:
            return False
        self.left -= 1
        self.history.append(self.left)
        return True

    def add_one(self):
        self.left += 1
        self.history.append(self.left)


def test_reading_follows_the_free_graphics_memory_and_still_reads_every_picture(details, stand_in_ocr, monkeypatch):
    pools = []
    monkeypatch.setattr(TextIndexer, "make_pool", staticmethod(lambda *sizes: pools.append(SqueezedPool(*sizes))
                                                               or pools[-1]))
    free_now = iter([9.5 * GB, 0.5 * GB] + [30 * GB] * 10000)   # another program takes the memory, then gives it back
    ticks = iter(range(0, 10 ** 9, int(settings.OCR_GROW_WAIT_SECONDS)))   # every check due, every wait over
    monkeypatch.setattr(TextIndexer, "make_budget", staticmethod(lambda on_gpu: OcrProcessBudget(
        True, free_vram=lambda: next(free_now), clock=lambda: next(ticks), cpu_threads=CPU_THREADS)))
    outcome = run(details, refresh(TEXT))
    assert outcome[-1][1] is PassState.FINISHED and len(details.texts) == PICTURES
    assert pools[-1].started_with == 2 and pools[-1].history[:2] == [1, 2], "one let go, then one added back"


def test_without_the_ocr_model_the_pass_fails_and_keeps_the_text_read_before(read, monkeypatch):
    def no_model():
        raise TextReaderUnavailable("no model")
    monkeypatch.setattr(TextIndexer, "check_reader", staticmethod(no_model))
    outcome = run(read, rebuild(TEXT))
    assert outcome[-1][1] is PassState.FAILED and len(read.texts) == PICTURES


def test_the_text_is_saved_next_to_the_index_and_loads_again(read):
    reopened = open_library(read.scope.data_dir)
    reopened.load_texts()
    assert reopened.texts.with_text() == WITH_TEXT
    assert reopened.texts.text_of("red_small.png") == "Darmowe piwo\nw piątek"


def test_a_quoted_phrase_must_be_on_the_picture(read):
    found = SearchEngine(read).by_text(FakeModel(), 'blue "PIATEK"', SearchOptions(min_score=EVERYTHING))
    assert names(found) == ["red_large.png", "red_small.png"], "any case, no diacritics needed"


def test_loose_words_seen_on_a_picture_raise_it_in_the_ranking(read, monkeypatch):
    engine = SearchEngine(read)
    monkeypatch.setattr(settings, "OCR_WORD_BONUS", 0.0)
    plain = engine.by_text(FakeModel(), "green piwo", SearchOptions(min_score=EVERYTHING))
    monkeypatch.setattr(settings, "OCR_WORD_BONUS", 10.0)
    boosted = engine.by_text(FakeModel(), "green piwo", SearchOptions(min_score=EVERYTHING))
    assert in_order(plain)[0] == "green.png" and in_order(boosted)[0].startswith("red_")


def test_a_damaged_text_file_is_not_trusted(read):
    damaged = open_library(read.scope.data_dir)
    with open(damaged.texts.path, "wb") as f:
        f.write(b"not an archive")
    damaged.load_texts()
    assert not damaged.texts.has_data


def test_a_text_file_naming_a_path_outside_the_folder_is_refused(read):
    read.texts.set("../outside.png", "x")
    read.texts.save()
    unsafe = open_library(read.scope.data_dir)
    unsafe.load_texts()
    assert not unsafe.texts.has_data


def test_engines_starting_together_on_a_new_computer_write_the_character_list_once(tmp_path, monkeypatch):
    characters = [chr(code) for code in range(0x4E00, 0x4E00 + CHARACTERS)]
    errors = []

    def start_engine():
        try:
            write_character_file(characters)
        except Exception as error:
            errors.append(error)
    for attempt in range(RACE_ATTEMPTS):   # a fresh computer each time: no file yet, every engine starts at once
        monkeypatch.setattr(settings, "MODELS_DIR", str(tmp_path / str(attempt)))
        engines = [threading.Thread(target=start_engine) for _ in range(RACING_ENGINES)]
        for engine in engines:
            engine.start()
        for engine in engines:
            engine.join()
    assert not errors, "no 'access denied'"
    with open(os.path.join(settings.MODELS_DIR, CHARACTER_FILE), encoding="utf-8") as f:
        assert f.read().split("\n") == characters


def test_with_ocr_on_gpu_off_text_is_read_on_the_cpu(monkeypatch):
    monkeypatch.setattr(settings, "OCR_ON_GPU", False)
    assert text_reading.reading_device() == text_reading.CPU_NAME


class SlowSession:
    """Stands in for an ONNX Runtime session on the graphics card; counts runs at the same time."""

    def __init__(self, counter):
        self._counter = counter

    def run(self, *args):
        with self._counter["lock"]:
            self._counter["running"] += 1
            self._counter["overlaps"] += self._counter["running"] > 1
        time.sleep(0.01)
        with self._counter["lock"]:
            self._counter["running"] -= 1
        return args

    def get_inputs(self):
        return ["image"]


def test_the_engines_runs_on_the_graphics_card_take_turns():
    counter = {"lock": threading.Lock(), "running": 0, "overlaps": 0}
    sessions = [text_reading._TakingTurns(SlowSession(counter)) for _ in range(4)]
    runners = [threading.Thread(target=lambda s=s: [s.run(n) for n in range(10)]) for s in sessions]
    for runner in runners:
        runner.start()
    for runner in runners:
        runner.join()
    assert counter["overlaps"] == 0, "DirectML crashes on two at once"
    assert sessions[0].get_inputs() == ["image"] and sessions[0].run(7) == (7,), "the rest of a session is its own"


def test_searching_words_on_pictures_while_their_text_is_being_read():
    count = 5000
    texts = TextIndex()
    for number in range(count):
        texts.set(f"t{number}.jpg", f"words number {number}")
    rows = EmbeddingIndex(ImageIndexFormat(SIGLIP2_SO400M))
    rows.add([f"t{number}.jpg" for number in range(count)], np.ones((count, 8), np.float32), unknown_info(count))
    view, stop, errors = rows.view(), threading.Event(), []

    def keep_reading():
        number = 0
        while not stop.is_set():
            texts.set(f"new{number}.jpg", "fresh words")
            number += 1
    reader = threading.Thread(target=keep_reading)
    reader.start()
    try:
        for _ in range(20):
            errors.append(fails(lambda: TextMatch(texts, ["number 7"], ["words"]).word_share(view)))
            errors.append(fails(lambda: TextMatch(texts, ["number 7"], []).required_rows(view)))
    finally:
        stop.set()
        reader.join()
    assert errors == [None] * 40
