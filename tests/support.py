"""Stand-ins and helpers shared by the tests (no GPU, no model, no display needed).

A stand-in encoder (mean colour -> unit vector) replaces the real model, so the tests check
the plumbing: index files, path validation, the image and video pipelines, search, ranking."""
import io
import os
import time

import cv2
import numpy as np
from PIL import Image

from vibekimus.decoding import DecodingOptions
from vibekimus.encoders.base import MediaEncoder
from vibekimus.encoders.catalog import SIGLIP2_SO400M
from vibekimus.indexing.base import IndexingListener
from vibekimus.indexing.controller import IndexingController, IndexingPass
from vibekimus.indexing.options import IndexingOptions
from vibekimus.library import MediaLibrary
from vibekimus.scope import FolderScope
from vibekimus.similarity import SimilarityEngine, top_indices

COLOURS = {
    "red": (220, 30, 30), "green": (30, 200, 40), "blue": (30, 40, 220),
    "purple": (150, 30, 200), "yellow": (230, 220, 30), "white": (245, 245, 245),
}
VIDEO_FPS = 10.0
GB = 1024 ** 3
EVERYTHING = -2.0   # a min. score every result passes (unwanted terms make scores negative)
HANG_SECONDS = 5        # each file "decodes" this long in the Stop tests
MAX_STOP_SECONDS = 1.5  # time allowed from the stop request to the end of the run
STUCK_SECONDS = 0.5     # by then the workers are stuck inside their first files


def colour_vec(rgb):
    v = np.array(list(rgb) + [255 - c for c in rgb], dtype=np.float32) - 127.5
    return v / np.linalg.norm(v)


def embed_pil(img):
    arr = np.asarray(img.convert("RGB").resize((8, 8)), dtype=np.float32)
    return colour_vec(arr.reshape(-1, 3).mean(axis=0))


class FakeModel(MediaEncoder):
    """The encoder interface, minus the neural network: a colour name or a picture's
    mean colour becomes the embedding."""
    image_size = 384

    def __init__(self):
        self.seen_sizes = []   # sizes of the pictures handed to prepare()

    def prepare(self, image):
        self.seen_sizes.append(image.size)
        return embed_pil(image)

    def encode_prepared(self, items):
        return np.stack(items)

    def encode_images(self, images):
        return np.stack([embed_pil(img) for img in images])

    def encode_text(self, texts):
        """The colour words of each text (other words carry no meaning here)."""
        return np.stack([np.mean([colour_vec(COLOURS[word]) for word in text.split() if word in COLOURS]
                                 or [colour_vec(COLOURS["white"])], axis=0) for text in texts])


class FakeJpegDecoder:
    """Stands in for the GPU JPEG decoder: decodes with PIL, and refuses every third
    file (like nvJPEG refuses CMYK JPEGs) so the hand-back to the CPU is exercised."""

    def __init__(self):
        self.decoded = self.refused = 0
        self._calls = 0

    def decode(self, blobs):
        out = []
        for blob in blobs:
            self._calls += 1
            if self._calls % 3 == 0:
                self.refused += 1
                out.append(None)
            else:
                self.decoded += 1
                out.append(embed_pil(Image.open(io.BytesIO(blob))))
        return out


class FakeGpuModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.jpeg_decoder = FakeJpegDecoder()

    def create_jpeg_decoder(self):
        return self.jpeg_decoder


class FakeTextReader:
    """Stands in for an OCR engine: "reads" a text chosen by the picture's colour."""
    TEXT_BY_COLOUR = {"red": "Darmowe piwo\nw piątek", "blue": "Zażółć gęślą jaźń", "green": ""}
    made = closed = 0

    def __init__(self, threads):
        FakeTextReader.made += 1

    def read(self, image):
        mean = np.asarray(image.convert("RGB").resize((1, 1)), dtype=np.float32).reshape(3)
        colour = min(self.TEXT_BY_COLOUR, key=lambda name: np.abs(mean - COLOURS[name]).sum())
        return self.TEXT_BY_COLOUR[colour]

    def close(self):
        FakeTextReader.closed += 1


class Recorder(IndexingListener):
    def __init__(self):
        self.outcomes = []

    def on_pass_done(self, outcome, is_last):
        self.outcomes.append((outcome.media, outcome.state, is_last))


def open_library(scope, model=SIGLIP2_SO400M):
    """A library of a folder (a path) or of any scope."""
    library = MediaLibrary(model)
    library.open(FolderScope(str(scope)) if isinstance(scope, (str, os.PathLike)) else scope)
    return library


def run(library, *passes, encoder=None, decoding=DecodingOptions(), max_video_seconds=None):
    """Run indexing passes to completion in this thread; returns (media, state, is_last) of each pass."""
    recorder = Recorder()
    options = (IndexingOptions(decoding=decoding) if max_video_seconds is None
               else IndexingOptions(decoding=decoding, max_video_seconds=max_video_seconds))
    IndexingController(library, recorder).run(encoder or FakeModel(), passes, options)
    return recorder.outcomes


def seconds_to_stop(library, media):
    """Start indexing, press Stop while the workers are stuck in their first files; how long it takes to end."""
    controller = IndexingController(library)
    controller.start(FakeModel(), [rebuild(media)])
    time.sleep(STUCK_SECONDS)
    pressed = time.time()
    controller.request_stop()
    while controller.is_indexing and time.time() - pressed < 2 * HANG_SECONDS:
        time.sleep(0.02)
    return time.time() - pressed


def rebuild(media):
    return IndexingPass(media, rebuild=True)


def refresh(media):
    return IndexingPass(media)


def rank(index, vec, limit=5):
    sims = SimilarityEngine().scores(index.format.kind, index.embeddings, vec)
    idx = top_indices(sims, 0.0, limit)
    return [int(i) for i in idx[np.argsort(-sims[idx])]]


def video_files(index):
    return sorted(index.files())


def names(results):
    """The file names of search results, sorted."""
    return sorted(os.path.basename(result.path) for result in results)


def in_order(results):
    """The file names of search results, in their order."""
    return [os.path.basename(result.path) for result in results]


def as_key(*parts):
    """An absolute index key: the path with forward slashes."""
    return os.path.join(*parts).replace("\\", "/")


def fails(action):
    """Runs action(); returns the error message if it raised, else None."""
    try:
        action()
    except Exception as error:
        return str(error) or type(error).__name__
    return None


def raised(action):
    """The exception action() raised, or None."""
    try:
        action()
    except Exception as error:
        return error
    return None


def save_picture(path, rgb, size=(400, 300), **save_options):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, rgb).save(path, **save_options)
    return path


def write_clip(path, rgb, frames=30, size=(640, 360)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, size)
    frame = np.zeros((size[1], size[0], 3), np.uint8)
    frame[:] = rgb[::-1]   # BGR
    for _ in range(frames):
        writer.write(frame)
    writer.release()
