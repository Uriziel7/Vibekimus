"""Reading and decoding images for the encoder, on CPU threads and optionally the GPU.

Files are handed out from a shared work queue; whatever is ready goes into one
bounded queue that the encoder drains. There is no per-batch barrier, so one slow
file never holds back the files after it, and the GPU always has work queued.

Decode modes (see vibekimus.decoding.DecodeMode):
  CPU     — CPU threads decode everything.
  GPU     — the GPU decodes all JPEGs; other formats are always decoded on the CPU.
  HYBRID  — the GPU decodes JPEGs and the CPU threads, once the other formats are
            done, take JPEGs from the same queue.
A JPEG the GPU cannot decode (e.g. CMYK) is handed back to the CPU threads.
"""
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

from vibekimus import settings
from vibekimus.decoding import DecodeMode
from vibekimus.imaging import open_image, read_bytes
from vibekimus.media_info import image_info
from vibekimus.runlog import safe_print

JPEG_EXTENSIONS = (".jpg", ".jpeg")
GPU_BATCH = 32                 # JPEGs decoded on the GPU per call
GPU_READ_THREADS = 4           # threads reading JPEG files for the GPU
QUEUED_CHUNKS = 2              # ready images held for the encoder, in STREAM_CHUNKs
WAIT_SECONDS = 0.2             # how often waiting threads re-check for a stop
_PRODUCER_DONE = object()


class _WorkQueue:
    """Files still to be decoded, split into JPEGs and everything else."""

    def __init__(self, files, gpu_takes_jpegs):
        is_jpeg = (lambda path: path.lower().endswith(JPEG_EXTENSIONS)) if gpu_takes_jpegs else (lambda path: False)
        self._jpegs = deque(path for path in files if is_jpeg(path))
        self._others = deque(path for path in files if not is_jpeg(path))
        self._gpu_running = gpu_takes_jpegs
        self._changed = threading.Condition()

    def take_for_cpu(self, may_take_jpegs, is_stopped):
        """Next file for a CPU thread, or None when there will be no more."""
        with self._changed:
            while not is_stopped():
                if self._others:
                    return self._others.popleft()
                if may_take_jpegs and self._jpegs:
                    return self._jpegs.popleft()
                if not self._gpu_running:
                    return None
                self._changed.wait(WAIT_SECONDS)   # the GPU may still hand files back
            return None

    def take_for_gpu(self, count):
        with self._changed:
            return [self._jpegs.popleft() for _ in range(min(count, len(self._jpegs)))]

    def hand_back(self, paths):
        """Files the GPU could not decode go to the CPU threads."""
        with self._changed:
            self._others.extend(paths)
            self._changed.notify_all()

    def gpu_finished(self):
        with self._changed:
            self._gpu_running = False
            self._changed.notify_all()


class LoadedImage(NamedTuple):
    path: str
    item: object    # what the encoder's prepare() made, or None if the file is unreadable
    info: tuple     # media_info record (date, size, dimensions)


class ImageLoader:
    """Yields a LoadedImage for every file, in the order they become ready."""

    def __init__(self, encoder, mode, should_stop):
        self._encoder = encoder
        self._mode = mode
        self._should_stop = should_stop
        self._closed = threading.Event()
        self._ready = queue.Queue(maxsize=QUEUED_CHUNKS * settings.STREAM_CHUNK)
        self.gpu_decoded = 0

    def _is_stopped(self):
        return self._closed.is_set() or self._should_stop()

    def load(self, files):
        gpu_decoder = self._encoder.create_jpeg_decoder() if self._mode.uses_gpu else None
        if self._mode.uses_gpu and gpu_decoder is None:
            safe_print("[INDEX] GPU decoding is not available here — decoding on the CPU")
        work = _WorkQueue(files, gpu_takes_jpegs=gpu_decoder is not None)
        cpu_may_take_jpegs = gpu_decoder is None or self._mode is DecodeMode.HYBRID
        producers = [threading.Thread(target=self._cpu_worker, args=(work, cpu_may_take_jpegs), daemon=True)
                     for _ in range(settings.LOAD_WORKERS)]
        if gpu_decoder is not None:
            producers.append(threading.Thread(target=self._gpu_worker, args=(work, gpu_decoder), daemon=True))
        for producer in producers:
            producer.start()
        try:
            yield from self._drain(len(producers))
        finally:
            self._closed.set()   # lets the producers exit if the consumer stops early

    def _drain(self, producer_count):
        running = producer_count
        while running and not self._should_stop():
            try:
                item = self._ready.get(timeout=WAIT_SECONDS)
            except queue.Empty:
                continue
            if item is _PRODUCER_DONE:
                running -= 1
            else:
                yield item

    def _put(self, item):
        while not self._is_stopped():
            try:
                self._ready.put(item, timeout=WAIT_SECONDS)
                return
            except queue.Full:
                continue

    def _finish_producer(self):
        try:
            self._ready.put(_PRODUCER_DONE, timeout=WAIT_SECONDS)
        except queue.Full:
            # Only possible after a stop; the consumer is no longer reading.
            pass

    # ── CPU ──────────────────────────────────────────────────────────────────
    def _cpu_worker(self, work, may_take_jpegs):
        try:
            while True:
                path = work.take_for_cpu(may_take_jpegs, self._is_stopped)
                if path is None:
                    return
                self._put(LoadedImage(path, self._decode_on_cpu(path), image_info(path)))
        finally:
            self._finish_producer()

    def _decode_on_cpu(self, path):
        """Only the prepared item is kept — holding decoded images in the queue would
        take gigabytes of RAM for large photos."""
        try:
            image = open_image(path, draft_size=self._encoder.decode_size_hint)
            return None if image is None else self._encoder.prepare(image)
        except Exception:
            return None

    # ── GPU ──────────────────────────────────────────────────────────────────
    def _gpu_worker(self, work, decoder):
        """Reads the next batch of JPEGs while the current one is being decoded."""
        readers = ThreadPoolExecutor(max_workers=GPU_READ_THREADS)
        try:
            pending = self._read_batch(readers, work.take_for_gpu(GPU_BATCH))
            while pending and not self._is_stopped():
                paths, reads = pending
                blobs = [read.result() for read in reads]
                pending = self._read_batch(readers, work.take_for_gpu(GPU_BATCH))
                self._decode_on_gpu(work, decoder, paths, blobs)
        except Exception as error:
            safe_print(f"[INDEX] GPU decoding stopped ({error}); the CPU takes over")
        finally:
            readers.shutdown(wait=False, cancel_futures=True)
            work.gpu_finished()
            self._finish_producer()

    @staticmethod
    def _read_batch(readers, paths):
        if not paths:
            return None
        return paths, [readers.submit(_read_file, path) for path in paths]

    def _decode_on_gpu(self, work, decoder, paths, blobs):
        readable = [i for i, blob in enumerate(blobs) if blob is not None]
        tensors = decoder.decode([blobs[i] for i in readable]) if readable else []
        decoded = {i: tensor for i, tensor in zip(readable, tensors) if tensor is not None}
        work.hand_back([path for i, path in enumerate(paths) if i not in decoded])
        for i, tensor in decoded.items():
            self._put(LoadedImage(paths[i], tensor, image_info(paths[i], blobs[i])))
        self.gpu_decoded += len(decoded)


def _read_file(path):
    try:
        return read_bytes(path)
    except OSError:
        return None

