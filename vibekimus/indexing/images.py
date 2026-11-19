"""Image indexing: ImageLoader decodes files (CPU threads, optionally the GPU) while
this thread encodes whatever is ready, STREAM_CHUNK images at a time."""
import time

from vibekimus import settings
from vibekimus.indexing.base import AutosaveTimer, MediaIndexer, RunStats
from vibekimus.indexing.image_loading import ImageLoader
from vibekimus.media_info import image_info
from vibekimus.runlog import log_event, safe_print

# empty_cache() is a GPU sync barrier that can take 100-500 ms on Windows.
EMPTY_CACHE_EVERY_CHUNKS = 40
UNREADABLE_REASON = "Failed to open or decode"


class ImageIndexer(MediaIndexer):
    media_name = "images"
    skipped_log_name = settings.SKIPPED_IMAGES_LOG

    @property
    def index(self):
        return self._library.images

    @property
    def _decode_mode(self):
        return self._options.decoding.images

    def index_summary(self):
        return f"{self._library.counts().images:,} images"

    def _read_info(self, abs_path):
        return image_info(abs_path)

    def _scan(self):
        return self._library.iter_images(self._should_stop, self._report_scan_progress)

    def _encode_files(self, files, is_update):
        index = self.index
        index.begin_run()
        run = RunStats(total=len(files))
        known = set(index.keys)
        self._encoder.reset_peak_memory()
        loader = ImageLoader(self._encoder, self._decode_mode, self._should_stop)
        try:
            self._encode_stream(loader.load(files), known, run, "Updating" if is_update else "Indexing")
            safe_print("")
            self._report_speed(run, loader.gpu_decoded)
        finally:
            self._free_accelerator_memory()
            index.checkpoint(partial=self._should_stop())   # a stop saves only the new rows: fast

    def _encode_stream(self, loaded, known, run, verb):
        autosave = AutosaveTimer()
        ready, chunks = [], 0
        for image in loaded:
            if image.item is None:
                self.failures.append((image.path, UNREADABLE_REASON))
                continue
            ready.append(image)
            if len(ready) >= settings.STREAM_CHUNK:
                chunks += 1
                self._encode_chunk(ready, known, run, chunks, autosave, verb)
                ready = []
        if ready and not self._should_stop():
            self._encode_chunk(ready, known, run, chunks + 1, autosave, verb)

    def _encode_chunk(self, images, known, run, number, autosave, verb):
        self._encode_images(images, known, run)
        if number % EMPTY_CACHE_EVERY_CHUNKS == 0:
            self._encoder.empty_cache()
        if autosave.due():
            self._autosave(f"{run.done:,} images")
        self._report_progress(run, f"{verb}: {run.done:,}/{run.total:,}  ({run.done / run.elapsed:,.0f}/s)")

    def _encode_images(self, images, known, run):
        try:
            started = time.time()
            features = self._encoder.encode_prepared([image.item for image in images])
            run.gpu_seconds += time.time() - started
            to_index_path = self._library.scope.index_path
            run.done += self._add_new([to_index_path(image.path) for image in images], features,
                                      [image.info for image in images], known)
        except Exception as error:
            safe_print(f"[ERROR] Stream encode chunk: {error}")
            log_event(f"IMAGES encode error: {error}")

    def _report_speed(self, run, gpu_decoded):
        """Tells which side is the bottleneck: a low encoder share means the GPU was
        waiting for files to be read and decoded; a share near 100% means encoding
        itself is the limit."""
        if not run.done:
            return
        peak = self._encoder.peak_memory_gb()
        peak_text = f" Peak VRAM {peak:.1f} GB." if peak is not None else ""
        report = (f"Speed: {run.done:,} of {run.total:,} images in {run.elapsed:,.0f}s = "
                  f"{run.done / run.elapsed:,.0f}/s. Decoding: {self._decode_mode.value} "
                  f"({gpu_decoded:,} decoded on the GPU). Encoder busy {run.gpu_busy_percent:.0f}% "
                  f"of the time, the rest it waited for files to be read and decoded.{peak_text}")
        safe_print(f"[INDEX] {report}")
        log_event(f"IMAGES {report} Could not open: {len(self.failures):,}.")
