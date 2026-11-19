"""Video indexing pipeline.

Worker threads decode the sample frames of several videos at once while the main
indexing thread encodes each video as soon as its frames are ready — in the order
they finish, so one slow file never holds back the others.
"""
import gc
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from vibekimus import hardware, settings
from vibekimus.indexing.base import AutosaveTimer, IndexingUnavailable, MediaIndexer, RunStats, chunked
from vibekimus.media_info import video_info
from vibekimus.runlog import log_event, safe_print
from vibekimus.storage import DRIVES
from vibekimus.video_io import VideoFrameSampler, silenced_stderr, video_dimensions

# Videos being decoded or waiting for the GPU per worker. Bounded, because every
# finished video holds its decoded frames in RAM until it is encoded.
IN_FLIGHT_PER_WORKER = 2
MIN_FRAMES_PER_GPU_CHUNK = 32
VIDEO_BATCHES_PER_GPU_CHUNK = 2
# Both are slow calls (a GPU sync / a walk over every Python object), so rarely.
EMPTY_CACHE_EVERY_VIDEOS = 50
COLLECT_GARBAGE_EVERY_VIDEOS = 200
WAIT_SECONDS = 0.2   # how often the loop re-checks for a stop while videos decode
NO_FRAMES_REASON = "No frames extracted"
OPENCV_MISSING_MESSAGE = ("OpenCV is required for video indexing.\n\n"
                          "Install it with:\n  pip install opencv-python")


class VideoIndexer(MediaIndexer):
    media_name = "videos"
    skipped_log_name = settings.SKIPPED_VIDEOS_LOG

    @property
    def index(self):
        return self._library.videos

    @property
    def _decode_mode(self):
        return self._options.decoding.videos

    def index_summary(self):
        counts = self._library.counts()
        return f"{counts.videos:,} videos, {counts.frames:,} frames"

    def _read_info(self, abs_path):
        return video_info(abs_path, *video_dimensions(abs_path))

    def _scan(self):
        return self._library.iter_videos(self._should_stop, self._report_scan_progress)

    def _encode_files(self, files, is_update):
        try:
            sampler = VideoFrameSampler(self._should_stop, self._decode_mode, self._options.max_video_seconds)
        except ImportError as error:
            safe_print("[VINDEX ERROR] OpenCV not installed. Run: pip install opencv-python")
            raise IndexingUnavailable(OPENCV_MISSING_MESSAGE) from error

        index = self.index
        index.begin_run()
        run = RunStats(total=len(files))
        known = set(index.keys) if is_update else set()
        try:
            with silenced_stderr():
                self._encode_videos(files, sampler, known, run, "Updating" if is_update else "Indexing")
            safe_print("")
            self._report_speed(run, sampler.hardware_decoded)
        finally:
            self._free_accelerator_memory()
            index.checkpoint(partial=self._should_stop())   # a stop saves only the new rows: fast

    def _encode_videos(self, files, sampler, known, run, verb):
        workers = settings.VIDEO_WORKERS
        autosave = AutosaveTimer()
        queued = iter(files)
        in_flight = {}   # future -> path
        executor = ThreadPoolExecutor(max_workers=workers)
        try:
            while True:
                self._top_up(executor, sampler, queued, in_flight, known, workers * IN_FLIGHT_PER_WORKER)
                if not in_flight:
                    return
                if self._should_stop():
                    safe_print("\n[VINDEX] Stopping video loop.")
                    return
                finished, _ = wait(in_flight, timeout=WAIT_SECONDS, return_when=FIRST_COMPLETED)
                for future in finished:
                    self._encode_video(in_flight.pop(future), future, known, run)
                    self._housekeeping(run)
                    self._report_progress(run, f"{verb} Videos: {run.done:,}/{run.total:,} files "
                                               f"({len(self.index.keys):,} frames)")
                    if autosave.due():
                        self._autosave(f"{run.done:,} videos")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _top_up(self, executor, sampler, queued, in_flight, known, limit):
        while len(in_flight) < limit and not self._should_stop():
            path = next(queued, None)
            if path is None:
                return
            in_flight[executor.submit(self._sample, sampler, path, known)] = path

    def _sample(self, sampler, abs_path, known):
        """Worker: returns (index_path, VideoSample, info record) for one video."""
        index_path = self._library.scope.index_path(abs_path)
        with DRIVES.reading(abs_path):   # a hard disk gives its videos one at a time
            sample = sampler.sample(abs_path, lambda moment: (index_path, moment) in known)
        return index_path, sample, video_info(abs_path, sample.width, sample.height)

    def _encode_video(self, abs_path, future, known, run):
        run.done += 1
        try:
            index_path, sample, info = future.result()
        except Exception as error:
            safe_print(f"[VINDEX] Could not process a video: {error}")
            return
        if sample.too_long:
            run.skipped_long += 1
        elif not sample.frames:
            self.failures.append((abs_path, NO_FRAMES_REASON))
        chunk_size = max(hardware.video_batch_size() * VIDEO_BATCHES_PER_GPU_CHUNK, MIN_FRAMES_PER_GPU_CHUNK)
        for chunk in chunked(sample.frames, chunk_size):
            if self._should_stop():
                return
            self._encode_frames(chunk, index_path, info, known, run)

    def _encode_frames(self, frames, index_path, info, known, run):
        """Every frame of a video shares the video file's info record."""
        try:
            started = time.time()
            features = self._encoder.encode_images([image for image, _moment in frames])
            run.gpu_seconds += time.time() - started
            self._add_new([(index_path, moment) for _image, moment in frames], features, [info] * len(frames), known)
        except Exception as error:
            safe_print(f"[VINDEX ERROR] Encoding failed: {error}")
            log_event(f"VIDEOS encode error: {error}")
            self._encoder.empty_cache()

    def _housekeeping(self, run):
        if run.done % EMPTY_CACHE_EVERY_VIDEOS == 0:
            self._encoder.empty_cache()
        if run.done % COLLECT_GARBAGE_EVERY_VIDEOS == 0:
            gc.collect()

    def _report_speed(self, run, hardware_decoded):
        if not run.done:
            return
        report = (f"Speed: {run.done:,} of {run.total:,} videos ({len(self.index.keys):,} frames in index) "
                  f"in {run.elapsed:,.0f}s = {run.done / run.elapsed:,.1f} videos/s. "
                  f"Decoding: {self._decode_mode.value} ({hardware_decoded:,} by the GPU video decoder). "
                  f"Encoder busy {run.gpu_busy_percent:.0f}% of the time, "
                  f"the rest it waited for frames to be decoded.")
        limit = self._options.max_video_seconds
        if limit:
            report += f" Skipped as longer than {limit}s: {run.skipped_long:,}."
        safe_print(f"[VINDEX] {report}")
        log_event(f"VIDEOS {report} No frames from: {len(self.failures):,}.")
