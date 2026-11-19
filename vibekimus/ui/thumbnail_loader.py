"""Making thumbnails in the background and handing them to the Tk thread.

Thumbnails are made by a few threads at once: a video's frame can take a quarter of a
second or more (decoding from its key frame), and pictures must not wait behind it. The Tk
thread picks the finished ones up in short slices, so the window stays responsive.
cancel_all() drops every pending request (a new page, a new search)."""
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from vibekimus.media_kind import MediaKind
from vibekimus.runlog import safe_print
from vibekimus.thumbnails import VideoSupportMissing, make_thumbnail

WORKERS = 4                    # thumbnails made at once
QUEUE_POLL_MS = 10             # how often finished thumbnails are moved to the window
WORKER_POLL_MS = 20            # same, while the queue is empty but a request still runs
MAX_UI_SLICE_SECONDS = 0.1     # longest the Tk thread hands thumbnails over before yielding
_FINISHED = object()           # queued after a request's last thumbnail


class _Request:
    def __init__(self, results, box, on_ready, on_done):
        self.results = results
        self.box = box
        self.on_ready = on_ready   # (result, PIL image) on the Tk thread
        self.on_done = on_done     # () on the Tk thread, after the last on_ready
        self.active = True
        self.asked = time.perf_counter()
        self.left = len(results)   # thumbnails not made yet
        self.loaded = self.failed = 0
        self.made_in = 0.0         # seconds until the last one was made
        self.lock = threading.Lock()


class ThumbnailLoader:
    def __init__(self, root, on_video_support_missing):
        self._root = root
        self._on_video_support_missing = on_video_support_missing
        self._queue = queue.Queue()
        self._requests = []
        self._polling = False
        self._workers = ThreadPoolExecutor(WORKERS, thread_name_prefix="thumbnail")

    def request(self, results, box, on_ready, on_done=lambda: None):
        """Make thumbnails no larger than box = (width, height) for the results."""
        job = _Request(list(results), box, on_ready, on_done)
        self._requests.append(job)
        if not job.results:
            self._queue.put((job, _FINISHED, None))
        for result in sorted(job.results, key=lambda result: result.kind == MediaKind.VIDEO):   # quick pictures first
            self._workers.submit(self._make, job, result)
        if not self._polling:
            self._polling = True
            self._root.after(QUEUE_POLL_MS, self._hand_over)

    def cancel_all(self):
        for job in self._requests:
            job.active = False
        self._requests.clear()
        with self._queue.mutex:
            self._queue.queue.clear()

    def _make(self, job, result):
        """Worker thread: one thumbnail (skipped once its request is cancelled)."""
        made = False
        try:
            if job.active:
                self._queue.put((job, result, make_thumbnail(result, job.box)))
                made = True
        except VideoSupportMissing:
            self._root.after(0, self._on_video_support_missing)
        except Exception:
            pass
        finally:
            self._count(job, made)

    def _count(self, job, made):
        with job.lock:
            job.loaded += made
            job.failed += not made and job.active
            job.left -= 1
            last = job.left == 0
        if last:
            job.made_in = time.perf_counter() - job.asked
            self._queue.put((job, _FINISHED, None))

    def _hand_over(self):
        """Tk thread: pass finished thumbnails on, a short slice at a time."""
        started = time.time()
        while time.time() - started <= MAX_UI_SLICE_SECONDS:
            try:
                job, result, image = self._queue.get_nowait()
            except queue.Empty:
                break
            if not job.active:
                continue
            try:
                if result is _FINISHED:
                    self._requests.remove(job)
                    safe_print(f"[THUMBNAILS] {job.loaded} loaded in {job.made_in:.2f}s, all shown after "
                               f"{time.perf_counter() - job.asked:.2f}s, {job.failed} failed")
                    job.on_done()
                else:
                    job.on_ready(result, image)
            except Exception as error:   # e.g. its widget was closed meanwhile; keep the others going
                safe_print(f"[THUMBNAILS] Could not show a thumbnail: {error}")
        if self._requests or not self._queue.empty():
            self._root.after(QUEUE_POLL_MS if not self._queue.empty() else WORKER_POLL_MS, self._hand_over)
        else:
            self._polling = False
