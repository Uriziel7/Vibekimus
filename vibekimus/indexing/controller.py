"""Runs indexing passes in a background thread and reports how each one ended."""
import threading
import traceback
from dataclasses import dataclass

from vibekimus import processor, settings
from vibekimus.indexing.base import IndexingListener, IndexingRun, IndexingUnavailable, PassState
from vibekimus.indexing.images import ImageIndexer
from vibekimus.indexing.options import IndexingOptions
from vibekimus.indexing.text_pass import TextIndexer
from vibekimus.indexing.videos import VideoIndexer
from vibekimus.runlog import log_event, safe_print
from vibekimus.storage import DRIVES

IMAGES = ImageIndexer.media_name
VIDEOS = VideoIndexer.media_name
TEXT = TextIndexer.media_name
INDEXERS = {IMAGES: ImageIndexer, VIDEOS: VideoIndexer, TEXT: TextIndexer}
# A following pass runs only after one of these; a stop or a failure ends the sequence.
STATES_THAT_CONTINUE = (PassState.FINISHED, PassState.UP_TO_DATE)
# States after which files were encoded: memory is released and failures are logged.
STATES_AFTER_ENCODING = (PassState.FINISHED, PassState.STOPPED)


@dataclass(frozen=True)
class IndexingPass:
    """One unit of work: rebuild or refresh the index of one media kind."""
    media: str
    rebuild: bool = False


@dataclass(frozen=True)
class PassOutcome:
    media: str
    state: PassState
    message: str = ""   # explanation when state is FAILED


class IndexingController:
    """Owns the 'is indexing' state and the stop request shared by all passes.

    Every run gets its own stop signal. A stopped run's worker threads may still be
    finishing the file they had started; they keep seeing their own (set) signal, so
    starting the next run right away does not bring them back to life.
    """

    def __init__(self, library, listener=None):
        self._library = library
        self._listener = listener or IndexingListener()
        self._stop = threading.Event()
        self._thread = None
        self.is_indexing = False

    def start(self, encoder, passes, options=IndexingOptions()):
        """Run `passes` one after another in a background thread. Returns False
        (and does nothing) if a run is already in progress."""
        if self.is_indexing:
            safe_print("[INDEX] ⚠ Indexing is already running, request ignored")
            return False
        self.is_indexing = True
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(encoder, list(passes), options, self._stop),
                                        daemon=True)
        self._thread.start()
        return True

    def request_stop(self):
        """Ask the running pass to save what it has and end. Returns True if a
        request was made now."""
        if not self.is_indexing or self._stop.is_set():
            return False
        self._stop.set()
        safe_print("\n[STOP] Stop requested.")
        return True

    def run(self, encoder, passes, options=IndexingOptions()):
        """Run the passes in the calling thread."""
        self.is_indexing = True
        self._stop = threading.Event()
        self._run(encoder, passes, options, self._stop)

    def _run(self, encoder, passes, options, stop):
        self._report_drives()
        try:
            with processor.in_background(settings.FREE_CPU_CORES):   # cores kept free for the rest of the computer
                self._run_passes(encoder, passes, options, stop)
        finally:
            self.is_indexing = False

    def _run_passes(self, encoder, passes, options, stop):
        run = IndexingRun(encoder, options, stop.is_set, self._listener)
        for number, indexing_pass in enumerate(passes):
            outcome = self._run_pass(run, indexing_pass)
            is_last = number == len(passes) - 1 or outcome.state not in STATES_THAT_CONTINUE
            if is_last:
                self.is_indexing = False
            self._listener.on_pass_done(outcome, is_last)
            if is_last:
                break

    def _report_drives(self):
        drives = ", ".join(DRIVES.describe(root) for root in self._library.scope.scan_roots)
        safe_print(f"[DRIVES] {drives}")
        log_event(f"DRIVES {drives}")

    def _run_pass(self, run, indexing_pass):
        media = indexing_pass.media
        indexer = INDEXERS[media](self._library, run)
        try:
            state = indexer.rebuild() if indexing_pass.rebuild else indexer.refresh()
        except IndexingUnavailable as error:
            return PassOutcome(media, PassState.FAILED, str(error))
        except Exception as error:
            traceback.print_exc()
            log_event(f"{media.upper()} pass failed: {error}")
            return PassOutcome(media, PassState.FAILED, str(error))
        if state in STATES_AFTER_ENCODING:
            self._after_encoding(run.encoder, indexer, state)
        return PassOutcome(media, state)

    def _after_encoding(self, encoder, indexer, state):
        ending = "stopped by user" if state is PassState.STOPPED else "finished"
        log_event(f"{indexer.media_name.upper()} pass {ending}: {indexer.index_summary()} in index")
        self._library.write_skipped_log(indexer.failures, indexer.skipped_log_name)
        encoder.release_memory()
