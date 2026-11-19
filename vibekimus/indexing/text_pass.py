"""The indexing pass that reads the text on pictures (OCR).

It follows the image index instead of scanning the disk: pictures in the image
index whose text has not been read yet are read, and entries of pictures that left
the image index are dropped. The pictures are read by worker processes, each with
several OCR engines (indexing.text_reading_pool); Stop ends them at once."""
import time

from vibekimus import processor, settings
from vibekimus.indexing.base import AutosaveTimer, Indexer, IndexingUnavailable, PassState, RunStats
from vibekimus.indexing.text_reading_pool import ProcessReadingPool
from vibekimus.ocr_capacity import OcrProcessBudget, ProcessChange
from vibekimus.runlog import log_event, safe_print
from vibekimus.text_reading import GPU_NAME, TextReaderUnavailable, check_available, reading_device

RESULT_WAIT_SECONDS = 0.2    # how often the collecting loop looks at the stop signal


class TextIndexer(Indexer):
    media_name = "text"
    # The readers (processes, engines each, CPU threads each engine) and the check that OCR
    # can run; tests put stand-ins here.
    make_pool = ProcessReadingPool
    check_reader = staticmethod(check_available)
    make_budget = OcrProcessBudget   # (reading on the graphics card) -> how many processes read

    @property
    def skipped_log_name(self):
        return settings.SKIPPED_TEXT_LOG

    def index_summary(self):
        texts = self._library.texts
        return f"{len(texts):,} pictures read, {texts.with_text():,} with text"

    def rebuild(self):
        self._ensure_reader()   # before anything is cleared: a missing model must not cost the text read
        self._library.texts.clear()
        return self._read_missing()

    def refresh(self):
        self._ensure_reader()
        return self._read_missing()

    # ── the work ─────────────────────────────────────────────────────────────
    def _ensure_reader(self):
        try:
            self.check_reader()
        except TextReaderUnavailable as error:
            raise IndexingUnavailable(str(error)) from error

    def _read_missing(self):
        images, texts = self._library.images, self._library.texts
        if not images.keys:
            return PassState.NOTHING_FOUND
        removed = texts.keep_only(set(images.keys))
        read = texts.keys()
        todo = [key for key in images.keys if key not in read]
        if not todo:
            if removed:
                texts.save()
            safe_print("[TEXT] Text on pictures is up to date.")
            return PassState.UP_TO_DATE
        device = reading_device()
        safe_print(f"[TEXT] Reading text on {len(todo):,} pictures ({device})...")
        log_event(f"TEXT pass: {len(todo):,} pictures to read on the {device}, {removed:,} entries removed")
        self._read_all(todo, self.make_budget(device == GPU_NAME))
        texts.save()
        return PassState.STOPPED if self._should_stop() else PassState.FINISHED

    def _read_all(self, keys, budget):
        processes, most = budget.to_start(), budget.most()
        engines = max(1, settings.OCR_WORKERS)
        usable = processor.background_threads(settings.FREE_CPU_CORES)   # some cores are kept free
        threads = max(1, usable // (most * engines))   # shared by as many processes as may ever run
        self._tell(f"{processes} reading processes to start with (at most {most}), "
                   f"{engines} pictures at a time in each")
        pool = self.make_pool(processes, engines, threads)
        to_abs = self._library.scope.absolute_path
        pool.start([(key, to_abs(key)) for key in keys])
        try:
            self._collect(keys, pool, to_abs, budget)
        finally:
            pool.stop()

    @staticmethod
    def _tell(text):
        safe_print(f"\n[TEXT] {text}")
        log_event(f"TEXT pass: {text}")

    def _follow_budget(self, pool, budget):
        """One process fewer when the graphics memory runs low (it finishes its pictures first),
        one more when there is room again."""
        change = budget.change(pool.running())
        if change == ProcessChange.FEWER and pool.retire_one():
            self._tell(f"The graphics memory is running low: one reading process ends, {pool.running()} go on")
        elif change == ProcessChange.MORE:
            pool.add_one()
            self._tell(f"Graphics memory free again: one more reading process, {pool.running()} now")

    def _collect(self, keys, pool, to_abs, budget):
        texts, run, autosave = self._library.texts, RunStats(total=len(keys)), AutosaveTimer()
        while run.done < run.total and not self._should_stop():
            self._follow_budget(pool, budget)
            result = pool.next_result(RESULT_WAIT_SECONDS)
            if result is None:
                if pool.finished():   # every reader ended early (a process crashed)
                    break
                continue
            key, text, problem = result
            if key is None:   # an engine could not even start
                raise IndexingUnavailable(problem)
            run.done += 1
            if problem is None:
                texts.set(key, text)
            else:
                self.failures.append((to_abs(key), problem))
            self._report(run, pool.running())
            if autosave.due():
                texts.save()
        lost = run.total - run.done
        if lost and not self._should_stop():
            self._tell(f"{lost:,} pictures were not read (a reading process ended); the next Refresh reads them")
        safe_print(f"\n[TEXT] {run.done:,} pictures read in {run.elapsed:.0f}s "
                   f"({run.done / run.elapsed:.1f}/s), {texts.with_text():,} with text")

    def _report(self, run, processes):
        rate = run.done / run.elapsed
        remaining = (run.total - run.done) / rate if rate else 0
        text = (f"Reading text: {run.done:,}/{run.total:,} ({rate:.1f}/s, "
                f"{time.strftime('%H:%M:%S', time.gmtime(remaining))} left, {processes} processes)")
        self._listener.on_progress(run.percent, text)
