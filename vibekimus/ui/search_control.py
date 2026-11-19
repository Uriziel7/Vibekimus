"""Running searches from the main window: by text, by a picture (a file, a dropped one, a pasted
one), like a result, an Explore topic, and duplicates. Every search goes the same way: the
options of the controls and the filter bar are read, the search is remembered (so a filter
change or Refine runs it again), and it runs in a thread of its own; its results go to the
results view on the Tk thread."""
import os
import time
import traceback
from dataclasses import dataclass, replace
from threading import Thread
from tkinter import filedialog, messagebox
from typing import Callable

from vibekimus import settings
from vibekimus.clipboard import pasted_picture
from vibekimus.duplicates import DuplicateFinder
from vibekimus.imaging import open_image
from vibekimus.runlog import safe_print
from vibekimus.search import NoSearchTerms, NotInIndex, SearchOptions, quoted_and_loose
from vibekimus.ui import theme

IMAGE_DIALOG_TYPES = [("Images", "*.jpg *.jpeg *.png *.webp")]
# Below this many results the user is told how to get more.
FEW_RESULTS = 6
# The hint is pointless when the score slider is already (almost) at the bottom.
HINT_MIN_SCORE = 0.05
HINT_DELAY_AFTER_RESULTS_MS = 500
HINT_DELAY_NO_RESULTS_MS = 100
REPEAT_SEARCH_DELAY_MS = 500
EMPTY_QUERY_MESSAGE = (
    "Please type something in the search box to search.\n\n"
    "To search by image similarity, use Search by image… next to the search box, drop a picture on the "
    "results, or paste one with Ctrl+V.\n\n"
    'Text in "quotes" must be written on the picture (once "Read text (OCR)" has read it).')
NO_TEXT_READ_NOTE = ('the text on the pictures has not been read yet, so "quoted" text was searched by meaning — '
                     'tick "Read text (OCR)" and click Refresh')
UNSUPPORTED_DROP_MESSAGE = ("Only image files can be used for image search.\n"
                            "Drop a JPG, PNG, WEBP or RAW file.")
NO_PICTURE_PASTED = "The clipboard holds no picture (copied files must be pictures)"
PASTED_PICTURE_NAME = "the pasted picture"
FEW_RESULTS_MESSAGE = (
    "Only {count} result(s) found at similarity score {score:.2f}.\n\n"
    "Things to check:\n"
    "• Make sure the Image and/or Video filter buttons are enabled next to the Deselect All button\n"
    "• Try lowering the Similarity Score slider to find more matches\n"
    "  — Text search: scores with this model are low, start at 0.00 and raise slowly\n"
    "  — Image search: scores are much higher, 0.60–0.85 is typical")


def _busy_text(action, text):
    """'Finding similar...' or, with text, 'Finding similar + “at night”...'."""
    return f"{action} + “{text}”..." if text else f"{action}..."


@dataclass(frozen=True)
class SearchControls:
    """The main window's controls a search reads (Tk variables)."""
    min_score: object
    include_images: object
    include_videos: object
    best_frame_only: object


@dataclass(frozen=True)
class SearchHooks:
    is_indexing: Callable           # () -> bool
    ensure_not_indexing: Callable   # () -> bool; tells the user when indexing runs
    schedule: Callable              # (action, delay_ms=0) on the Tk thread
    query: Callable                 # () -> the text in the search field


@dataclass(frozen=True)
class Prepared:
    """What a search is run with: its options, and the filter bar's choice they came from."""
    options: SearchOptions
    choice: object   # filter_bar.FilterChoice


@dataclass(frozen=True)
class _Shown:
    """What a search hands to the results view."""
    results: list
    summary: str = None       # status text instead of "Found N results"
    has_scores: bool = True   # ranked by the Min. score slider (not duplicates)


@dataclass(frozen=True)
class _LastSearch:
    """How to run the shown search again, e.g. when the filters change."""
    run: Callable                        # (paths to keep selected) -> None
    blocked_while_indexing: bool = False


class SearchControl:
    def __init__(self, library, engine, results, filter_bar, controls, models, status, hooks):
        self._library, self._engine, self._results = library, engine, results
        self._filter_bar, self._controls, self._models, self._status = filter_bar, controls, models, status
        self._hooks = hooks
        self._last = None   # _LastSearch of the results shown, None when there are none

    # ── the shown results ────────────────────────────────────────────────────
    def cancel(self):
        """Stop the search that is running (its results are not shown) and empty the results."""
        was_searching = self._results.is_searching
        self._results.cancel()
        if not self._hooks.is_indexing():
            self._status.clear_progress()
        if was_searching:
            self._status.set_status("Search Cancelled", theme.STATUS_BUSY)

    def forget_results(self):
        """Empty the results; changing the filters then has nothing to search again."""
        self.cancel()
        self._last = None
        self._results.forget_ratings()

    def search_again(self, tell_when_blocked=True):
        """Run the shown search again: with the filter bar as it is now, and refined by the
        ratings given to its results (Refine results). A search that may not run during
        indexing says so, or (tell_when_blocked=False) is quietly left as it is."""
        last = self._last
        if last is None:
            return
        try:
            self._filter_bar.read()
        except ValueError as error:   # an unreadable date: say so and keep the results
            self._status.set_status(f"Filter: {error}", theme.STATUS_ERROR)
            return
        if last.blocked_while_indexing and self._hooks.is_indexing():
            if tell_when_blocked:
                self._hooks.ensure_not_indexing()
            return
        keep_selected = frozenset(self._results.selected)
        self.cancel()
        last.run(keep_selected)

    def repeat_after_indexing(self):
        """Now that the index has changed: the search shown, run again (with its ratings and
        the filters); with nothing shown, the text in the search field."""
        if not self._library.is_searchable:
            return
        if self._last is not None:
            self._hooks.schedule(lambda: self.search_again(tell_when_blocked=False), REPEAT_SEARCH_DELAY_MS)
        elif self._hooks.query():
            self._hooks.schedule(self._search_by_text, REPEAT_SEARCH_DELAY_MS)

    def prepared(self):
        """The options of the controls and the filter bar, or None (after telling the user)
        when a filter cannot be read."""
        try:
            choice = self._filter_bar.read()
        except ValueError as error:
            self._status.set_status(f"Filter: {error}", theme.STATUS_ERROR)
            return None
        controls = self._controls
        return Prepared(SearchOptions(
            min_score=controls.min_score.get(),
            include_images=controls.include_images.get(),
            include_videos=controls.include_videos.get(),
            best_frame_per_video=controls.best_frame_only.get(),
            filters=choice.filters,
            media_type=choice.media_type,
            sort=choice.sort,
            feedback=self._results.ratings.feedback(),
        ), choice)

    # ── the searches (each a new one: it begins without ratings) ─────────────
    def _new(self, start):
        self._results.forget_ratings()
        start()

    def search_clicked(self):
        self.cancel()
        self._new(self._search_by_text)

    def find_similar(self, result, text=""):
        self._new(lambda: self._find_similar(result, text))

    def show_topic(self, topic):
        self._new(lambda: self._show_topic(topic))

    def find_duplicates(self, threshold):
        self._new(lambda: self._find_duplicates(threshold))

    def image_clicked(self):
        # Searching by image during indexing is not allowed: same rule as the original app.
        if not self._hooks.ensure_not_indexing() or not self._models.ensure_ready():
            return
        self.cancel()
        path = filedialog.askopenfilename(filetypes=IMAGE_DIALOG_TYPES)
        if path:
            self._search_by_file(path)

    def image_dropped(self, paths):
        """Of several dropped files only the first is used."""
        if not paths or not self._ready_for_picture_search():
            return
        path = paths[0]
        if not path.lower().endswith(settings.IMAGE_EXTS):
            messagebox.showwarning("Unsupported File", UNSUPPORTED_DROP_MESSAGE)
            return
        if os.path.isfile(path):
            self.cancel()
            self._search_by_file(path)

    def paste_picture(self):
        """Search for the picture on the clipboard. Returns whether the key was used for it (False
        when there was no picture: a text field then pastes its text)."""
        if not self._ready_for_picture_search():
            return True
        pasted = pasted_picture()
        if pasted is None:
            self._status.set_status(NO_PICTURE_PASTED, theme.STATUS_BUSY)
            return False
        self.cancel()
        if pasted.path is not None:
            self._search_by_file(pasted.path)
        else:
            self._new(lambda: self._search_by_image(lambda: pasted.image, PASTED_PICTURE_NAME, self._hooks.query()))
        return True

    # ── how each search is run ───────────────────────────────────────────────
    def _search_by_text(self):
        if self._results.is_searching or not self._models.ensure_ready():
            return
        if not self._library.is_searchable:
            messagebox.showwarning("No Data", "Index is empty. Please select a folder.")
            return
        query = self._hooks.query()
        if not query:
            self._status.set_status("Enter a search term", theme.STATUS_BUSY)
            messagebox.showinfo("Empty Search", EMPTY_QUERY_MESSAGE)
            return
        self._search_for(query)

    def _search_for(self, query, keep_selected=frozenset()):
        def find(prepared, generation):
            found = self._engine.by_text(self._models.encoder, query, prepared.options,
                                         lambda: self._results.is_stale(generation))
            return self._shown(found, prepared.choice, query)

        safe_print(f"\n[SEARCH] Starting search for: '{query}'")
        self._run(lambda keep: self._search_for(query, keep), find, "Searching...", "No results found", keep_selected)

    def _find_similar(self, result, text="", keep_selected=frozenset()):
        """Results that look like this one (and match `text`, if given), from its stored
        embedding (instant)."""
        if not self._models.ensure_ready():
            return
        self.cancel()
        self._run(lambda keep: self._find_similar(result, text, keep),
                  lambda prepared, generation: self._shown(
                      self._engine.by_result(self._models.encoder, result, prepared.options, text), prepared.choice),
                  _busy_text("Finding similar", text), "No similar results", keep_selected)

    def _show_topic(self, topic, keep_selected=frozenset()):
        """The pictures of an Explore topic as results, closest to its centre first."""
        if not self._models.ensure_ready():
            return
        self.cancel()

        def find(prepared, generation):
            shown = self._shown(self._engine.by_topic(self._models.encoder, topic, prepared.options), prepared.choice)
            count = len(shown.results)
            of_all = f" of its {topic.size:,}" if count < topic.size else ""
            return replace(shown, summary=shown.summary or f"Topic “{topic.name}”: {count:,}{of_all} pictures")

        self._run(lambda keep: self._show_topic(topic, keep), find, f"Showing “{topic.name}”...",
                  "No pictures of this topic pass the filters", keep_selected)

    def _find_duplicates(self, threshold, keep_selected=frozenset()):
        """Groups of duplicates with at least one image passing the filter bar (type, dates,
        size, resolution, folder), shown whole; groups follow its sort order."""
        self.cancel()
        encoder = self._models.encoder
        device = encoder.search_device if encoder is not None else None

        def find(prepared, generation):
            images = self._library.images
            images.flush()
            allowed = self._engine.allowed_rows(encoder, images, prepared.options)
            report = DuplicateFinder(self._library).find(threshold, device, lambda: self._results.is_stale(generation),
                                                         allowed, prepared.choice.sort)
            scope = "" if allowed is None else " with at least one image matching the filters"
            summary = (f"{report.groups:,} groups of duplicates{scope} — {len(report.to_remove):,} extra copies. "
                       f"Select what to remove; Delete moves the selected files to the Recycle Bin.")
            missing = len(images.rows_missing_info())
            if prepared.choice.needs_file_info and missing:
                summary += f" {missing:,} images have no date or size yet; click Refresh to read them."
            return _Shown(report.results, summary, has_scores=False)

        self._run(lambda keep: self._find_duplicates(threshold, keep), find,
                  "Comparing every image with every other one...", "No duplicates found", keep_selected,
                  blocked_while_indexing=True,
                  ready=lambda prepared: not prepared.choice.media_type or self._models.ensure_ready())

    def _ready_for_picture_search(self):
        """Whether a dropped or pasted picture can be searched for now (otherwise says why)."""
        if not self._models.ensure_ready() or not self._hooks.ensure_not_indexing():
            return False
        if not self._library.is_open:
            messagebox.showwarning("No Folder", "Please select a folder first before searching by image.")
            return False
        if not self._library.is_searchable:
            messagebox.showwarning("Not Indexed", "Please index a folder first before searching by image.")
            return False
        return True

    def _search_by_file(self, path):
        self._new(lambda: self._search_by_image(
            lambda: open_image(path, draft_size=self._models.encoder.decode_size_hint), os.path.basename(path),
            self._hooks.query()))

    def _search_by_image(self, read_picture, name, text="", keep_selected=frozenset()):
        """Files like the picture `read_picture()` returns — and, with text, like the picture
        and the text. name: what the picture is called in messages. Searching by a picture
        during indexing is not allowed, not even to apply filters."""
        def find(prepared, generation):
            image = read_picture()
            if image is None:
                raise ValueError(f"cannot read {name}")
            return self._shown(self._engine.by_image(self._models.encoder, image, prepared.options, text),
                               prepared.choice)

        self._run(lambda keep: self._search_by_image(read_picture, name, text, keep), find,
                  _busy_text("Searching by image", text), "No matches", keep_selected, blocked_while_indexing=True)

    def _shown(self, found, choice, query=""):
        """Search results ready to show, with a note when filters need file details that older
        index entries do not have yet, or when "quoted" text could not be looked for on the pictures."""
        if found is None:
            return None
        notes = []
        library = self._library
        missing = len(library.images.rows_missing_info()) + len(library.videos.rows_missing_info())
        if choice.needs_file_info and missing:
            notes.append(f"{missing:,} indexed files have no date or size yet; click Refresh to read them")
        if quoted_and_loose(query)[0] and not library.texts.has_data:
            notes.append(NO_TEXT_READ_NOTE)
        summary = f"Found {len(found):,} results — " + "; ".join(notes) if notes else None
        return _Shown(found, summary=summary)

    # ── the one way every search runs ────────────────────────────────────────
    def _run(self, rerun, find, busy_text, no_match_text, keep_selected, blocked_while_indexing=False,
             ready=lambda prepared: True):
        """rerun(paths to keep selected) runs the search again; find(prepared, generation) -> _Shown
        (None when cancelled) runs in the search thread; ready(prepared) may still refuse."""
        prepared = self.prepared()
        if prepared is None or not ready(prepared):
            return
        self._last = _LastSearch(rerun, blocked_while_indexing)
        generation = self._results.begin()
        Thread(target=self._search_thread,
               args=(generation, lambda: find(prepared, generation), busy_text, no_match_text, keep_selected),
               daemon=True).start()

    def _search_thread(self, generation, find, busy_text, no_match_text, keep_selected):
        """Compute the results and hand them to the results view. keep_selected: paths that stay
        selected if they are among the new results."""
        results, schedule, status = self._results, self._hooks.schedule, self._status
        if self._models.encoder is not None:
            self._models.encoder.activate()
        owns_progress = not self._hooks.is_indexing()   # otherwise it shows indexing progress
        if owns_progress:
            schedule(lambda: status.set_status(busy_text, theme.STATUS_BUSY))
            schedule(status.show_busy)
        started = time.perf_counter()
        try:
            shown = find()
        except NoSearchTerms:
            schedule(lambda: status.set_status("No positive search terms", theme.STATUS_BUSY))
            results.finish()
            return
        except NotInIndex:
            schedule(lambda: status.set_status("This file is no longer in the index", theme.STATUS_BUSY))
            results.finish()
            return
        except Exception as error:
            if not results.is_stale(generation):
                safe_print(f"[SEARCH ERROR] {error}")
                traceback.print_exc()
                schedule(lambda: status.set_status("Search error - check console", theme.STATUS_ERROR))
            results.finish()
            return
        finally:
            if owns_progress:
                schedule(status.hide_busy)
        if shown is None:
            return   # cancelled; the newer search owns the view now
        self._hand_over(shown, generation, keep_selected, no_match_text, time.perf_counter() - started)

    def _hand_over(self, shown, generation, keep_selected, no_match_text, seconds):
        """Search thread: the results to the view (on the Tk thread), or the word that there are none."""
        results, schedule = self._results, self._hooks.schedule
        found = shown.results
        safe_print(f"[SEARCH] Found {len(found)} results in {seconds:.2f}s")
        if found:
            selected = {result.path for result in found if result.path in keep_selected}
            schedule(lambda: results.show(found, generation, selected, shown.summary))
            schedule(results.focus)   # Space previews now; typing goes back to the search field
            if len(found) < FEW_RESULTS and shown.has_scores:
                schedule(self._suggest_lower_score, HINT_DELAY_AFTER_RESULTS_MS)
        else:
            schedule(lambda: self._status.set_status(no_match_text, theme.STATUS_OK))
            if shown.has_scores:
                schedule(self._suggest_lower_score, HINT_DELAY_NO_RESULTS_MS)
            results.finish()

    def _suggest_lower_score(self):
        score = self._controls.min_score.get()
        if score > HINT_MIN_SCORE:
            messagebox.showinfo("Few Results Found",
                                FEW_RESULTS_MESSAGE.format(count=self._results.total_found, score=score))
