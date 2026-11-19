"""The main window: it builds the toolbar, the search bar, the controls, the filter bar, the
results and the status line, and connects the parts that do the work — ModelControl (the
search model), LibraryControl (which index is open), SearchControl (the searches), KeyRouter
(the keys) and the IndexingController."""
import os
import tkinter as tk
from dataclasses import replace
from tkinter import messagebox, ttk

from vibekimus import advanced_options, settings
from vibekimus.explore import TopicFinder, TopicNamer
from vibekimus.indexing.controller import IMAGES, TEXT, VIDEOS, IndexingController, IndexingPass
from vibekimus.last_index import LastIndex
from vibekimus.library import MediaLibrary
from vibekimus.library_stats import StatisticsCollector
from vibekimus.media_kind import MediaKind
from vibekimus.model_store import OCR_KEY
from vibekimus.output_log import OUTPUT, capture
from vibekimus.preferences import Preferences
from vibekimus.runlog import safe_print
from vibekimus.search import SearchEngine, SearchResult
from vibekimus.thumbnails import make_thumbnail
from vibekimus.ui import theme
from vibekimus.ui.console_window import ConsoleWindow
from vibekimus.ui.dialogs import CloseWhileIndexingDialog, ExclusionsDialog
from vibekimus.ui.duplicates_dialog import DuplicatesDialog
from vibekimus.ui.encryption_control import EncryptionControl
from vibekimus.ui.explore_window import ExploreHooks, ExploreWindow
from vibekimus.ui.filter_bar import FilterBar
from vibekimus.ui.flow import FlowFrame
from vibekimus.ui.index_info import show_index_info
from vibekimus.ui.indexing_reporter import IndexingReporter
from vibekimus.ui.key_router import KeyRouter
from vibekimus.ui.library_control import LibraryControl, LibraryHooks
from vibekimus.ui.model_control import ModelControl
from vibekimus.ui.result_actions import ResultActions
from vibekimus.ui.results_view import SELECTION_COMMANDS, ResultsView, ResultsViewHooks
from vibekimus.ui.search_bar import SearchBar
from vibekimus.ui.search_control import SearchControl, SearchControls, SearchHooks
from vibekimus.ui.selection_actions import SelectionActions
from vibekimus.ui.settings_dialog import SettingsDialog
from vibekimus.ui.statistics_window import StatisticsWindow
from vibekimus.ui.status_bar import StatusBar
from vibekimus.ui.widgets import ControlGroup, LabeledSlider

# Drag-and-drop support — optional, the app works without tkinterdnd2.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    DND_FILES = TkinterDnD = None

WINDOW_TITLE = "Vibekimus - AI Media Search"
WINDOW_GEOMETRY = "1400x900"
# Below this the controls would wrap into more rows than leave room for results.
MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT = 560, 520
SLIDER_LENGTH = 140
BANNER_WIDTH = 60
SCORE_SLIDER_STEP = 0.01
PAGE_SIZE_MIN, PAGE_SIZE_MAX, PAGE_SIZE_STEP = 10, 500, 10
CLOSE_POLL_MS = 200
INDEXING_BUSY_MESSAGE = (
    "Indexing is currently running.\n\n"
    "Please wait for it to finish, or press Stop indexing to save your partial cache and stop.\n\n"
    "(Your partial index is safe - stopping saves everything indexed so far.)")
MODEL_CHANGE_WHILE_INDEXING = ("The model and VRAM Save cannot be changed while indexing is running. "
                               "The other settings were saved; change them after indexing.")
NEEDS_IMAGES_MESSAGE = "Index the images of this library first."


class MainWindow:
    def __init__(self, root):
        self.root = root
        root.title(WINDOW_TITLE)
        root.geometry(WINDOW_GEOMETRY)
        root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        root.configure(bg=theme.BG)
        theme.apply_dark_title_bar(root)
        theme.apply_styles()

        self._preferences = Preferences.load()
        advanced_options.apply(self._preferences.advanced)
        self.library = MediaLibrary(self._preferences.model)
        self._engine = SearchEngine(self.library)
        self._topic_namer = TopicNamer()   # keeps the words' vectors between Explore windows
        self._explore = None               # the ExploreWindow open, if any
        self._indexing = IndexingController(self.library, IndexingReporter(self))
        self._result_actions = ResultActions(self.set_status)
        self._build(root)
        self._connect(root)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after_idle(self._start)

    def _build(self, root):
        """The window's widgets. Pack order matters: what is packed last is squeezed first when
        the window is small, so the results (packed last) give way, never the controls."""
        header = ttk.Frame(root, padding=(theme.PAD, theme.PAD, theme.PAD, 0))
        header.pack(side="top", fill="x")
        self._build_toolbar(header)
        self._search_bar = SearchBar(header, on_search=lambda: self._search.search_clicked(),
                                     on_image=lambda: self._search.image_clicked(),
                                     paste_picture=lambda: self._search.paste_picture())
        self._build_controls(header)
        self._filter_bar = FilterBar(header, lambda: self._search.search_again())
        self._status = StatusBar(root)
        body = tk.Frame(root, bg=theme.BG, padx=theme.PAD, pady=theme.PAD)
        body.pack(side="top", fill="both", expand=True)
        self.results = ResultsView(body, ResultsViewHooks(
            set_status=self.set_status,
            is_indexing=lambda: self._indexing.is_indexing,
            page_size=self._page_size.get,
            copy_selected=lambda: self._actions.copy(),
            move_selected=lambda: self._actions.move(),
            delete_selected=lambda: self._actions.delete(),
            open_result=self._result_actions.open,
            find_similar=lambda result, text: self._search.find_similar(result, text),
            typed_query=lambda: self._search_bar.query,
            copy_files=self._result_actions.copy_files,
            copy_picture=self._result_actions.copy_picture,
            thumbnail_size_changed=self._on_thumbnail_size_changed,
            text_of=lambda result: self.library.texts.text_of(result.key),
            refine=lambda: self._search.search_again(),
        ), self._preferences.thumbnail_size, drag_type=DND_FILES)

    def _connect(self, root):
        """The parts that do the work, each told what it needs of the others."""
        self._actions = SelectionActions(root, self.library, self.results, self.set_status, self.refresh_stats)
        self._encryption = EncryptionControl(root, self.library, lambda: self._preferences, self.set_status,
                                             lambda: self._libraries.load_saved_indexes())
        self._models = ModelControl(root, lambda: self._preferences, self._status, self.schedule,
                                    on_switch=self._switch_model, before_change=self._close_explore)
        self._search = SearchControl(self.library, self._engine, self.results, self._filter_bar, self._controls,
                                     self._models, self._status, SearchHooks(
                                         is_indexing=lambda: self._indexing.is_indexing,
                                         ensure_not_indexing=self._ensure_not_indexing,
                                         schedule=self.schedule,
                                         query=lambda: self._search_bar.query))
        self._libraries = LibraryControl(root, self.library, self._status, LibraryHooks(
            ensure_not_indexing=self._ensure_not_indexing,
            ensure_model_ready=self._models.ensure_ready,
            cancel_search=self._search.cancel,
            library_changed=self._library_changed,
            files_loaded=self._files_loaded,
            start_indexing=self._start_indexing,
            media_to_index=self._media_to_index,
            clear_query=self._search_bar.clear))
        if DND_FILES is not None:
            self.results.register_drop_target(DND_FILES, self._search.image_dropped)
        self._keys = KeyRouter(root, self._search_bar.entry, self.results, ConsoleWindow(root, OUTPUT),
                               self._result_actions.copy_files, self._search.paste_picture)

    def _start(self):
        """Once the window is up: unlock the index key (it may ask for a password), start
        loading the model, and open the index that was open last time."""
        self._encryption.start()
        self._models.start_with_available()
        scope = LastIndex.scope()
        if scope is not None:
            self._libraries.show_saved(scope)

    # ── construction ─────────────────────────────────────────────────────────
    def _build_toolbar(self, parent):
        bar = FlowFrame(parent)
        bar.pack(fill="x")

        library = bar.add(ControlGroup(bar))
        menu_button = ttk.Menubutton(library, text="Library")
        menu = tk.Menu(menu_button, tearoff=0, bg=theme.CARD_BG, fg=theme.FG,
                       activebackground=theme.ACCENT, activeforeground=theme.MENU_ACTIVE_FG)
        menu.add_command(label="New index…", command=lambda: self._libraries.new_index())
        menu.add_command(label="Open saved index…", command=lambda: self._libraries.open_saved())
        menu.add_command(label="Folders of this index…", command=lambda: self._libraries.edit_folders())
        menu.add_command(label="Statistics…", command=self._on_statistics_click)
        menu.add_separator()
        menu.add_command(label="Scan all drives…", command=lambda: self._libraries.scan_all_drives())
        menu.add_separator()
        menu.add_command(label="Index encryption…", command=lambda: self._encryption.open_dialog())
        menu_button["menu"] = menu
        library.add(menu_button)
        library.button("Refresh", self._on_refresh_click)
        self._preference_checkbox(library, "Read text (OCR)", "read_text", needs_model=OCR_KEY)
        self._preference_checkbox(library, "Index videos", "index_videos")

        indexing = bar.add(ControlGroup(bar))
        indexing.button("Stop indexing", self._on_stop_click, theme.DANGER_BUTTON)

        tools = bar.add(ControlGroup(bar))
        tools.button("Explore", self._on_explore_click)
        tools.button("Duplicates", self._on_duplicates_click)
        tools.button("Exclusions", self._on_exclusions_click)
        tools.button("Settings", self._on_settings_click)
        tools.button("Index info", lambda: show_index_info(self.library))

    def _build_controls(self, parent):
        bar = FlowFrame(parent)
        bar.pack(fill="x")

        min_score = tk.DoubleVar(value=settings.DEFAULT_MIN_SCORE)
        bar.add(LabeledSlider(bar, "Min. score", min_score, 0.0, 1.0, SCORE_SLIDER_STEP, SLIDER_LENGTH, "{:.2f}"))
        self._page_size = tk.IntVar(value=settings.DEFAULT_RESULTS_PER_PAGE)
        bar.add(LabeledSlider(bar, "Per page", self._page_size, PAGE_SIZE_MIN, PAGE_SIZE_MAX, PAGE_SIZE_STEP,
                              SLIDER_LENGTH, "{:d}"))

        kinds = bar.add(ControlGroup(bar))
        self._controls = SearchControls(min_score=min_score, include_images=kinds.checkbox("Images", True),
                                        include_videos=kinds.checkbox("Videos", True),
                                        best_frame_only=kinds.checkbox("Best frame per video", False))

        selection = bar.add(ControlGroup(bar))
        for label, command in SELECTION_COMMANDS:
            selection.button(label, lambda command=command: command(self.results))
        file_actions = bar.add(ControlGroup(bar))
        file_actions.button("Copy to…", lambda: self._actions.copy())
        file_actions.button("Move to…", lambda: self._actions.move())
        file_actions.button("Delete", lambda: self._actions.delete(), theme.DANGER_BUTTON)

        clearing = bar.add(ControlGroup(bar))
        clearing.button("Clear results", self._on_clear_click)

    def _preference_checkbox(self, group, label, name, needs_model=None):
        """A box for a true/false preference, saved as soon as it is changed. needs_model:
        ticking it first offers to download that model when it is missing."""
        def remember(value):
            variable.set(value)
            setattr(self._preferences, name, value)
            self._preferences.save()

        def changed():
            if variable.get() and needs_model and not self._models.is_available(needs_model):
                variable.set(False)
                self._models.offer_download(needs_model, on_ready=lambda key: remember(True), on_cancel=lambda: None)
                return
            remember(variable.get())
        variable = group.checkbox(label, getattr(self._preferences, name), changed)

    # ── services for the parts (also from other threads: schedule) ──────────
    def schedule(self, action, delay_ms=0):
        """Run `action` on the Tk thread. Safe from any thread, and ignored once the window is
        gone (e.g. during stop-and-close)."""
        try:
            self.root.after(delay_ms, action)
        except Exception:
            pass

    def set_status(self, text, colour=theme.FG):
        self._status.set_status(text, colour)

    def set_progress(self, percent, text):
        self._status.set_progress(percent, text)

    def refresh_stats(self):
        self._status.show_counts(self.library.counts())

    def repeat_search(self):
        self._search.repeat_after_indexing()

    def _library_changed(self):
        """Another index is open (or other folders, or none): what belonged to the old one goes."""
        self._close_explore()   # its topics are of the library being closed
        self._engine.invalidate()
        self._search.forget_results()
        self.refresh_stats()

    def _files_loaded(self):
        self._encryption.bring_files_in_line()
        self.refresh_stats()

    # ── indexing ─────────────────────────────────────────────────────────────
    def _ensure_not_indexing(self):
        """True if no indexing is running; otherwise tells the user to wait. Indexing is never
        stopped implicitly — only the Stop indexing button does that."""
        if self._indexing.is_indexing:
            messagebox.showinfo("Indexing in Progress", INDEXING_BUSY_MESSAGE)
            return False
        return True

    def _start_indexing(self, rebuild=False):
        """Index the images of the open library, then read the text on them when "Read text
        (OCR)" is ticked, then its videos when "Index videos" is ticked. rebuild=True starts
        from nothing instead of adding new files."""
        preferences = self._preferences
        passes = [IndexingPass(IMAGES, rebuild)]
        if preferences.read_text:
            passes.append(IndexingPass(TEXT, rebuild))
        if preferences.index_videos:
            passes.append(IndexingPass(VIDEOS, rebuild))
        if self._indexing.start(self._models.encoder, passes, preferences.indexing_options()):
            self.set_status("Indexing...", theme.STATUS_BUSY)

    def _media_to_index(self):
        """What a scan will look for, for the questions asked before it."""
        parts = (["images"] + (["the text on them"] if self._preferences.read_text else [])
                 + (["videos"] if self._preferences.index_videos else []))
        return ", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]

    def _on_refresh_click(self):
        if not self._ensure_not_indexing():
            return
        self._search.cancel()
        if not self._libraries.ensure_open() or not self._models.ensure_ready():
            return
        self._start_indexing()

    def _on_stop_click(self):
        if self._indexing.request_stop():
            self.set_status("Stopping... Please wait for file save.", theme.STATUS_STOPPED)

    # ── preferences ──────────────────────────────────────────────────────────
    def _on_thumbnail_size_changed(self, size):
        self._preferences.thumbnail_size = size
        self._preferences.save()

    def _on_settings_click(self):
        SettingsDialog(self.root, self._preferences, self._save_preferences)

    def _save_preferences(self, chosen):
        """From the Settings window; a model that is not on this computer is offered for
        download first (cancelled: the other settings are still saved)."""
        if self._models.is_available(chosen.model.key):
            self._apply_preferences(chosen)
            return
        previous = self._preferences
        unchanged = chosen.model == previous.model   # still the model that was never downloaded

        def downloaded(key):
            self._apply_preferences(chosen)
            if unchanged:
                self._models.start_loading()

        self._models.offer_download(chosen.model.key, on_ready=downloaded, on_cancel=lambda: self._apply_preferences(
            chosen if unchanged else replace(chosen, model=previous.model)))

    def _apply_preferences(self, chosen):
        previous = self._preferences
        reload_needed = chosen.model != previous.model or chosen.vram_save != previous.vram_save
        if reload_needed and self._indexing.is_indexing:
            messagebox.showinfo("Indexing in Progress", MODEL_CHANGE_WHILE_INDEXING)
            chosen.model, chosen.vram_save = previous.model, previous.vram_save
            reload_needed = False
        self._preferences = chosen
        chosen.save()
        advanced_options.apply(chosen.advanced)
        if chosen.advanced.get("GPU_SEARCH") != previous.advanced.get("GPU_SEARCH"):
            self._engine.invalidate()   # the index moves to or from the graphics card
        if chosen.model != previous.model:
            self._switch_model()
        elif reload_needed:
            self._models.start_loading()   # the same model, its weights in the other precision
        else:
            self.set_status("Settings saved", theme.STATUS_OK)

    def _switch_model(self):
        """Use the newly chosen model: its own indexes for the open library, and the model
        itself (loaded in the background)."""
        self._search.forget_results()
        self.library.use_model(self._preferences.model)
        self._engine.invalidate()
        if self.library.is_open:
            self._libraries.load_saved_indexes()
        self.refresh_stats()
        self._models.start_loading()

    # ── tools ────────────────────────────────────────────────────────────────
    def _on_clear_click(self):
        self._search.forget_results()
        self.set_status("Results cleared", theme.STATUS_OK)

    def _on_exclusions_click(self):
        ExclusionsDialog(self.root, self.library.exclusions, self.library.scope)

    def _on_duplicates_click(self):
        if not self._ensure_not_indexing() or not self._libraries.ensure_open():
            return
        if not self.library.images.has_data:
            messagebox.showinfo("Find duplicates", NEEDS_IMAGES_MESSAGE)
            return
        DuplicatesDialog(self.root, self.library.model, self._search.find_duplicates)

    def _close_explore(self):
        if self._explore is not None and self._explore.is_open:
            self._explore.close()
        self._explore = None

    def _on_explore_click(self):
        """The pictures (those passing the filter bar) grouped into topics, in a window of their own."""
        if not self._libraries.ensure_open() or not self._models.ensure_ready():
            return
        if not self.library.images.has_data:
            messagebox.showinfo("Explore", NEEDS_IMAGES_MESSAGE)
            return
        prepared = self._search.prepared()
        if prepared is None:
            return
        encoder, images = self._models.encoder, self.library.images.view()
        allowed = self._engine.allowed_rows(encoder, images, prepared.options)
        pictures = len(images.keys) if allowed is None else int(allowed.sum())
        finder = TopicFinder(self.library, self._topic_namer)
        to_abs = self.library.scope.absolute_path
        self._close_explore()
        self._explore = ExploreWindow(self.root, f"Explore — {self.library.scope.description}", ExploreHooks(
            find=lambda within: finder.find(encoder, allowed, within, encoder.search_device),
            show=self._search.show_topic,
            picture=lambda key, size: make_thumbnail(SearchResult(0.0, to_abs(key), MediaKind.IMAGE), size),
            pictures=pictures))

    def _on_statistics_click(self):
        if self._libraries.ensure_open():
            StatisticsWindow(self.root, f"Statistics — {self.library.scope.description}", self._collect_statistics)

    def _collect_statistics(self, should_stop):
        """Worker thread. The kinds of pictures and the duplicates need the model; the
        duplicates are not compared while indexing changes the index."""
        library, encoder = self.library, self._models.encoder
        if encoder is None or not library.images.has_data:
            return StatisticsCollector(library).collect(should_stop=should_stop)
        return StatisticsCollector(library).collect(
            lambda view: self._engine.media_types_of(encoder, view), encoder.search_device,
            count_duplicates=not self._indexing.is_indexing, should_stop=should_stop)

    # ── exit ─────────────────────────────────────────────────────────────────
    def _quit(self):
        try:
            self.root.destroy()
        except Exception:
            pass
        os._exit(0)   # worker threads may be inside native code; do not wait for them

    def _on_close(self):
        if not self._indexing.is_indexing:
            self._quit()
            return
        CloseWhileIndexingDialog(self.root, self._stop_save_and_quit, lambda: os._exit(0))

    def _stop_save_and_quit(self):
        self._indexing.request_stop()
        self.set_status("Stopping & saving before exit...", theme.ORANGE)
        safe_print("\n[CLOSE] Stop & save requested.")
        self._quit_when_indexing_ends()

    def _quit_when_indexing_ends(self):
        if self._indexing.is_indexing:
            self.schedule(self._quit_when_indexing_ends, CLOSE_POLL_MS)
        else:
            self._quit()


def main():
    capture()   # for the console window (` key)
    print("=" * BANNER_WIDTH)
    print("Vibekimus - AI Media Search")
    print("=" * BANNER_WIDTH)
    root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
    theme.apply_icon(root)
    MainWindow(root)
    root.mainloop()
