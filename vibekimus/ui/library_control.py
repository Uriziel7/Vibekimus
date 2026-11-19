"""Which index the main window has open: a new index of chosen folders, a saved one, the
all-drives scan, the folders of the open index, and loading its files."""
from dataclasses import dataclass
from tkinter import filedialog, messagebox
from typing import Callable

from vibekimus import settings
from vibekimus.encryption import LockedError
from vibekimus.index_migration import move_folder_index
from vibekimus.last_index import LastIndex
from vibekimus.library import IncompatibleIndex
from vibekimus.runlog import log_event, safe_print
from vibekimus.saved_indexes import SavedIndexes, label_of
from vibekimus.scanner import detect_drive_roots
from vibekimus.scope import AllDrivesScope, same_or_inside
from vibekimus.ui import theme
from vibekimus.ui.folders_dialog import FoldersDialog
from vibekimus.ui.saved_indexes_dialog import SavedIndexesDialog

BANNER_WIDTH = 60
INDEX_DIALOG_TYPES = [("Vibekimus index", "*" + settings.INDEX_FILE_SUFFIX)]
NO_LIBRARY_MESSAGE = "Create or open an index first (Library menu)."
NEW_INDEX_INTRO = ("Pick the folders to index (each with its subfolders). The index is kept in the program's "
                   "data; nothing is written to these folders. More folders can be added later.")
FOLDERS_INTRO = ("Folders added here are indexed right after Save (only what is new); the entries of a "
                 "removed folder leave the index.")
FOLDERS_TITLE = "Folders of this index"
MOVE_INDEX_QUESTION = ("This index is kept inside the folder\n\n{folder}\n\nTo add folders to it, it is copied to "
                       "the program's data first: everything indexed comes along (every model, the text read from "
                       "pictures, the exclusions) and nothing is indexed again. The files in the folder are left "
                       "as they are.\n\nCopy it now?")
REMOVE_FOLDERS_QUESTION = "Remove these folders, and what was indexed in them, from the index?\n\n{folders}"
FOLDERS_NOT_EDITABLE = "The all-drives index covers every drive. Use Library → New index… for chosen folders."
SYSTEM_FOLDERS_SKIPPED_NOTE = "\n\nWindows, program and app-data folders are skipped."
ALL_DRIVES_QUESTION = ("Index all {media} on:\n\n    {drives}{note}\n\nFiles already in the index are not processed "
                       "again, so you can stop at any time and continue later.\n\nStart now?")
DRIVE_SEPARATOR = "   "


@dataclass(frozen=True)
class LibraryHooks:
    """What opening an index needs from the main window."""
    ensure_not_indexing: Callable   # () -> bool; tells the user when indexing runs
    ensure_model_ready: Callable    # () -> bool; tells the user when the model is not loaded
    cancel_search: Callable         # ()
    library_changed: Callable       # () — another index, other folders, or none: what belonged to the old goes
    files_loaded: Callable          # () — the index files were loaded: counts, encryption brought in line
    start_indexing: Callable        # (rebuild)
    media_to_index: Callable        # () -> "images, the text on them and videos"
    clear_query: Callable           # ()


class LibraryControl:
    def __init__(self, root, library, status, hooks):
        self._root, self._library, self._status, self._hooks = root, library, status, hooks

    def ensure_open(self):
        """Whether an index is open; otherwise tells the user."""
        if not self._library.is_open:
            messagebox.showwarning("Warning", NO_LIBRARY_MESSAGE)
            return False
        return True

    def _ready_to_change(self):
        if not self._hooks.ensure_not_indexing():
            return False
        self._hooks.cancel_search()
        return self._hooks.ensure_model_ready()

    def _saved_indexes(self):
        return SavedIndexes(self._library.model)

    # ── opening ──────────────────────────────────────────────────────────────
    def open_scope(self, scope):
        """Make `scope` the open library (its index files are not loaded yet)."""
        self._library.open(scope)
        LastIndex.remember(scope)
        self._hooks.library_changed()
        safe_print(f"\n{'=' * BANNER_WIDTH}\n[LIBRARY] {scope.description}")

    def show_saved(self, scope):
        """Open a library that has index files and load them."""
        self.open_scope(scope)
        if scope.kept_in_indexed_folder:   # the list of saved indexes shows it from now on
            self._saved_indexes().remember(scope.data_dir)
        self.load_saved_indexes()
        self._show_loaded_status()

    def load_saved_indexes(self):
        """Load whatever index files exist for the open library. Returns False if there are none at
        all. Encrypted files that cannot be unlocked close the library, so nothing can overwrite them."""
        library = self._library
        has_images, has_videos = library.has_saved_images(), library.has_saved_videos()
        try:
            self._load_indexes(has_images, has_videos)
        except LockedError as error:
            library.close()
            self._hooks.library_changed()
            messagebox.showerror("Encrypted index", str(error))
            self._status.set_status("Encrypted index: unlock or import its key (Library → Index encryption)",
                                    theme.STATUS_ERROR)
        else:
            self._hooks.files_loaded()
        return has_images or has_videos

    def _load_indexes(self, has_images, has_videos):
        library = self._library
        if has_images:
            try:
                self._status.set_status("Loading cache from disk...", theme.STATUS_BUSY)
                library.load_images()
            except LockedError:
                raise
            except Exception as error:
                messagebox.showerror("Error", f"Load failed: {error}")
                self._status.set_status("Cache load failed", theme.STATUS_ERROR)
        if has_videos:
            library.load_videos()
        library.load_texts()

    def _show_loaded_status(self):
        counts = self._library.counts()
        if counts.images and counts.videos:
            self._status.set_status(f"Loaded {counts.images:,} images + {counts.videos:,} videos", theme.STATUS_OK)
        elif counts.videos:
            self._status.set_status(f"Loaded video cache: {counts.videos:,} videos", theme.STATUS_OK)
        elif counts.images:
            self._status.set_status(f"Loaded {counts.images:,} images", theme.STATUS_OK)

    # ── Library menu ─────────────────────────────────────────────────────────
    def new_index(self):
        """A new index of one or more folders, indexed at once."""
        if self._ready_to_change():
            FoldersDialog(self._root, "New index", NEW_INDEX_INTRO, [], "Create and index", self._create_index,
                          name="")

    def _create_index(self, name, folders):
        self.open_scope(SavedIndexes.create(name or label_of(folders[0]), folders))
        self._hooks.clear_query()
        self._hooks.start_indexing(rebuild=True)

    def open_saved(self):
        """The indexes saved for the selected model, to search one without scanning."""
        if self._ready_to_change():
            SavedIndexesDialog(self._root, self._saved_indexes(), self._library.model.label, self._open_listed,
                               self._browse_saved_index)

    def _open_listed(self, saved):
        try:
            scope = saved.scope()
        except Exception as error:   # damaged list of folders, or encrypted with a locked key
            messagebox.showerror("Cannot open this index", str(error))
            return
        self.show_saved(scope)

    def _browse_saved_index(self):
        """Any index file, chosen in the classic dialog."""
        index_file = filedialog.askopenfilename(title="Open saved index", filetypes=INDEX_DIALOG_TYPES,
                                                initialdir=settings.ALL_DRIVES_INDEX_DIR)
        if not index_file:
            return
        try:
            scope = self._library.scope_of_saved_index(index_file)
        except IncompatibleIndex as error:
            messagebox.showerror("Cannot open this index", str(error))
            return
        self.show_saved(scope)

    def edit_folders(self):
        """Add folders to the open index, or remove some."""
        if not self._hooks.ensure_not_indexing() or not self.ensure_open():
            return
        scope = self._library.scope
        if not scope.folders_editable:
            messagebox.showinfo(FOLDERS_TITLE, FOLDERS_NOT_EDITABLE)
            return
        if scope.kept_in_indexed_folder:
            scope = self._moved_to_program_data(scope)
            if scope is None:
                return
        FoldersDialog(self._root, f"Folders of {scope.name}", FOLDERS_INTRO, scope.scan_roots, "Save",
                      lambda name, folders: self._change_folders(folders))

    def _moved_to_program_data(self, folder_scope):
        """An index kept inside its folder, copied into an index of the program's data (and opened), or None."""
        folder = folder_scope.data_dir
        if not messagebox.askyesno(FOLDERS_TITLE, MOVE_INDEX_QUESTION.format(folder=folder)):
            return None
        try:
            scope = move_folder_index(folder, label_of(folder))
        except LockedError as error:
            messagebox.showerror("Encrypted index", str(error))
            return None
        except Exception as error:   # disk full, a damaged file…
            messagebox.showerror(FOLDERS_TITLE, f"The index could not be copied:\n\n{error}")
            return None
        self.show_saved(scope)
        return scope

    def _change_folders(self, folders):
        library = self._library
        removed = [root for root in library.scope.scan_roots if not any(same_or_inside(root, f) for f in folders)]
        if removed and not messagebox.askyesno(FOLDERS_TITLE,
                                               REMOVE_FOLDERS_QUESTION.format(folders="\n".join(removed))):
            return
        try:
            for folder in removed:
                library.remove_folder(folder)
            added = [folder for folder in folders if library.add_folder(folder)]
        except Exception as error:   # IndexSaveError, or the list of folders could not be written
            messagebox.showerror(FOLDERS_TITLE, f"The folders could not be changed:\n\n{error}")
            return
        self._hooks.library_changed()
        if added:
            self._hooks.start_indexing(rebuild=False)
        else:
            self._status.set_status("Folders of the index saved", theme.STATUS_OK)

    def scan_all_drives(self):
        if not self._hooks.ensure_not_indexing() or not self._hooks.ensure_model_ready():
            return
        roots = detect_drive_roots()
        if not roots:
            messagebox.showwarning("No Drives", "No local drives were found.")
            return
        note = "" if settings.ALL_DRIVES_SCAN_SYSTEM_FOLDERS else SYSTEM_FOLDERS_SKIPPED_NOTE
        question = ALL_DRIVES_QUESTION.format(media=self._hooks.media_to_index(), drives=DRIVE_SEPARATOR.join(roots),
                                              note=note)
        if not messagebox.askyesno("Scan All Drives", question):
            return
        self._hooks.cancel_search()
        scope = AllDrivesScope(roots)
        self.open_scope(scope)
        safe_print(f"[ALL DRIVES] Index folder: {scope.data_dir}")
        log_event(f"ALL DRIVES scan of {', '.join(roots)}")
        self.load_saved_indexes()
        self._hooks.clear_query()
        self._hooks.start_indexing(rebuild=False)
