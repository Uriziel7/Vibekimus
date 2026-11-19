"""The media library: a scope, its exclusions and the image and video indexes."""
import os
from dataclasses import dataclass
from datetime import datetime

from vibekimus import settings
from vibekimus.embedding_index import EmbeddingIndex
from vibekimus.encoders.catalog import model_for_meta
from vibekimus.encryption import LockedError, active_vault, is_sealed
from vibekimus.exclusions import Exclusions
from vibekimus.index_files import ImageIndexFormat, VideoIndexFormat, index_exists, peek_index_meta
from vibekimus.runlog import safe_print
from vibekimus.scanner import MediaScanner
from vibekimus.scope import in_folder, scope_of_index_folder
from vibekimus.text_index import TextIndex

SCAN_PROGRESS_EVERY = 2000   # report the running count after this many files found


class IncompatibleIndex(ValueError):
    """The chosen file is not an index the current model can search."""


@dataclass(frozen=True)
class LibraryCounts:
    images: int
    videos: int
    frames: int
    texts_read: int = 0   # pictures whose text has been read (OCR)…
    with_text: int = 0    # …and those on which text was found


class MediaLibrary:
    """Everything known about the indexed media of the currently opened scope, for
    one model (each model has its own index files; the text read from pictures is
    shared by all models)."""

    def __init__(self, model):
        self.scope = None
        self.exclusions = Exclusions()
        self.texts = TextIndex()
        self.use_model(model)

    def use_model(self, model):
        """Switch to the indexes of another model. The open scope stays open; load
        its indexes for the new model with load_images() / load_videos()."""
        self.model = model
        self.images = EmbeddingIndex(ImageIndexFormat(model))
        self.videos = EmbeddingIndex(VideoIndexFormat(model))
        if self.scope is not None:
            self.open(self.scope)

    # ── opening ──────────────────────────────────────────────────────────────
    @property
    def is_open(self):
        return self.scope is not None

    @property
    def is_searchable(self):
        """Open, with images or video frames to search (also ones an indexing run is adding)."""
        return self.is_open and (self.images.is_searchable or self.videos.is_searchable)

    def open(self, scope):
        """Switch to another scope. The indexes start empty; load them with
        load_images() / load_videos()."""
        scope.prepare()
        self.scope = scope
        for index in (self.images, self.videos):
            index.clear()
            index.path = os.path.join(scope.data_dir, index.format.file_name())
        self.texts.open(scope.data_dir)
        self.exclusions.load(scope.data_dir)

    def open_saved_index(self, index_file):
        """Open the library a saved index file belongs to, without scanning anything.
        Load its contents with load_images() / load_videos(). Raises IncompatibleIndex."""
        self.open(self.scope_of_saved_index(index_file))

    def scope_of_saved_index(self, index_file):
        """The library a saved index file belongs to: the all-drives index, an index of
        several folders, or the folder the file lies in. Raises IncompatibleIndex."""
        try:
            meta = peek_index_meta(index_file)
        except Exception as error:
            raise IncompatibleIndex(f"The index file cannot be read: {error}") from error
        made_with = model_for_meta(meta)
        if made_with is None:
            raise IncompatibleIndex(
                f"This index was made with a model this program does not offer "
                f"({meta.get('model')} / {meta.get('pretrained')}). Index the folder again.")
        if made_with != self.model:
            raise IncompatibleIndex(
                f"This index was made with {made_with.label}, but {self.model.label} is selected.\n\n"
                f"Choose {made_with.label} in Settings to search it.")
        names = {index.format.file_name() for index in (self.images, self.videos)}
        if os.path.basename(index_file) not in names:
            raise IncompatibleIndex(
                f"This file has been renamed; the program looks for its indexes by name.\n\n"
                f"Index files of {self.model.label} are named like:\n{self.images.format.file_name()}")
        return scope_of_index_folder(os.path.dirname(os.path.abspath(index_file)))

    def has_saved_images(self):
        return index_exists(self.images.path)

    def has_saved_videos(self):
        return index_exists(self.videos.path)

    def load_images(self):
        """Load the saved image index. Raises if the file is damaged or unsafe."""
        safe_print(f"[CACHE] Loading: {self.images.path}")
        self.images.load(self.images.path, self.scope.validate)
        safe_print(f"[CACHE] Success. {len(self.images.keys):,} images.")

    def load_videos(self):
        """Load the saved video index; a damaged one is reported and left empty."""
        safe_print(f"[VCACHE] Loading: {self.videos.path}")
        try:
            self.videos.load(self.videos.path, self.scope.validate)
        except LockedError:
            raise   # not damaged: it must not be replaced by a new, empty index
        except Exception as e:
            safe_print(f"[VCACHE] Load error: {e}")
            self.videos.clear()
            return
        counts = self.counts()
        safe_print(f"[VCACHE] Loaded {counts.frames:,} frames from {counts.videos:,} videos.")

    def load_texts(self):
        """Load the text read from pictures, if any; a damaged file is reported and
        left empty (the text is read again)."""
        if not self.texts.exists():
            return
        try:
            self.texts.load(self.scope.validate)
        except LockedError:
            raise
        except Exception as e:
            safe_print(f"[TEXT] Load error: {e}")
            self.texts.clear()

    def close(self):
        """No library open (e.g. its files could not be decrypted)."""
        self.scope = None
        for index in (self.images, self.videos):
            index.clear()
        self.texts.clear()

    # ── encryption ───────────────────────────────────────────────────────────
    def _saved_files(self):
        candidates = [self.images.path, self.videos.path, self.texts.path,
                      os.path.join(self.scope.data_dir, settings.EXCLUSIONS_FILE),
                      os.path.join(self.scope.data_dir, settings.LIBRARY_FILE)]
        return [path for path in candidates if path and os.path.exists(path)]

    def needs_rewrite(self):
        """Whether some saved file is not in the form the vault writes now
        (plain while encryption is on, or encrypted while it is off)."""
        sealed = active_vault().enabled
        return any(is_sealed(path) != sealed for path in self._saved_files())

    def rewrite_files(self):
        """Save every loaded file again: encrypted or plain, as the vault writes now. With
        encryption on, the logs of unreadable files (readable, and naming files) are removed.
        Raises embedding_index.IndexSaveError."""
        if active_vault().enabled:
            self._remove_skipped_logs()
        for index in (self.images, self.videos):
            if index.has_data:
                index.save()
        if self.texts.has_data:
            self.texts.save()
        if self.exclusions.is_readable:
            self.exclusions.save()
        self.scope.prepare()   # the index's name and folders, when it keeps them

    # ── contents ─────────────────────────────────────────────────────────────
    def counts(self):
        return LibraryCounts(
            images=len(self.images.keys) if self.images.has_data else 0,
            videos=len(self.videos.files()) if self.videos.has_data else 0,
            frames=len(self.videos.keys) if self.videos.has_data else 0,
            texts_read=len(self.texts),
            with_text=self.texts.with_text(),
        )

    def iter_images(self, should_stop=None, on_progress=None):
        """Yield (abs_path, index_path) for every image in scope, exclusions applied."""
        return self._iter_media(settings.IMAGE_EXTS, True, should_stop, on_progress)

    def iter_videos(self, should_stop=None, on_progress=None):
        """Yield (abs_path, index_path) for every video in scope, exclusions applied."""
        return self._iter_media(settings.VIDEO_EXTS, False, should_stop, on_progress)

    def _iter_media(self, extensions, skip_small_files, should_stop, on_progress):
        scanner = MediaScanner(self.scope.scan_rules(skip_small_files), should_stop)
        found = 0
        for abs_path in scanner.files(self.scope.scan_roots, extensions):
            index_path = self.scope.index_path(abs_path)
            if self.exclusions.matches(index_path):
                continue
            found += 1
            if on_progress and found % SCAN_PROGRESS_EVERY == 0:
                on_progress(found)
            yield abs_path, index_path

    # ── changes ──────────────────────────────────────────────────────────────
    def add_folder(self, folder):
        """Index one more folder (a library of several folders only). Returns False when
        the library already covers it; index it afterwards (it is not scanned here)."""
        return self.scope.add_root(folder)

    def remove_folder(self, folder):
        """Stop indexing a folder of a library of several folders, and drop its entries."""
        prefix = self.scope.folder_prefix(folder)
        if prefix is not None:
            known = set(self.images.keys) | self.videos.files() | self.texts.keys()
            self._forget_index_paths({path for path in known if in_folder(path, prefix)})
        self.scope.remove_root(folder)

    def forget_files(self, abs_paths):
        """Drop files that were moved away or deleted from both indexes and save."""
        index_paths = set()
        for path in abs_paths:
            try:
                index_paths.add(self.scope.index_path(path))
            except ValueError:
                pass   # Windows: a path on another drive has no relative form
        self._forget_index_paths(index_paths)

    def _forget_index_paths(self, index_paths):
        """Raises embedding_index.IndexSaveError when an index cannot be saved."""
        if not index_paths:
            return
        if self.images.remove_files(index_paths):
            self.images.save()
        if self.videos.remove_files(index_paths):
            self.videos.save()
        if self.texts.has_data and self.texts.keep_only(self.texts.keys() - index_paths):
            self.texts.save()

    def _remove_skipped_logs(self):
        for name in (settings.SKIPPED_IMAGES_LOG, settings.SKIPPED_VIDEOS_LOG, settings.SKIPPED_TEXT_LOG):
            try:
                os.remove(os.path.join(self.scope.data_dir, name))
            except FileNotFoundError:
                pass
            except OSError as error:
                safe_print(f"[LOG] Could not remove {name}: {error}")

    def write_skipped_log(self, failures, log_name):
        """Append this run's unreadable files — (path, reason) pairs — to a log in the
        data folder. Each run is separated by a dated header. Not written while index
        encryption is on: the log is meant to be read in any editor, and it names files."""
        unique = list(dict(failures).items())
        if not unique or active_vault().enabled:
            return
        try:
            with open(os.path.join(self.scope.data_dir, log_name), "a", encoding="utf-8") as f:
                f.write(f"\n=== Index run: {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
                for path, reason in unique:
                    f.write(f"{os.path.basename(path)} — {reason}\n")
                f.write(f"=== {len(unique)} failed ===\n")
            safe_print(f"[LOG] Wrote {len(unique)} failed entries to {log_name}")
        except Exception as e:
            safe_print(f"[LOG] Could not write failed log: {e}")
