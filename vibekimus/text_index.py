"""The text read from the library's pictures (OCR), one entry per image file.

Kept in its own file next to the image index. It does not depend on the search
model, so the same text serves every model. Pictures without text are stored with
an empty text, so they are not read again. The file holds JSON as bytes in an .npz
and is loaded with allow_pickle=False (and encrypted), like the indexes."""
import os
import threading

from vibekimus import settings
from vibekimus.index_files import json_bytes, json_value, open_npz, save_npz_atomically
from vibekimus.runlog import safe_print
from vibekimus.text_matching import normalize

FORMAT_VERSION = 1
KIND = "text"


class TextIndex:
    """Written by the text-reading pass while searches read it: every change and every
    walk over the entries holds one lock (a search walking the texts while a picture's
    text is added failed with "dictionary changed size during iteration")."""

    def __init__(self):
        self.path = None
        self._texts = {}         # index path -> text as read (lines joined by newlines)
        self._normalized = None  # index path -> normalize(text) of files with text, made when first searched
        self._lock = threading.RLock()

    def __len__(self):
        return len(self._texts)

    @property
    def has_data(self):
        return bool(self._texts)

    def with_text(self):
        """Number of files on which some text was found."""
        with self._lock:
            return sum(1 for text in self._texts.values() if text)

    def keys(self):
        with self._lock:
            return set(self._texts)

    def text_of(self, key):
        return self._texts.get(key, "")

    def normalized_texts(self):
        """{index path: normalised text} of the files with text — a copy of its own, so a
        search can go through it while text is still being read."""
        with self._lock:
            if self._normalized is None:
                self._normalized = {key: normalize(text) for key, text in self._texts.items() if text}
            return dict(self._normalized)

    # ── changes ──────────────────────────────────────────────────────────────
    def open(self, data_dir):
        """Use the text file of a library folder; empty until load()."""
        self.path = os.path.join(data_dir, settings.TEXT_INDEX_FILE)
        self.clear()

    def clear(self):
        with self._lock:
            self._texts = {}
            self._normalized = None

    def set(self, key, text):
        with self._lock:
            self._texts[key] = text
            if self._normalized is not None:   # kept up to date instead of made again for every picture
                if text:
                    self._normalized[key] = normalize(text)
                else:
                    self._normalized.pop(key, None)

    def keep_only(self, keys):
        """Drop the entries of files no longer in the image index. Returns how many."""
        with self._lock:
            stale = [key for key in self._texts if key not in keys]
            for key in stale:
                del self._texts[key]
                if self._normalized is not None:
                    self._normalized.pop(key, None)
            return len(stale)

    # ── disk ─────────────────────────────────────────────────────────────────
    def exists(self):
        return self.path is not None and os.path.exists(self.path)

    def load(self, validate):
        """Read the file; validate(keys) refuses paths that point outside the library."""
        with open_npz(self.path) as archive:
            meta = json_value(archive["meta"])
            keys, texts = json_value(archive["keys"]), json_value(archive["texts"])
        if (not isinstance(meta, dict) or meta.get("kind") != KIND
                or not isinstance(keys, list) or not isinstance(texts, list) or len(keys) != len(texts)
                or not all(isinstance(item, str) for item in keys + texts)):
            raise ValueError("the text file is damaged")
        validate(keys)
        with self._lock:
            self._texts = dict(zip(keys, texts))
            self._normalized = None
        safe_print(f"[TEXT] Loaded text of {len(self._texts):,} images ({self.with_text():,} with text)")

    def save(self):
        with self._lock:
            keys = list(self._texts)
            texts = [self._texts[key] for key in keys]
        save_npz_atomically(self.path, meta=json_bytes({"version": FORMAT_VERSION, "kind": KIND}),
                            keys=json_bytes(keys), texts=json_bytes(texts))
