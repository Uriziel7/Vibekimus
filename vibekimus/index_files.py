"""On-disk index format.

An index is an uncompressed .npz: float16 embeddings plus JSON metadata stored as
bytes. It is always loaded with allow_pickle=False, so opening an index file can
never execute code. A long run also leaves small ".partNNNNN" pieces next to the
main file (see EmbeddingIndex.save); they are merged on the next complete save.
"""
import glob
import io
import json
import os
from abc import ABC, abstractmethod

import numpy as np

from vibekimus import settings
from vibekimus.encryption import open_read, write_file_atomically
from vibekimus.files import forward_slashes
from vibekimus.media_info import INFO_DTYPE, unknown_info
from vibekimus.runlog import safe_print

FORMAT_VERSION = 1
SHARD_INFIX = ".part"
SHARD_DIGITS = 5
FIRST_SHARD_NUMBER = 1
RESERVED_ARRAYS = ("meta", "count")
NOT_MEDIA_MESSAGE = "the {kind} index lists a file that is not of that kind"


def json_bytes(value):
    """A JSON value as a uint8 array, to be stored in an .npz without pickling."""
    return np.frombuffer(json.dumps(value, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)


def json_value(array):
    """Inverse of json_bytes."""
    return json.loads(array.tobytes().decode("utf-8"))


def save_npz_atomically(path, **arrays):
    """Written to a temp file first and swapped in, so a crash mid-save never leaves
    a truncated file behind; encrypted while index encryption is on."""
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    write_file_atomically(path, buffer.getvalue())


def open_npz(path):
    """An index-like file for reading (decrypted when it is encrypted). Never unpickles."""
    return np.load(open_read(path), allow_pickle=False)


def save_index_file(path, embeddings, meta, **arrays):
    """Write one index file (atomically)."""
    save_npz_atomically(
        path,
        emb=np.ascontiguousarray(embeddings, dtype=np.float16),
        meta=json_bytes(meta),
        count=np.array(len(embeddings), dtype=np.int64),
        **arrays,
    )


def load_index_file(path):
    """Read one index file. Never unpickles anything.

    Returns (meta_dict, {name: array}); embeddings come back as float32.
    """
    with open_npz(path) as archive:
        meta = _read_meta(archive)
        arrays = {name: archive[name] for name in archive.files if name not in RESERVED_ARRAYS}
    emb = arrays.get("emb")
    if emb is None or emb.ndim != 2:
        raise ValueError("index has no embedding matrix")
    arrays["emb"] = emb.astype(np.float32)
    return meta, arrays


def peek_index_meta(path):
    """The metadata of an index file, without loading the embeddings."""
    with open_npz(path) as archive:
        return _read_meta(archive)


def _read_meta(archive):
    meta = json_value(archive["meta"])
    if not isinstance(meta, dict):
        raise ValueError("index metadata is not an object")
    return meta


def peek_index_count(path):
    """Number of entries in an index file, without loading the embeddings."""
    with open_npz(path) as archive:
        return int(archive["count"])


def index_shards(path):
    """Partial-save pieces that belong to an index file, oldest first."""
    pattern = glob.escape(path) + SHARD_INFIX + "[0-9]" * SHARD_DIGITS
    return sorted(glob.glob(pattern))


def next_shard_path(path):
    shards = index_shards(path)
    number = int(shards[-1][-SHARD_DIGITS:]) + 1 if shards else FIRST_SHARD_NUMBER
    return f"{path}{SHARD_INFIX}{number:0{SHARD_DIGITS}d}"


def remove_shards(path):
    for shard in index_shards(path):
        try:
            os.remove(shard)
        except OSError:
            pass


def index_exists(path):
    return os.path.exists(path) or bool(index_shards(path))


def _is_string_list(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


class IndexFormat(ABC):
    """How one kind of index (images or video frames) maps its keys to a file."""

    kind = ""          # stored in the file's metadata
    name_infix = ""    # distinguishes the file names of the two kinds
    log_tag = ""       # console prefix
    entry_noun = ""    # what one row is called in messages

    def __init__(self, model):
        self.model = model   # the encoders.catalog.ModelSpec whose embeddings the index holds

    def file_name(self):
        """The model is part of the name, so an index built with a different model
        is never mixed up with this one."""
        return (f"{settings.INDEX_FILE_PREFIX}{self.name_infix}{self.model.index_label}"
                f"{settings.INDEX_FILE_SUFFIX}")

    def write(self, path, keys, embeddings, info):
        key_meta, key_arrays = self._pack_keys(keys)
        meta = {"version": FORMAT_VERSION, "kind": self.kind, "model": self.model.name,
                "pretrained": self.model.pretrained, **key_meta}
        save_index_file(path, embeddings, meta, info=np.ascontiguousarray(info, dtype=INFO_DTYPE), **key_arrays)

    def read(self, path):
        """Returns (keys, embeddings, info) from one index file. Files written before
        the info records existed get UNKNOWN records."""
        meta, arrays = load_index_file(path)
        keys = self._unpack_keys(meta, arrays)
        self._refuse_other_files(keys)
        info = arrays.get("info")
        if info is None or info.dtype != INFO_DTYPE or len(info) != len(keys):
            info = unknown_info(len(keys))
        return keys, arrays["emb"], info

    def read_merged(self, path):
        """Read the main index file plus any partial-save pieces left by an interrupted
        run, as one (keys, embeddings, info) triple. An entry present twice is kept once."""
        files = ([path] if os.path.exists(path) else []) + index_shards(path)
        if not files:
            raise FileNotFoundError(path)
        keys, embedding_parts, info_parts, seen = [], [], [], set()
        for file_path in files:
            try:
                file_keys, file_emb, file_info = self.read(file_path)
            except Exception as err:
                if file_path == path:
                    raise
                safe_print(f"[CACHE] Skipping unreadable piece {os.path.basename(file_path)}: {err}")
                continue
            fresh = []
            for i, key in enumerate(file_keys):
                if key not in seen:
                    seen.add(key)
                    fresh.append(i)
            if len(fresh) != len(file_keys):
                file_keys = [file_keys[i] for i in fresh]
                file_emb, file_info = file_emb[fresh], file_info[fresh]
            if file_keys:
                keys.extend(file_keys)
                embedding_parts.append(file_emb)
                info_parts.append(file_info)
        if not embedding_parts:
            raise ValueError("index is empty")
        return keys, np.concatenate(embedding_parts, axis=0), np.concatenate(info_parts)

    def _refuse_other_files(self, keys):
        """Only pictures (videos) are ever indexed. A file of any other kind in an index was
        put there by someone: opening that "result" would run it (a .exe, .bat, .lnk…)."""
        extensions = self.media_extensions()
        if not all(self.file_of(key).lower().endswith(extensions) for key in keys):
            raise ValueError(NOT_MEDIA_MESSAGE.format(kind=self.kind))

    @abstractmethod
    def media_extensions(self):
        """The file extensions this kind of index holds (lower case)."""

    @abstractmethod
    def file_of(self, key):
        """The media file an index key belongs to."""

    @abstractmethod
    def _pack_keys(self, keys):
        """Returns (metadata fields, extra arrays) describing the keys."""

    @abstractmethod
    def _unpack_keys(self, meta, arrays):
        """Inverse of _pack_keys; raises ValueError for an inconsistent file."""


class ImageIndexFormat(IndexFormat):
    """One row per image; the key is the image's index path."""

    kind = "images"
    log_tag = "CACHE"
    entry_noun = "images"

    def media_extensions(self):
        return settings.IMAGE_EXTS

    def file_of(self, key):
        return key

    def _pack_keys(self, keys):
        return {"paths": [forward_slashes(p) for p in keys]}, {}

    def _unpack_keys(self, meta, arrays):
        paths = meta.get("paths")
        if not _is_string_list(paths) or len(paths) != arrays["emb"].shape[0]:
            raise ValueError("index file is inconsistent (paths vs embeddings)")
        return [forward_slashes(p) for p in paths]


class VideoIndexFormat(IndexFormat):
    """One row per sampled frame; the key is (video index path, timestamp in seconds).
    Each file name is stored once and every frame refers to it by number."""

    kind = "videos"
    name_infix = "videos_"
    log_tag = "VCACHE"
    entry_noun = "frame entries"

    def media_extensions(self):
        return settings.VIDEO_EXTS

    def file_of(self, key):
        return key[0]

    @staticmethod
    def moment_of(key):
        """Seconds into the video of a frame's key."""
        return key[1]

    def _pack_keys(self, keys):
        file_ids = {}
        file_numbers = np.empty(len(keys), dtype=np.int32)
        timestamps = np.empty(len(keys), dtype=np.float64)
        for i, (video_path, timestamp) in enumerate(keys):
            file_numbers[i] = file_ids.setdefault(forward_slashes(video_path), len(file_ids))
            timestamps[i] = timestamp
        return {"files": list(file_ids)}, {"vid_idx": file_numbers, "ts": timestamps}

    def _unpack_keys(self, meta, arrays):
        files = meta.get("files")
        file_numbers = arrays.get("vid_idx")
        timestamps = arrays.get("ts")
        rows = arrays["emb"].shape[0]
        if (not _is_string_list(files) or file_numbers is None or timestamps is None
                or len(file_numbers) != rows or len(timestamps) != rows):
            raise ValueError("video index file is inconsistent")
        if rows and (int(file_numbers.min()) < 0 or int(file_numbers.max()) >= len(files)):
            raise ValueError("video index file refers to an unknown video")
        files = [forward_slashes(p) for p in files]
        return [(files[i], t) for i, t in zip(file_numbers.tolist(), timestamps.tolist())]
