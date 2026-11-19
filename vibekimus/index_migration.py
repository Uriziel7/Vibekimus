"""An index kept inside a folder (made by older versions, paths relative to the folder)
turned into an index in the program's data (full paths), so that more folders can be
added to it. Everything indexed comes along — the index files of every model, the text
read from the pictures and the exclusions — and nothing is computed again. The old files
are left where they are."""
import os

from vibekimus.encoders.catalog import MODELS
from vibekimus.exclusions import Exclusions
from vibekimus.index_files import ImageIndexFormat, VideoIndexFormat, index_exists
from vibekimus.saved_indexes import SavedIndexes
from vibekimus.scope import FolderScope
from vibekimus.text_index import TextIndex


class _PathMapping:
    """Index paths of the old (folder) scope to those of the new one."""

    def __init__(self, old, new):
        self._old, self._new = old, new

    def path(self, index_path):
        return self._new.index_path(self._old.absolute_path(index_path))

    def image_key(self, key):
        return self.path(key)

    def video_key(self, key):
        return self.path(key[0]), key[1]


def move_folder_index(folder, name):
    """The new index (a MultiFolderScope of just this folder) with the folder's contents.
    Raises encryption.LockedError when the old files are encrypted and the key is locked."""
    old = FolderScope(folder)
    new = SavedIndexes.create(name, [folder])
    mapping = _PathMapping(old, new)
    for model in MODELS:
        for index_format, key_of in ((ImageIndexFormat(model), mapping.image_key),
                                     (VideoIndexFormat(model), mapping.video_key)):
            _move_index(old, new, index_format, key_of)
    _move_texts(old, new, mapping)
    _move_exclusions(old, new, mapping)
    return new


def _move_index(old, new, index_format, key_of):
    source = os.path.join(old.data_dir, index_format.file_name())
    if not index_exists(source):
        return
    keys, embeddings, info = index_format.read_merged(source)
    old.validate(index_format.file_of(key) for key in keys)
    index_format.write(os.path.join(new.data_dir, index_format.file_name()), [key_of(key) for key in keys],
                       embeddings, info)


def _move_texts(old, new, mapping):
    source = TextIndex()
    source.open(old.data_dir)
    if not source.exists():
        return
    source.load(old.validate)
    target = TextIndex()
    target.open(new.data_dir)
    for key in source.keys():
        target.set(mapping.path(key), source.text_of(key))
    target.save()


def _move_exclusions(old, new, mapping):
    source, target = Exclusions(), Exclusions()
    source.load(old.data_dir)
    target.load(new.data_dir)
    for pattern in source.sorted():
        target.add(pattern)
    for folder in source.folders():
        target.add_folder(mapping.path(folder))
