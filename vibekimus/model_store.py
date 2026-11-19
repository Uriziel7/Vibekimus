"""Which models are on this computer, and downloading one.

The files of each model (repository, revision, path, size, SHA-256) are pinned in
model_files.json (tools/make_model_lock.py). They are written into the Hugging Face cache
the way huggingface_hub itself lays it out (snapshots/<revision>/<path>, refs/main), so
open_clip, transformers and the OCR reader find them offline without any change. The
download goes through runtime.downloader: only Hugging Face servers, every file checked."""
import json
import os
from dataclasses import dataclass

from vibekimus.encoders.catalog import model_by_key
from vibekimus.encoders.compact_weights import compact_file_for
from vibekimus.runtime.downloader import Downloader, RemoteFile

MODEL_FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_files.json")
OCR_KEY = "ocr"   # the text reader's two models (not a ModelSpec)
RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
SNAPSHOTS, REFS, MAIN = "snapshots", "refs", "main"


@dataclass(frozen=True)
class ModelFile:
    repo: str
    revision: str
    path: str
    size: int
    sha256: str


def _cache_dir():
    from huggingface_hub.constants import HF_HUB_CACHE
    return HF_HUB_CACHE


def _repo_folder(cache, repo):
    return os.path.join(cache, "models--" + repo.replace("/", "--"))


class ModelStore:
    def __init__(self, lock_file=MODEL_FILES, cache_dir=None, downloader=None, resolve=RESOLVE):
        """resolve: the download address pattern (another server only in tests)."""
        with open(lock_file, "r", encoding="utf-8") as f:
            self._lock = json.load(f)
        self._cache = cache_dir or _cache_dir()
        self._downloader = downloader or Downloader()
        self._resolve = resolve

    def _files(self, key):
        return [ModelFile(source["repo"], source["revision"], f["path"], f["size"], f["sha256"])
                for source in self._lock[key]["repos"] for f in source["files"]]

    def _replaced(self, key, model_file):
        """A file the compact weights stand in for, once they exist."""
        replaced = self._lock[key].get("replaced_by_compact")
        if not replaced or (replaced["repo"], replaced["path"]) != (model_file.repo, model_file.path):
            return False
        return os.path.isfile(compact_file_for(model_by_key(key)))

    def _cached(self, model_file):
        """The file in the cache: at the pinned revision, or at whatever the cache calls
        main (an installation that downloaded the model the earlier way)."""
        folder = _repo_folder(self._cache, model_file.repo)
        if os.path.isfile(os.path.join(folder, SNAPSHOTS, model_file.revision, model_file.path)):
            return True
        try:
            with open(os.path.join(folder, REFS, MAIN), "r", encoding="utf-8") as f:
                revision = f.read().strip()
        except OSError:
            return False
        return os.path.isfile(os.path.join(folder, SNAPSHOTS, revision, model_file.path))

    def _missing(self, key):
        return [f for f in self._files(key) if not self._cached(f) and not self._replaced(key, f)]

    def is_available(self, key):
        return key in self._lock and not self._missing(key)

    def download_bytes(self, key):
        return sum(f.size for f in self._missing(key))

    def disk_bytes(self, key):
        """Space the model takes once ready (SigLIP: after its compact weights are made)."""
        return self._lock[key]["disk_bytes"]

    def download(self, key, progress, should_stop):
        """Download what is missing of model `key` (progress: runtime.downloader.ProgressCounter)."""
        for model_file in self._missing(key):
            progress.set_item(f"Downloading {model_file.repo.split('/')[-1]}: {model_file.path}")
            folder = _repo_folder(self._cache, model_file.repo)
            target = os.path.join(folder, SNAPSHOTS, model_file.revision, *model_file.path.split("/"))
            url = self._resolve.format(repo=model_file.repo, revision=model_file.revision, path=model_file.path)
            self._downloader.fetch(RemoteFile(url, model_file.sha256, model_file.size, model_file.path), target,
                                   progress, should_stop)
            self._point_main_at(folder, model_file.revision)

    @staticmethod
    def _point_main_at(folder, revision):
        os.makedirs(os.path.join(folder, REFS), exist_ok=True)
        with open(os.path.join(folder, REFS, MAIN), "w", encoding="utf-8") as f:
            f.write(revision)
