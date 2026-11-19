"""Writes vibekimus/model_files.json: for each model the program can download, the exact
files (repository, revision, path, size, SHA-256) - the revisions this project was tested
with, as found in the local Hugging Face cache. Run it after updating a model:

    venv\\Scripts\\python tools\\make_model_lock.py

Reads only public Hugging Face metadata (and downloads the small non-LFS files to hash them)."""
import fnmatch
import hashlib
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibekimus import text_reading  # noqa: E402
from vibekimus.encoders.catalog import SIGLIP2_SO400M  # noqa: E402
from vibekimus.encoders.compact_weights import compact_file_for  # noqa: E402
from vibekimus.model_store import MODEL_FILES, OCR_KEY  # noqa: E402

TREE = "https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1"
RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
TIMEOUT_SECONDS = 60
SIGLIP_REPO = "timm/ViT-SO400M-16-SigLIP2-384"
SIGLIP_WEIGHTS = "open_clip_model.safetensors"
SKIPPED = ("README.md", ".gitattributes", "scripts/*")   # not needed to run a model (and no code)
# key -> [(repository, wanted path patterns)]; the first pattern list that matches decides.
MODELS = {
    SIGLIP2_SO400M.key: [(SIGLIP_REPO, [SIGLIP_WEIGHTS, "tokenizer.json", "tokenizer_config.json",
                                        "special_tokens_map.json"])],
    OCR_KEY: [(repo, [text_reading.MODEL_FILE, text_reading.CONFIG_FILE])
              for repo in (text_reading.DETECTION_REPO, text_reading.RECOGNITION_REPO)],
}
REPLACED_BY_COMPACT = {SIGLIP2_SO400M.key: (SIGLIP_REPO, SIGLIP_WEIGHTS, SIGLIP2_SO400M)}


def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "vibekimus-lock"}),
                                timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def _local_revision(repo):
    from huggingface_hub.constants import HF_HUB_CACHE
    with open(os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"), "refs", "main")) as f:
        return f.read().strip()


def _files(repo, patterns):
    revision = _local_revision(repo)
    entries = json.loads(_get(TREE.format(repo=repo, revision=revision)))
    files = []
    for entry in entries:
        path = entry["path"]
        if entry["type"] != "file" or any(fnmatch.fnmatch(path, skip) for skip in SKIPPED):
            continue
        if not any(fnmatch.fnmatch(path, pattern) for pattern in patterns):
            continue
        lfs = entry.get("lfs")
        sha256 = lfs["oid"] if lfs else hashlib.sha256(_get(RESOLVE.format(repo=repo, revision=revision,
                                                                           path=path))).hexdigest()
        files.append({"path": path, "size": entry["size"], "sha256": sha256})
    return {"repo": repo, "revision": revision, "files": files}


def main():
    lock = {}
    for key, sources in MODELS.items():
        repos = [_files(repo, patterns) for repo, patterns in sources]
        download = sum(f["size"] for r in repos for f in r["files"])
        entry = {"repos": repos, "download_bytes": download, "disk_bytes": download}
        if key in REPLACED_BY_COMPACT:
            repo, path, spec = REPLACED_BY_COMPACT[key]
            weights = next(f["size"] for r in repos if r["repo"] == repo for f in r["files"] if f["path"] == path)
            compact = compact_file_for(spec)
            entry["replaced_by_compact"] = {"repo": repo, "path": path}
            entry["disk_bytes"] = download - weights + (os.path.getsize(compact) if os.path.exists(compact) else weights)
        lock[key] = entry
        print(f"{key}: {len([f for r in repos for f in r['files']])} files, download {download / 1e9:.2f} GB, "
              f"on disk {entry['disk_bytes'] / 1e9:.2f} GB")
    with open(MODEL_FILES, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=1)


if __name__ == "__main__":
    main()
