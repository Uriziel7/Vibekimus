"""Installs the locked wheels into the .exe's own library folder: downloads what is
missing, unpacks it (a wheel is a zip archive), and on a newer .exe replaces only the
libraries whose version changed. Standard library only."""
import json
import os
import shutil
import sys
import zipfile

from vibekimus import settings
from vibekimus.files import write_atomically
from vibekimus.runtime.downloader import Downloader, RemoteFile

SITE_PACKAGES = "site-packages"
DOWNLOADS = "downloads"
MANIFEST = "installed.json"
DATA_SUFFIX = ".data"
LIBRARY_SCHEMES = ("purelib", "platlib")   # parts of a wheel's .data folder that are code
# Files at most this many folders deep are checked on every start (a package's __init__.py,
# top-level modules, its dist-info): ~1000 files, under 0.1 s. Checking all ~20 000 takes seconds.
CHECKED_DEPTH = 1


class InstallError(Exception):
    pass


class RuntimeInstaller:
    def __init__(self, lock, root=None, downloader=None):
        self._lock = lock
        self._root = root or settings.RUNTIME_DIR
        self._downloader = downloader or Downloader()
        self.site_packages = os.path.join(self._root, SITE_PACKAGES)

    # ── state ────────────────────────────────────────────────────────────────
    def _manifest(self):
        """name -> {"filename": wheel installed, "files": its files relative to site-packages}"""
        try:
            with open(os.path.join(self._root, MANIFEST), "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save_manifest(self, manifest):
        write_atomically(os.path.join(self._root, MANIFEST), json.dumps(manifest).encode("utf-8"))

    def missing(self):
        """The wheels not installed yet, installed in another version, or with files gone
        (an older installer deleted files that onnxruntime-directml shares with onnxruntime)."""
        manifest = self._manifest()
        return [w for w in self._lock.wheels
                if manifest.get(w.name, {}).get("filename") != w.filename or not self._is_intact(manifest[w.name])]

    def _is_intact(self, entry):
        """The installed library's top-level files are all still there."""
        return all(os.path.exists(os.path.join(self.site_packages, relative))
                   for relative in entry.get("files", []) if relative.count(os.sep) <= CHECKED_DEPTH)

    def leftovers(self):
        """Installed libraries the lock no longer lists."""
        wanted = {w.name for w in self._lock.wheels}
        return [name for name in self._manifest() if name not in wanted]

    @property
    def is_complete(self):
        return not self.missing() and not self.leftovers()

    def space_needed(self, wheels):
        """Free disk space the installation needs: everything installed, plus the largest
        download while it is unpacked."""
        return self._lock.installed_bytes + max((w.size for w in wheels), default=0)

    def has_room(self, wheels):
        os.makedirs(self._root, exist_ok=True)
        return shutil.disk_usage(self._root).free >= self.space_needed(wheels)

    def activate(self):
        """Make the installed libraries importable."""
        if self.site_packages not in sys.path:
            sys.path.insert(0, self.site_packages)

    # ── installing ───────────────────────────────────────────────────────────
    def install(self, progress, should_stop):
        """Download and unpack every missing wheel; progress counts downloaded bytes.
        Stops with DownloadCancelled (finished wheels stay installed)."""
        wheels = sorted(self.missing(), key=lambda w: w.size, reverse=True)
        os.makedirs(self.site_packages, exist_ok=True)
        manifest = self._manifest()
        # Libraries no longer wanted go first: one replaced by another may share its folder and
        # file names (onnxruntime -> onnxruntime-directml), and removing it afterwards would
        # delete the new files.
        for name in self.leftovers():
            self._remove(manifest.pop(name))
            self._save_manifest(manifest)
        for wheel in wheels:
            progress.set_item(f"Downloading {wheel.name} {wheel.version}")
            archive = os.path.join(self._root, DOWNLOADS, wheel.filename)
            self._downloader.fetch(RemoteFile(wheel.url, wheel.sha256, wheel.size, wheel.name), archive,
                                   progress, should_stop)
            progress.set_item(f"Installing {wheel.name} {wheel.version}")
            self._remove(manifest.pop(wheel.name, None))
            files = self._unpack(archive)
            manifest[wheel.name] = {"filename": wheel.filename, "files": files}
            self._save_manifest(manifest)
            os.remove(archive)

    def _unpack(self, archive):
        """Unpack a wheel into site-packages; returns the files it wrote."""
        written = []
        root = os.path.realpath(self.site_packages)
        with zipfile.ZipFile(archive) as wheel:
            for member in wheel.infolist():
                if member.is_dir():
                    continue
                relative = self._target(member.filename)
                if relative is None:
                    continue
                target = os.path.realpath(os.path.join(root, relative))
                if os.path.commonpath([root, target]) != root:
                    raise InstallError(f"{os.path.basename(archive)}: refused a path outside the library folder")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with wheel.open(member) as source, open(target, "wb") as out:
                    shutil.copyfileobj(source, out)
                written.append(os.path.relpath(target, root))
        return written

    @staticmethod
    def _target(name):
        """Where a wheel member goes, relative to site-packages (None: not needed)."""
        parts = name.split("/")
        if parts[0].endswith(DATA_SUFFIX):
            if len(parts) < 3 or parts[1] not in LIBRARY_SCHEMES:
                return None   # scripts, headers, data files: a program needs none of them
            parts = parts[2:]
        if any(part in ("", "..") for part in parts) or ":" in parts[0]:
            raise InstallError(f"unsafe path in a wheel: {name}")
        return os.path.join(*parts)

    def _remove(self, entry):
        if not entry:
            return
        folders = set()
        for relative in entry.get("files", []):
            path = os.path.join(self.site_packages, relative)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            folders.add(os.path.dirname(path))
            cache = os.path.join(os.path.dirname(path), "__pycache__")
            if os.path.isdir(cache):
                shutil.rmtree(cache, ignore_errors=True)
        for folder in sorted(folders, key=len, reverse=True):   # empty folders, deepest first
            while folder.startswith(self.site_packages) and folder != self.site_packages:
                try:
                    os.rmdir(folder)
                except OSError:
                    break
                folder = os.path.dirname(folder)
