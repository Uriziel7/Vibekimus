"""The .exe's downloads, against a local HTTP server: installing wheels (resume, a dropped
connection, a wrong file, Stop, a newer version, a library no longer needed, unsafe paths,
unexpected servers), the graphics card check, the data folder, and model downloads into
the Hugging Face cache layout. Nothing is downloaded from the internet."""
import hashlib
import http.server
import json
import os
import subprocess
import sys
import threading
import zipfile

import pytest

from vibekimus import settings
from vibekimus.runtime import downloader as downloader_module
from vibekimus.runtime.downloader import Downloader, DownloadCancelled, DownloadError, ProgressCounter, RemoteFile
from vibekimus.runtime.gpu_check import evaluate
from vibekimus.runtime.installer import InstallError, RuntimeInstaller
from vibekimus.runtime.lock import RuntimeLock, Wheel

LOCAL = ("127.0.0.1",)
BIG_BYTES = 3 * (1 << 20)   # several download chunks
CUDA = "12.8"
MIN_CAPABILITY = (7, 0)
SUBPROCESS_SECONDS = 120


class Server(http.server.ThreadingHTTPServer):
    """Serves a folder with Range support; can drop a connection or redirect."""
    daemon_threads = True

    def __init__(self, folder):
        self.folder = folder
        self.requests = []         # (path, Range header)
        self.drop_after = None     # bytes sent before the next response is cut off
        self.redirect_to = None    # the next request is redirected here
        super().__init__(("127.0.0.1", 0), Handler)

    @property
    def base(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        server = self.server
        server.requests.append((self.path, self.headers.get("Range")))
        if server.redirect_to:
            target, server.redirect_to = server.redirect_to, None
            self.send_response(302)
            self.send_header("Location", target)
            self.end_headers()
            return
        path = os.path.join(server.folder, self.path.lstrip("/").replace("/", os.sep))
        if not os.path.isfile(path):
            self.send_error(404)
            return
        data = open(path, "rb").read()
        start = int(self.headers["Range"].split("=")[1].split("-")[0]) if self.headers.get("Range") else 0
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        body = data[start:]
        if server.drop_after is not None:
            body, server.drop_after = body[:server.drop_after], None
            self.wfile.write(body)
            self.wfile.flush()
            self.connection.shutdown(2)
            return
        self.wfile.write(body)


def make_wheel(folder, name, version, files, extra_members=()):
    """A wheel with `files` (path -> bytes) in its package, a .data/purelib module, a
    .data/scripts file (not to be installed) and its dist-info."""
    filename = f"{name}-{version}-py3-none-any.whl"
    path = os.path.join(folder, filename)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as wheel:
        for member, data in files.items():
            wheel.writestr(f"{name}/{member}", data)
        wheel.writestr(f"{name}-{version}.data/purelib/{name}_extra.py", "EXTRA = True\n")
        wheel.writestr(f"{name}-{version}.data/scripts/{name}-tool", "#!python\n")
        wheel.writestr(f"{name}-{version}.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n")
        for member, data in extra_members:
            wheel.writestr(member, data)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return Wheel(name=name, version=version, filename=filename, url=None, sha256=digest,
                 size=os.path.getsize(path))


def locked(server, *wheels):
    with_urls = tuple(Wheel(**{**vars(w), "url": f"{server.base}/{w.filename}"}) for w in wheels)
    return RuntimeLock(python="cp312", wheels=with_urls, cuda=CUDA, min_compute_capability=MIN_CAPABILITY,
                       installed_bytes=sum(w.size for w in wheels))


@pytest.fixture
def work(tmp_path):
    return str(tmp_path)


@pytest.fixture
def server(tmp_path):
    """A local HTTP server of a folder of its own."""
    served = tmp_path / "served"
    served.mkdir()
    running = Server(str(served))
    threading.Thread(target=running.serve_forever, daemon=True).start()
    yield running
    running.shutdown()


def install(lock, root, should_stop=lambda: False):
    installer = RuntimeInstaller(lock, root, Downloader(LOCAL, schemes=("http",)))
    counter = ProgressCounter(sum(w.size for w in installer.missing()))
    installer.install(counter, should_stop)
    return installer, counter


def test_installing_upgrading_and_removing_libraries(work, server, big, monkeypatch):
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    alpha = make_wheel(server.folder, "alpha", "1.0", {"__init__.py": "VERSION = 1\n", "old_only.py": "X = 1\n"})
    root = os.path.join(work, "runtime")
    lock = locked(server, alpha, big)

    installer, counter = install(lock, root)
    sp = installer.site_packages
    done, total, _ = counter.snapshot()
    assert os.path.isfile(os.path.join(sp, "alpha", "__init__.py")), "the package"
    assert os.path.isfile(os.path.join(sp, "alpha_extra.py")), "its .data/purelib module"
    assert os.path.isdir(os.path.join(sp, "alpha-1.0.dist-info")), "its dist-info"
    assert not os.path.exists(os.path.join(sp, "alpha-tool")) and installer.is_complete, "no scripts"
    assert (done == total == alpha.size + big.size), (
        "progress counts every byte once", (done, total))
    assert (not os.listdir(os.path.join(root, "downloads"))), (
        "the downloaded wheels are removed after unpacking")
    installer.activate()
    import alpha as imported   # noqa: E402
    assert (imported.VERSION == 1), (
        "an installed library can be imported")

    requests_before = len(server.requests)
    install(lock, root)
    assert (len(server.requests) == requests_before), (
        "nothing is downloaded again when everything is installed")

    alpha2 = make_wheel(server.folder, "alpha", "2.0", {"__init__.py": "VERSION = 2\n"})
    installer, _ = install(locked(server, alpha2, big), root)
    fetched = [path for path, _ in server.requests[requests_before:]]
    assert not os.path.exists(os.path.join(sp, "alpha", "old_only.py")), "a newer version: the old files gone"
    assert "VERSION = 2" in open(os.path.join(sp, "alpha", "__init__.py")).read()
    assert not os.path.exists(os.path.join(sp, "alpha-1.0.dist-info"))
    assert fetched == [f"/{alpha2.filename}"], ("the unchanged wheel is not fetched", fetched)
    installer, _ = install(locked(server, alpha2), root)
    assert (not os.path.exists(os.path.join(sp, "bigpkg")) and installer.is_complete), (
        "a library the lock no longer lists is removed (with its folders)")
    # onnxruntime -> onnxruntime-directml: another library, its files in the same folder
    plain = make_wheel(server.folder, "ort", "1.0", {"__init__.py": "KIND = 'cpu'\n", "cpu_only.py": "X = 1\n"})
    install(locked(server, alpha2, plain), root)
    other = make_wheel(server.folder, "ortdml", "1.0", {},
                       extra_members=[("ort/__init__.py", "KIND = 'directml'\n"), ("ort/dml.py", "Y = 1\n")])
    installer, _ = install(locked(server, alpha2, other), root)
    ort_init = os.path.join(sp, "ort", "__init__.py")
    # a library replaced by another with files in the same folder: the new files stay, the old ones go
    assert os.path.isfile(ort_init) and "directml" in open(ort_init).read()
    assert os.path.isfile(os.path.join(sp, "ort", "dml.py"))
    assert not os.path.exists(os.path.join(sp, "ort", "cpu_only.py")) and installer.is_complete
    # what an older installer left: the shared files deleted, the manifest saying "installed"
    os.remove(ort_init)
    damaged = [w.name for w in installer.missing()]
    installer, _ = install(locked(server, alpha2, other), root)
    assert (damaged == ["ortdml"] and os.path.isfile(ort_init)
          and "directml" in open(ort_init).read() and installer.is_complete), (
        "a library whose files are gone is installed again (the .exe's 'onnxruntime has no attribute "
          "get_available_providers')",
        damaged)



@pytest.fixture
def big(server, monkeypatch):
    """A wheel of several download chunks, served; a failed request is tried again at once."""
    monkeypatch.setattr(downloader_module, "RETRY_DELAY_SECONDS", 0)
    return make_wheel(server.folder, "bigpkg", "1.0", {"__init__.py": "", "blob.bin": os.urandom(BIG_BYTES)})


def test_stop_keeps_what_was_downloaded_and_the_next_start_continues_from_there(work, server, big):
    root = os.path.join(work, "runtime")
    stops = iter([False, True])
    with pytest.raises(DownloadCancelled):
        install(locked(server, big), root, should_stop=lambda: next(stops, True))
    partial = os.path.join(root, "downloads", big.filename + ".part")
    kept = os.path.getsize(partial) if os.path.exists(partial) else 0
    before = len(server.requests)
    installer, counter = install(locked(server, big), root)
    resumed = server.requests[before:]
    assert 0 < kept < big.size and resumed == [(f"/{big.filename}", f"bytes={kept}-")], (kept, resumed)
    assert installer.is_complete and counter.snapshot()[0] == big.size


def test_a_dropped_connection_is_resumed_with_a_range_request(work, server, big):
    server.drop_after = BIG_BYTES // 2
    installer, _ = install(locked(server, big), os.path.join(work, "runtime"))
    ranges = [header for _, header in server.requests]
    assert installer.is_complete and len(ranges) == 2 and ranges[0] is None and ranges[1].startswith("bytes="), ranges


def test_a_file_whose_sha256_differs_is_refused_and_deleted_and_nothing_installed(work, server):
    alpha = make_wheel(server.folder, "alpha", "2.0", {"__init__.py": "VERSION = 2\n"})
    root = os.path.join(work, "runtime")
    with pytest.raises(DownloadError):
        install(locked(server, Wheel(**{**vars(alpha), "sha256": "0" * 64})), root)
    assert not os.listdir(os.path.join(root, "downloads"))
    assert not os.path.exists(os.path.join(root, "site-packages", "alpha"))


def test_a_wheel_with_a_path_outside_the_library_folder_is_refused(work, server):
    evil = make_wheel(server.folder, "evil", "1.0", {"__init__.py": ""}, extra_members=[("../escaped.py", "X = 1")])
    root = os.path.join(work, "runtime")
    with pytest.raises(InstallError):
        install(locked(server, evil), root)
    assert not os.path.exists(os.path.join(root, "escaped.py"))


def fetch(downloader, url, target, sha256="0" * 64, size=1):
    downloader.fetch(RemoteFile(url, sha256, size, "x"), target, ProgressCounter(size), lambda: False)


def test_a_file_from_an_unexpected_server_is_refused_before_any_request(work):
    with pytest.raises(DownloadError):
        fetch(Downloader(), "https://example.com/x.whl", os.path.join(work, "x.bin"))


def test_plain_http_is_refused_https_only_outside_tests(work, server):
    with pytest.raises(DownloadError):
        fetch(Downloader(), f"{server.base}/x.whl", os.path.join(work, "x.bin"))
    assert not server.requests


def test_a_redirect_to_an_unexpected_server_is_refused(work, server):
    wheel = make_wheel(server.folder, "redirected", "1.0", {"__init__.py": ""})
    server.redirect_to = f"http://localhost:{server.server_address[1]}/{wheel.filename}"
    target = os.path.join(work, "x.bin")
    with pytest.raises(DownloadError):
        fetch(Downloader(LOCAL, schemes=("http",)), f"{server.base}/{wheel.filename}", target, wheel.sha256,
              wheel.size)
    assert not os.path.exists(target)


SMI = "| NVIDIA-SMI 616.92    Driver Version: 616.92    CUDA Version: 13.0 |"
OLD_DRIVER = "| NVIDIA-SMI 472.12    Driver Version: 472.12    CUDA Version: 11.4 |"


def card(cards, smi=SMI):
    return evaluate(cards, smi, MIN_CAPABILITY, CUDA)


def test_an_rtx_5090_with_a_new_driver_is_fine():
    rtx = card("NVIDIA GeForce RTX 5090, 12.0\n")
    assert rtx.ok and rtx.name == "NVIDIA GeForce RTX 5090", rtx


def test_no_nvidia_smi_means_no_nvidia_card():
    assert not card(None, None).ok


def test_a_gtx_1080_is_too_old_for_this_pytorch():
    pascal = card("NVIDIA GeForce GTX 1080, 6.1\n")
    assert not pascal.ok and "too old" in pascal.message, pascal


def test_an_old_driver_is_reported():
    assert "driver" in card("NVIDIA GeForce RTX 2060, 7.5\n", OLD_DRIVER).message


def test_with_two_cards_the_newer_one_counts():
    two = card("NVIDIA GeForce GTX 1050, 6.1\nNVIDIA GeForce RTX 3060, 8.6\n")
    assert two.ok and "3060" in two.name, two


def test_vibekimus_data_dir_moves_every_file_of_the_program_the_hugging_face_cache_too(work, project_root):
    code = ("import os; from vibekimus import settings; "
            "print(settings.APP_DIR); print(settings.RUNTIME_DIR); print(os.environ.get('HF_HOME'))")
    env = {**os.environ, "VIBEKIMUS_DATA_DIR": work}
    env.pop("HF_HOME", None)
    out = subprocess.run([sys.executable, "-c", code], cwd=project_root, env=env, capture_output=True,
                         text=True).stdout
    app_dir, runtime, hf_home = out.split("\n")[:3]
    assert app_dir == work and runtime == os.path.join(work, "runtime"), out
    assert hf_home == os.path.join(work, "huggingface"), out


def test_models(work, server, monkeypatch):
    from huggingface_hub import try_to_load_from_cache
    from vibekimus.encoders.catalog import SIGLIP2_SO400M
    from vibekimus.encoders.compact_weights import compact_file_for
    from vibekimus.model_store import ModelStore
    repo, revision = "someone/model", "a" * 40
    files = {"model.safetensors": os.urandom(BIG_BYTES), "config.json": b"{}", "sub/extra.json": b"[]"}
    for path, data in files.items():
        target = os.path.join(server.folder, repo, "resolve", revision, *path.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        open(target, "wb").write(data)
    entry = {"repos": [{"repo": repo, "revision": revision, "files": [
        {"path": p, "size": len(d), "sha256": hashlib.sha256(d).hexdigest()} for p, d in files.items()]}],
        "download_bytes": sum(len(d) for d in files.values()), "disk_bytes": 1}
    lock_file = os.path.join(work, "model_files.json")
    json.dump({SIGLIP2_SO400M.key: {**entry, "replaced_by_compact": {"repo": repo, "path": "model.safetensors"}},
               "plain": entry}, open(lock_file, "w"))
    cache = os.path.join(work, "hub")
    store = ModelStore(lock_file, cache, Downloader(LOCAL, schemes=("http",)),
                       resolve=server.base + "/{repo}/resolve/{revision}/{path}")
    assert (not store.is_available("plain") and store.download_bytes("plain") == entry["download_bytes"]), (
        "a model that is not downloaded is missing, with its full size")
    counter = ProgressCounter(store.download_bytes("plain"))
    store.download("plain", counter, lambda: False)
    found = try_to_load_from_cache(repo, "sub/extra.json", cache_dir=cache)
    assert (store.is_available("plain") and isinstance(found, str) and open(found, "rb").read() == b"[]"
          and counter.snapshot()[0] == counter.total), (
        "downloaded into the Hugging Face cache layout: huggingface_hub finds every file offline", found)
    os.remove(os.path.join(cache, f"models--{repo.replace('/', '--')}", "snapshots", revision, "model.safetensors"))
    monkeypatch.setattr(settings, "MODELS_DIR", os.path.join(work, "models"))
    without_compact = store.is_available(SIGLIP2_SO400M.key)
    os.makedirs(settings.MODELS_DIR)
    open(compact_file_for(SIGLIP2_SO400M), "wb").close()
    assert not without_compact, "SigLIP's removed download is missing without its compact weights"
    assert store.is_available(SIGLIP2_SO400M.key), "…and counts as present once they exist"
    assert store.download_bytes(SIGLIP2_SO400M.key) == 0


SETUP_THEN_WORKER_GC = r'''
import gc, sys, threading
sys.path.insert(0, sys.argv[1])
from vibekimus import launcher
from vibekimus.runtime.gpu_check import GpuCheck
from vibekimus.runtime.lock import RuntimeLock
from vibekimus.ui.setup_window import SetupWindow

class InstantInstaller:
    def __init__(self, lock): self._lock, self._done = lock, False
    def missing(self): return [] if self._done else list(self._lock.wheels)
    def space_needed(self, wheels): return 0
    def has_room(self, wheels): return True
    def install(self, progress, should_stop): progress.add(progress.total); self._done = True

def show():
    lock = RuntimeLock.load()
    window = SetupWindow(lock, InstantInstaller(lock), GpuCheck(True, "test card", ""))
    window._root.after(300, window._start)   # as if "Download and start" was pressed
    return window.run()

gc.disable()   # nothing may free the closed window except what follows
installed = launcher._run_setup(show)
worker = threading.Thread(target=gc.collect)   # as when the model loads
worker.start(); worker.join()
print("survived", installed)
'''

CONSOLE_OUTPUT = r'''
import ctypes, os, sys
sys.path.insert(0, sys.argv[1])
sys.stdout = sys.stderr = None   # as in the .exe without a console
from vibekimus import launcher
launcher._keep_output()
print("python line")
os.write(2, b"fd line\n")
written = ctypes.c_ulong()
ctypes.windll.kernel32.WriteFile(ctypes.windll.kernel32.GetStdHandle(ctypes.c_ulong(-12)), b"handle line\n", 12,
                                 ctypes.byref(written), None)
'''
LIBRARY_IMPORT = r'''
import sys
sys.path.insert(0, sys.argv[1])
from vibekimus import launcher
launcher._compile_libraries_once()
sys.path.insert(0, sys.argv[2])
import downloaded_library
'''


def test_compiled_libraries(work, project_root):
    """Run with -B, as PyInstaller's .exe does: the launcher still lets the libraries'
    compiled files be written once, so later starts do not compile them again."""
    library = os.path.join(work, "site_packages_bytecode")
    os.makedirs(library)
    with open(os.path.join(library, "downloaded_library.py"), "w") as f:
        f.write("VALUE = 1\n")
    env = {key: value for key, value in os.environ.items() if key != "PYTHONDONTWRITEBYTECODE"}
    run = subprocess.run([sys.executable, "-B", "-c", LIBRARY_IMPORT, project_root, library], env=env,
                         capture_output=True, text=True, timeout=SUBPROCESS_SECONDS)
    cache = os.path.join(library, "__pycache__")
    assert (run.returncode == 0 and os.path.isdir(cache)
          and any(name.startswith("downloaded_library.") for name in os.listdir(cache))), (
        "the .exe writes the compiled files of the downloaded libraries once (PyInstaller turns that off; "
          "they were compiled again at every start)",
        (run.returncode, run.stderr.strip()[-200:]))


def test_setup_window_and_console(work, project_root):
    data = os.path.join(work, "launcher_data")
    env = {**os.environ, "VIBEKIMUS_DATA_DIR": data}
    run = subprocess.run([sys.executable, "-c", SETUP_THEN_WORKER_GC, project_root], env=env, capture_output=True,
                         text=True, timeout=SUBPROCESS_SECONDS)
    assert (run.returncode == 0 and "survived True" in run.stdout), (
        "after the setup window, collecting garbage in a worker thread does not stop the program "
          "(the first-start crash: Tcl_AsyncDelete)",
        (run.returncode, run.stdout.strip(), run.stderr.strip()[-200:]))
    console = os.path.join(data, "vibekimus_console.txt")
    os.makedirs(data, exist_ok=True)
    with open(console, "w") as f:
        f.write("the run before\n")
    subprocess.run([sys.executable, "-c", CONSOLE_OUTPUT, project_root], env=env, timeout=60)
    text = open(console, encoding="utf-8").read()
    previous = os.path.join(data, "vibekimus_console.previous.txt")
    assert (all(line in text for line in ("python line", "fd line", "handle line"))
          and open(previous).read() == "the run before\n"), (
        "without a console: Python output, C-level writes and the Windows error handle go to the console "
          "file; the previous run's file is kept",
        text)
    with open(console, encoding="utf-8") as held_by_a_running_copy:   # Windows: cannot be renamed now
        second = subprocess.run([sys.executable, "-c", CONSOLE_OUTPUT, project_root], env=env, capture_output=True,
                                text=True, timeout=60)
        own_files = [name for name in os.listdir(data) if name.startswith("vibekimus_console.")
                     and name not in ("vibekimus_console.txt", "vibekimus_console.previous.txt")]
    own_text = open(os.path.join(data, own_files[0]), encoding="utf-8").read() if len(own_files) == 1 else ""
    assert (second.returncode == 0 and "python line" in own_text), (
        "a second copy started while the first holds the console file writes to a file of its own instead "
          "of stopping (WinError 32)",
        (second.returncode, own_files, second.stderr.strip()[-300:]))
