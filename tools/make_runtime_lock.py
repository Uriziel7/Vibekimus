"""Writes vibekimus/runtime/runtime_lock.json: every library of the development venv (except
the build tools) as the exact wheel for this Python on Windows x64, with its address,
size and SHA-256 from PyPI or the PyTorch index. Run it after changing requirements:

    venv\\Scripts\\python tools\\make_runtime_lock.py

Reads only public package metadata."""
import importlib.metadata as metadata
import json
import os
import re
import sys
import urllib.parse
import urllib.request

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibekimus.runtime.lock import RuntimeLock, Wheel  # noqa: E402

# Build and test tools (requirements-dev.txt): in the venv, not in the .exe's libraries.
NOT_SHIPPED = {"pip", "setuptools", "wheel", "pyinstaller", "pyinstaller-hooks-contrib", "altgraph", "pefile",
               "pywin32-ctypes", "pytest", "iniconfig", "pluggy", "pygments"}
PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"
PYTORCH_INDEX = "https://download.pytorch.org/whl/{variant}/{name}/"
PYTORCH_HOST = "https://download.pytorch.org"
LOCAL_VARIANT = re.compile(r"\+(cu\d+|cpu)$")
TIMEOUT_SECONDS = 60
TAG_RANK = {tag: rank for rank, tag in enumerate(sys_tags())}   # best match for this Python first


def _best(candidates):
    """(rank, filename, url, sha256, size) of the wheel this Python prefers, or None."""
    ranked = []
    for filename, url, sha256, size in candidates:
        if not filename.endswith(".whl"):
            continue
        _, _, _, tags = parse_wheel_filename(filename)
        rank = min((TAG_RANK[tag] for tag in tags if tag in TAG_RANK), default=None)
        if rank is not None:
            ranked.append((rank, filename, url, sha256, size))
    return min(ranked) if ranked else None


def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "vibekimus-lock"}),
                                timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def _from_pypi(name, version):
    files = json.loads(_get(PYPI_JSON.format(name=name, version=version)))["urls"]
    return _best((f["filename"], f["url"], f["digests"]["sha256"], f["size"]) for f in files)


def _from_pytorch(name, version, variant):
    page = _get(PYTORCH_INDEX.format(variant=variant, name=name)).decode()
    candidates = []
    for href in re.findall(r'href="([^"]+)"', page):
        link, _, fragment = href.partition("#sha256=")
        filename = urllib.parse.unquote(link.rsplit("/", 1)[-1])
        if not fragment or not filename.startswith(f"{name.replace('-', '_')}-{version}-"):
            continue
        url = link if link.startswith("http") else PYTORCH_HOST + link
        request = urllib.request.Request(url, method="HEAD")
        candidates.append((filename, url, fragment, None, request))
    best = _best((f, u, s, r) for f, u, s, _, r in candidates)
    if best is None:
        return None
    rank, filename, url, sha256, request = best
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return rank, filename, url, sha256, int(response.headers["Content-Length"])


def _torch_arch():
    import torch
    capabilities = [tuple(divmod(int(arch.split("_")[1]), 10)) for arch in torch.cuda.get_arch_list()
                    if arch.startswith("sm_")]
    return torch.version.cuda, min(capabilities)


def _installed_size(dist):
    paths = [dist.locate_file(f) for f in dist.files or ()]
    return sum(os.path.getsize(path) for path in paths if os.path.isfile(path))


def main():
    wheels, failures, installed = [], [], 0
    for dist in sorted(metadata.distributions(), key=lambda d: canonicalize_name(d.metadata["Name"])):
        name = canonicalize_name(dist.metadata["Name"])
        if name in NOT_SHIPPED:
            continue
        version = dist.version
        variant = LOCAL_VARIANT.search(version)
        found = _from_pytorch(name, version, variant.group(1)) if variant else _from_pypi(name, version)
        if found is None:
            failures.append(f"{name} {version}")
            continue
        _, filename, url, sha256, size = found
        wheels.append(Wheel(name=name, version=version, filename=filename, url=url, sha256=sha256, size=size))
        installed += _installed_size(dist)
        print(f"{size / 1e6:9.1f} MB  {filename}")
    if failures:
        sys.exit("No wheel for this Python on Windows x64: " + ", ".join(failures))
    cuda, min_capability = _torch_arch()
    lock = RuntimeLock(python=f"cp{sys.version_info.major}{sys.version_info.minor}", wheels=tuple(wheels),
                       cuda=cuda, min_compute_capability=min_capability, installed_bytes=installed)
    lock.save()
    print(f"\n{len(wheels)} wheels, {lock.download_bytes / 1e9:.2f} GB ({installed / 1e9:.2f} GB installed); "
          f"CUDA {cuda}, GPUs from compute "
          f"capability {min_capability[0]}.{min_capability[1]}")


if __name__ == "__main__":
    main()
