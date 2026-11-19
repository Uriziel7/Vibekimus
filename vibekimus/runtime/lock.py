"""The exact libraries the .exe needs: every wheel with its address, size and SHA-256,
plus what the PyTorch build requires of the graphics card. Written by
tools/make_runtime_lock.py from the development venv; read-only here."""
import hashlib
import json
import os
from dataclasses import dataclass

from vibekimus.files import write_atomically

LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime_lock.json")


@dataclass(frozen=True)
class Wheel:
    name: str        # distribution name, lower case
    version: str
    filename: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class RuntimeLock:
    python: str                      # e.g. "cp312": the .exe's Python must match
    wheels: tuple
    cuda: str                        # CUDA version of the PyTorch build, e.g. "12.8"
    min_compute_capability: tuple    # oldest GPU generation the build has code for, e.g. (7, 0)
    installed_bytes: int             # disk space once installed (measured in the development venv)

    @classmethod
    def load(cls, path=LOCK_FILE):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(python=data["python"], wheels=tuple(Wheel(**wheel) for wheel in data["wheels"]),
                   cuda=data["cuda"], min_compute_capability=tuple(data["min_compute_capability"]),
                   installed_bytes=data["installed_bytes"])

    def save(self, path=LOCK_FILE):
        data = {"python": self.python, "cuda": self.cuda, "min_compute_capability": list(self.min_compute_capability),
                "installed_bytes": self.installed_bytes, "wheels": [vars(wheel) for wheel in self.wheels]}
        write_atomically(path, json.dumps(data, indent=1).encode("utf-8"))

    @property
    def download_bytes(self):
        return sum(wheel.size for wheel in self.wheels)

    def fingerprint(self):
        return hashlib.sha256("\n".join(sorted(w.sha256 for w in self.wheels)).encode()).hexdigest()
