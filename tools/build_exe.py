"""Builds dist\\Vibekimus.exe: one small file with Python, Tkinter and the program's own
code. The libraries (PyTorch and the rest, vibekimus/runtime/runtime_lock.json) and the
models are downloaded by the program itself on first start.

    venv\\Scripts\\python tools\\build_exe.py      (or Build-Exe.bat)

The icon is made from assets\\icon.png when it is there."""
import importlib.metadata as metadata
import os
import sys

from packaging.utils import canonicalize_name

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from vibekimus.runtime.lock import LOCK_FILE, RuntimeLock  # noqa: E402

NAME = "Vibekimus"
ENTRY = os.path.join(PROJECT, "vibekimus", "launcher.py")
ICON_PNG = os.path.join(PROJECT, "assets", "icon.png")
ICON_ICO = os.path.join(PROJECT, "assets", "icon.ico")
ICON_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
ICON_CENTRING = (0.5, 0.35)   # square crop of a portrait: a little above the middle (the face)
MODEL_FILES = os.path.join(PROJECT, "vibekimus", "model_files.json")
BUILD_DIR = os.path.join(PROJECT, "build")
DIST_DIR = os.path.join(PROJECT, "dist")
# Parts of the standard library a program never needs (tests, the IDLE editor, demos).
STDLIB_SKIPPED = {"idlelib", "turtledemo", "test", "lib2to3", "ensurepip", "venv", "pydoc_data", "antigravity",
                  "this", "turtle"}


def stdlib_modules():
    """Every standard-library module: the downloaded libraries may import any of them, and
    PyInstaller only packs what the program's own code imports."""
    return sorted(name for name in sys.stdlib_module_names if not name.startswith("_") and name not in STDLIB_SKIPPED)


def runtime_modules(lock):
    """Top-level modules of the downloaded libraries: kept out of the .exe."""
    wanted = {wheel.name for wheel in lock.wheels}
    return sorted(module for module, dists in metadata.packages_distributions().items()
                  if any(canonicalize_name(dist) in wanted for dist in dists) and module.isidentifier())


def make_icon():
    if not os.path.isfile(ICON_PNG):
        print(f"NOTE: no {ICON_PNG} - the .exe gets the default icon")
        return None
    from PIL import Image, ImageOps
    image = Image.open(ICON_PNG).convert("RGBA")
    side = min(image.size)
    square = ImageOps.fit(image, (side, side), Image.LANCZOS, centering=ICON_CENTRING)
    square.save(ICON_ICO, sizes=ICON_SIZES)
    return ICON_ICO


def main():
    import PyInstaller.__main__
    lock = RuntimeLock.load()
    icon = make_icon()
    python3_dll = os.path.join(sys.base_prefix, "python3.dll")   # abi3 extensions (OpenCV, safetensors) need it
    args = [ENTRY, "--noconfirm", "--onefile", "--windowed", "--name", NAME,
            "--distpath", DIST_DIR, "--workpath", BUILD_DIR, "--specpath", BUILD_DIR,
            "--paths", PROJECT, "--collect-submodules", "vibekimus",
            "--add-data", f"{LOCK_FILE};vibekimus/runtime", "--add-data", f"{MODEL_FILES};vibekimus",
            "--add-binary", f"{python3_dll};."]
    if icon:
        args += ["--icon", icon, "--add-data", f"{icon};assets"]
    for module in stdlib_modules():
        args += ["--collect-submodules", module]
    for module in runtime_modules(lock):
        args += ["--exclude-module", module]
    PyInstaller.__main__.run(args)
    exe = os.path.join(DIST_DIR, NAME + ".exe")
    print(f"\n{exe}: {os.path.getsize(exe) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
