"""Handing files over to the operating system: show in file manager, open in viewer."""
import os
import shutil
import subprocess
import sys

from vibekimus.files import plain_path
from vibekimus.runlog import safe_print

# Linux file managers and how each is asked to show a file ("{path}") or, where it
# cannot highlight a file, its folder ("{folder}"). First one installed wins.
LINUX_FILE_MANAGERS = (
    ("dolphin", ["dolphin", "--select", "{path}"]),    # KDE
    ("nautilus", ["nautilus", "--select", "{path}"]),  # GNOME
    ("nemo", ["nemo", "{path}"]),                      # Linux Mint
    ("caja", ["caja", "--select", "{path}"]),          # MATE
    ("thunar", ["thunar", "{folder}"]),                # XFCE
    ("pcmanfm", ["pcmanfm", "{folder}"]),              # LXDE
)
LINUX_FALLBACK = ["xdg-open", "{folder}"]


def _spawn_quietly(command):
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def reveal_in_file_manager(path):
    """Open the folder that holds `path`, with the file highlighted where supported."""
    if not os.path.exists(path):
        return
    path = plain_path(path)   # Explorer rejects the long-path prefix
    if os.name == "nt":
        subprocess.Popen(f'explorer /select,"{path}"')
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
    else:
        installed = (command for name, command in LINUX_FILE_MANAGERS if shutil.which(name))
        command = next(installed, LINUX_FALLBACK)
        _spawn_quietly([part.format(path=path, folder=os.path.dirname(path)) for part in command])


def open_with_default_app(path):
    """Open the file in whatever program the system associates with it."""
    if not os.path.exists(path):
        return
    try:
        if os.name == "nt":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            _spawn_quietly(["xdg-open", path])
    except Exception as error:
        safe_print(f"[OPEN] Failed to open viewer: {error}")
