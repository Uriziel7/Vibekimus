"""Opening a video at a given moment in a player that can start there."""
import ctypes
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable

from vibekimus import settings
from vibekimus.shell import open_with_default_app
from vibekimus.units import MILLISECONDS_PER_SECOND, SECONDS_PER_HOUR, SECONDS_PER_MINUTE

PROGRAM_FOLDERS = ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")
ASSOCSTR_EXECUTABLE = 2
MAX_PATH_CHARS = 1024


def clock(seconds):
    """hh:mm:ss.mmm"""
    hours, rest = divmod(seconds, SECONDS_PER_HOUR)
    minutes, rest = divmod(rest, SECONDS_PER_MINUTE)
    return f"{int(hours):02d}:{int(minutes):02d}:{rest:06.3f}"


@dataclass(frozen=True)
class PlayerSpec:
    name: str
    executables: tuple      # file names of the program, lower case
    install_paths: tuple    # relative to Program Files / Local AppData
    arguments: Callable     # (video path, seconds) -> command-line arguments after the program


KNOWN_PLAYERS = (
    PlayerSpec("PotPlayer", ("potplayermini64.exe", "potplayermini.exe", "potplayer64.exe", "potplayer.exe"),
               (r"DAUM\PotPlayer\PotPlayerMini64.exe", r"PotPlayer\PotPlayerMini64.exe"),
               lambda path, seconds: [path, f"/seek={clock(seconds)}"]),
    PlayerSpec("VLC", ("vlc.exe", "vlc"),
               (r"VideoLAN\VLC\vlc.exe",),
               lambda path, seconds: [f"--start-time={seconds:.3f}", path]),
    PlayerSpec("MPC-HC", ("mpc-hc64.exe", "mpc-hc.exe"),
               (r"MPC-HC\mpc-hc64.exe", r"K-Lite Codec Pack\MPC-HC64\mpc-hc64.exe"),
               lambda path, seconds: [path, "/start", str(int(seconds * MILLISECONDS_PER_SECOND))]),
    PlayerSpec("mpv", ("mpv.exe", "mpv"),
               (r"mpv\mpv.exe",),
               lambda path, seconds: [f"--start={seconds:.3f}", path]),
)


def _spec_for(executable):
    name = os.path.basename(executable).lower()
    return next((spec for spec in KNOWN_PLAYERS if name in spec.executables), None)


def _associated_program(extension):
    """The program Windows opens files of this extension with, or None."""
    if os.name != "nt":
        return None
    try:
        size = ctypes.c_uint(MAX_PATH_CHARS)
        buffer = ctypes.create_unicode_buffer(MAX_PATH_CHARS)
        result = ctypes.windll.shlwapi.AssocQueryStringW(0, ASSOCSTR_EXECUTABLE, extension, None,
                                                        buffer, ctypes.byref(size))
        return buffer.value if result == 0 and buffer.value else None
    except Exception:
        return None


def _installed(spec):
    for folder in PROGRAM_FOLDERS:
        base = os.environ.get(folder)
        for relative in spec.install_paths if base else ():
            candidate = os.path.join(base, relative)
            if os.path.isfile(candidate):
                return candidate
    return next(filter(None, (shutil.which(name) for name in spec.executables)), None)


def find_player(extension):
    """(PlayerSpec, program path) able to start a video at a moment, or None.
    settings.VIDEO_PLAYER wins; then the program Windows uses for the extension, if
    it is a known player; then the first known player that is installed."""
    for candidate in (settings.VIDEO_PLAYER, _associated_program(extension)):
        spec = _spec_for(candidate) if candidate else None
        if spec and os.path.isfile(candidate):
            return spec, candidate
    for spec in KNOWN_PLAYERS:
        program = _installed(spec)
        if program:
            return spec, program
    return None


def open_video_at(path, seconds):
    """Play the video from `seconds`. Returns the player's name, or None if no
    player that can start at a moment was found (the video then opens from the
    start in the default program)."""
    found = find_player(os.path.splitext(path)[1].lower())
    if found is None:
        open_with_default_app(path)
        return None
    spec, program = found
    subprocess.Popen([program, *spec.arguments(os.path.normpath(path), max(0.0, seconds))])
    return spec.name
