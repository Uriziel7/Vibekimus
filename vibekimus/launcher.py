"""Entry point of the single .exe (tools/build_exe.py). The .exe holds Python, Tkinter and
the program's own code only; the libraries (PyTorch and the rest) are installed into the
data folder on first start, then the program starts as Vibekimus.py would.

Everything imported before the libraries are installed must use the standard library only."""
import faulthandler
import gc
import multiprocessing
import os
import sys

from vibekimus import settings
from vibekimus.console_file import PrivateConsoleFile
from vibekimus.encryption import active_vault
from vibekimus.runlog import STDERR_FD

PREVIOUS_SUFFIX = ".previous"
STD_ERROR_HANDLE = -12   # Windows: GetStdHandle / SetStdHandle


def _keep_output():
    """A .exe without a console has no stdout/stderr; libraries that print would fail. The
    output goes to the console file (the previous run's is kept), and a hard crash writes
    its traceback there too."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    os.makedirs(settings.APP_DIR, exist_ok=True)
    console = _open_console_file()
    printed = PrivateConsoleFile(console, _previous_console_file(), lambda: active_vault().enabled)
    sys.stdout = sys.stdout or printed
    sys.stderr = sys.stderr or printed
    # A hard crash still writes its traceback to the file itself (the program's own code only).
    _native_errors_to(console)
    faulthandler.enable(file=console, all_threads=True)


def _previous_console_file():
    base, extension = os.path.splitext(settings.CONSOLE_FILE)
    return base + PREVIOUS_SUFFIX + extension


def _open_console_file():
    """The console file, the previous run's kept. While another copy of the program runs,
    it holds the file open: this copy then writes to a file of its own (with its process
    number) instead of stopping before its window ever shows."""
    base, extension = os.path.splitext(settings.CONSOLE_FILE)
    try:
        if os.path.exists(settings.CONSOLE_FILE):
            os.replace(settings.CONSOLE_FILE, _previous_console_file())
        return open(settings.CONSOLE_FILE, "w", encoding="utf-8", buffering=1, errors="replace")
    except OSError:
        return open(f"{base}.{os.getpid()}{extension}", "w", encoding="utf-8", buffering=1, errors="replace")


def _native_errors_to(console):
    """What C libraries write to the error output (Tcl's last words before it stops the
    program, for one) goes to the console file too."""
    os.dup2(console.fileno(), STDERR_FD)
    if os.name == "nt":
        import ctypes
        import msvcrt
        ctypes.windll.kernel32.SetStdHandle(ctypes.c_ulong(STD_ERROR_HANDLE),   # c_ulong wraps the -12
                                            ctypes.c_void_p(msvcrt.get_osfhandle(console.fileno())))


def _compile_libraries_once():
    """PyInstaller turns off writing compiled Python files (.pyc). The downloaded libraries
    (thousands of modules: PyTorch, transformers, open_clip...) were then compiled again at
    every start: about 3 s, more with a cold disk. Written once next to them, the next
    starts read them, as a run from the source does."""
    sys.dont_write_bytecode = False


def _run_setup(show):
    """show() runs the setup window, which has its own Tk interpreter. Its widgets refer
    to each other, so only the garbage collector frees it - and Tcl must be freed in the
    thread that made it: collected later in a worker thread (loading the model), it stops
    the program ("Tcl_AsyncDelete: async handler deleted by the wrong thread"). So it is
    collected here, on the main thread, as soon as the window is gone."""
    installed = show()
    gc.collect()
    return installed


def _libraries_ready():
    from vibekimus.runtime.gpu_check import check_gpu
    from vibekimus.runtime.installer import RuntimeInstaller
    from vibekimus.runtime.lock import RuntimeLock
    lock = RuntimeLock.load()
    installer = RuntimeInstaller(lock)
    if not installer.is_complete:
        from vibekimus.ui.setup_window import SetupWindow
        if not _run_setup(lambda: SetupWindow(lock, installer, check_gpu(lock)).run()):
            return False
    installer.activate()
    return True


def main():
    multiprocessing.freeze_support()
    _keep_output()
    from vibekimus.output_log import capture
    capture()   # the console window shows the setup's output too
    _compile_libraries_once()
    if not _libraries_ready():
        return
    from vibekimus.ui.main_window import main as start_program
    start_program()


if __name__ == "__main__":
    main()
