"""Console output and the run log file."""
from datetime import datetime

from vibekimus import settings

STDERR_FD = 2   # the process's error output (C libraries write straight to it)


def safe_print(text, end="\n"):
    """print() that never raises (a closed or non-UTF-8 console must not stop indexing)."""
    try:
        print(text, end=end)
    except Exception:
        pass


def log_event(text):
    """Append one timestamped line to the run log.

    The log holds configuration, speed reports and errors only — never search
    queries or file names.
    """
    try:
        with open(settings.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {text}\n")
    except Exception:
        pass
