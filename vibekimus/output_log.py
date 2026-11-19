"""What the program prints (its stdout and stderr), kept in memory for the console window.

The output still goes where it went before (the console of Run.bat, or the .exe's console
file); a copy of the last settings.CONSOLE_MAX_LINES lines is kept here. A carriage return
without a new line (progress bars) rewrites the current line, as a real console does.
Standard library only: the .exe's launcher captures the output before any library is
installed."""
import itertools
import sys
import threading
from collections import deque

from vibekimus import settings

NEW_LINE, CARRIAGE_RETURN, WINDOWS_NEW_LINE = "\n", "\r", "\r\n"
FALLBACK_ENCODING = "utf-8"


class OutputLog:
    """Completed lines (numbered from the start, the oldest dropped first) and the line
    still being written. Safe to write from any thread."""

    def __init__(self, max_lines):
        self._lines = deque(maxlen=max_lines)
        self._total = 0        # completed lines ever written
        self._partial = ""     # the current line, not finished yet
        self._lock = threading.Lock()

    def write(self, text):
        segments = text.replace(WINDOWS_NEW_LINE, NEW_LINE).split(NEW_LINE)
        with self._lock:
            for finished in segments[:-1]:
                self._lines.append(self._rewritten(self._partial + finished))
                self._total += 1
                self._partial = ""
            self._partial = self._rewritten(self._partial + segments[-1])

    @staticmethod
    def _rewritten(line):
        """What is left of a line after carriage returns: the text after the last one."""
        return line.rsplit(CARRIAGE_RETURN, 1)[-1]

    def since(self, seen):
        """(the completed lines after the first `seen`, the line being written, how many
        completed lines there are now). Lines already dropped are skipped."""
        with self._lock:
            first_kept = self._total - len(self._lines)
            new = list(itertools.islice(self._lines, max(seen, first_kept) - first_kept, None))
            return new, self._partial, self._total


class _Tee:
    """A stream that writes to the original one (when there is one) and to the log."""

    def __init__(self, original, log):
        self._original, self._log = original, log

    def write(self, text):
        self._log.write(text)
        if self._original is not None:
            return self._original.write(text)
        return len(text)

    def flush(self):
        if self._original is not None:
            self._original.flush()

    @property
    def encoding(self):
        return getattr(self._original, "encoding", FALLBACK_ENCODING)

    def __getattr__(self, name):   # isatty, fileno, errors...: those of the original stream
        return getattr(self._original, name)


OUTPUT = OutputLog(settings.CONSOLE_MAX_LINES)


def capture():
    """Copy stdout and stderr into OUTPUT from now on (once; later calls do nothing)."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if not isinstance(stream, _Tee):
            setattr(sys, name, _Tee(stream, OUTPUT))
