"""The .exe's console file (what Run.bat's console would show). What the program prints
names the user's files, folders and searches, so while index encryption is on nothing is
written to it, and what it held — with the previous run's copy — is erased the first time
encryption is seen on. Standard library only: the .exe's launcher uses it before any
library is installed."""
import os

PRIVATE_NOTICE = ("(Index encryption is on: the program's output is not kept in this file, because it "
                  "names your files. The console window — the ` key — still shows it.)\n")


class PrivateConsoleFile:
    """stream: the open console file; previous: the previous run's copy; is_private():
    whether index encryption is on now."""

    def __init__(self, stream, previous, is_private):
        self._stream, self._previous, self._is_private = stream, previous, is_private
        self._erased = False

    def write(self, text):
        if not self._is_private():
            return self._stream.write(text)
        if not self._erased:
            self._erase()
        return len(text)

    def _erase(self):
        self._erased = True
        try:
            self._stream.seek(0)
            self._stream.truncate()
            self._stream.write(PRIVATE_NOTICE)
            self._stream.flush()
        except (OSError, ValueError):
            pass
        try:
            os.remove(self._previous)
        except OSError:
            pass

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):   # fileno, encoding, isatty...: those of the file
        return getattr(self._stream, name)
