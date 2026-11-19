"""The Windows clipboard: putting files and pictures on it, ready to paste anywhere
(Ctrl+V in Discord, a chat, an image editor or File Explorer), and taking a picture
from it to search with."""
import ctypes
import io
import os
import struct
import time
from ctypes import wintypes

from dataclasses import dataclass

from vibekimus import settings
from vibekimus.imaging import to_rgb

CF_DIB = 8                  # device-independent bitmap: what every program can paste
CF_HDROP = 15               # a list of files, as File Explorer's Ctrl+C makes
CF_DIBV5 = 17               # a bitmap with colour space (screenshot tools)
CF_TEXT = 1
CF_UNICODETEXT = 13
PNG_FORMAT_NAME = "PNG"     # registered format browsers and chat apps prefer (keeps transparency)
GMEM_MOVEABLE = 0x0002
DROPFILES_HEADER = "<IiiII"   # pFiles, pt.x, pt.y, fNC, fWide
UNICODE_PATHS = 1
BITMAP_FILE_HEADER_BYTES = 14   # a .bmp file minus this header is a CF_DIB
OPEN_ATTEMPTS = 10              # another program may be holding the clipboard for a moment
OPEN_RETRY_SECONDS = 0.05


class ClipboardUnavailable(RuntimeError):
    """The clipboard cannot be used (not on Windows, or held by another program)."""


def drop_files_payload(paths):
    """CF_HDROP data: a DROPFILES header followed by the paths, each ending in a
    NUL character, the list ending in one more."""
    header = struct.pack(DROPFILES_HEADER, struct.calcsize(DROPFILES_HEADER), 0, 0, 0, UNICODE_PATHS)
    names = "".join(path + "\0" for path in paths) + "\0"
    return header + names.encode("utf-16-le")


def dib_payload(image):
    """CF_DIB data of a PIL image (24-bit, transparency flattened on white)."""
    buffer = io.BytesIO()
    to_rgb(image).save(buffer, "BMP")
    return buffer.getvalue()[BITMAP_FILE_HEADER_BYTES:]


def png_payload(image):
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


class _WindowsClipboard:
    def __init__(self):
        if os.name != "nt":
            raise ClipboardUnavailable("copying to the clipboard is only supported on Windows")
        self._user32 = user32 = ctypes.windll.user32
        self._kernel32 = kernel32 = ctypes.windll.kernel32
        user32.OpenClipboard.argtypes = [wintypes.HWND]
        user32.OpenClipboard.restype = wintypes.BOOL
        user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        user32.SetClipboardData.restype = wintypes.HANDLE
        user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
        user32.RegisterClipboardFormatW.restype = wintypes.UINT
        user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
        user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = wintypes.LPVOID
        kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]

    def register(self, name):
        return self._user32.RegisterClipboardFormatW(name)

    def holds_any(self, format_ids):
        """Whether the clipboard holds one of these formats (it is not opened for that)."""
        return any(self._user32.IsClipboardFormatAvailable(format_id) for format_id in format_ids)

    def put(self, formats):
        """Replace the clipboard contents with {format id: bytes}."""
        self._open()
        try:
            self._user32.EmptyClipboard()
            for format_id, data in formats.items():
                self._set(format_id, data)
        finally:
            self._user32.CloseClipboard()

    def _open(self):
        for _attempt in range(OPEN_ATTEMPTS):
            if self._user32.OpenClipboard(None):
                return
            time.sleep(OPEN_RETRY_SECONDS)
        raise ClipboardUnavailable("another program is using the clipboard")

    def _set(self, format_id, data):
        kernel32 = self._kernel32
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        if not handle:
            raise MemoryError("cannot allocate clipboard memory")
        pointer = kernel32.GlobalLock(handle)
        ctypes.memmove(pointer, data, len(data))
        kernel32.GlobalUnlock(handle)
        if not self._user32.SetClipboardData(format_id, handle):
            kernel32.GlobalFree(handle)   # still ours: the clipboard did not take it
            raise ClipboardUnavailable("the clipboard refused the data")


def copy_files(paths):
    """Copy files like File Explorer does: pasting gives the original files."""
    _WindowsClipboard().put({CF_HDROP: drop_files_payload(paths)})


def copy_picture(image):
    """Copy a PIL image as a picture (also as PNG, which keeps transparency)."""
    clipboard = _WindowsClipboard()
    clipboard.put({CF_DIB: dib_payload(image), clipboard.register(PNG_FORMAT_NAME): png_payload(image)})


def _picture_formats(clipboard):
    return CF_DIB, CF_DIBV5, clipboard.register(PNG_FORMAT_NAME), CF_HDROP


def _holds(formats_of):
    """A quick look (nothing is read): the clipboard holds one of formats_of(clipboard)."""
    try:
        clipboard = _WindowsClipboard()
    except ClipboardUnavailable:
        return False
    return clipboard.holds_any(formats_of(clipboard))


def may_hold_picture():
    """The clipboard holds a picture or copied files (which may still turn out not to
    be pictures)."""
    return _holds(_picture_formats)


def holds_text():
    """The clipboard holds text — also when a program (Word, Excel, a browser) put a
    picture of it there alongside."""
    return _holds(lambda clipboard: (CF_UNICODETEXT, CF_TEXT))


@dataclass(frozen=True)
class PastedPicture:
    """A picture on the clipboard: its pixels (a screenshot, an image copied in a
    browser), or the file of a picture copied in File Explorer (read when searched,
    not here: a camera RAW file takes a second)."""
    image: object = None
    path: str = None


def pasted_picture(grab=None):
    """The PastedPicture on the clipboard: copied pixels, or the first picture file
    among copied files. None when there is none."""
    if grab is None:
        from PIL import ImageGrab
        grab = ImageGrab.grabclipboard
    try:
        content = grab()
    except Exception:   # held by another program, or a picture PIL cannot read (OSError, ValueError…)
        return None
    if isinstance(content, list):
        path = next((name for name in content if name.lower().endswith(settings.IMAGE_EXTS)
                     and os.path.isfile(name)), None)
        return None if path is None else PastedPicture(path=path)
    # copy(): decoded now, not lazily
    return PastedPicture(image=to_rgb(content).copy()) if content is not None else None
