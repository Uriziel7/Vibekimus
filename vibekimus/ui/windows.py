"""Making the program's windows (dialogs and tool windows), all the same way: title,
background, the program's icon and a dark title bar."""
import tkinter as tk

from vibekimus.ui import theme

MAXIMIZED = "zoomed"   # Tk's window state on Windows


def new_window(parent, title, *, background=theme.PANEL_BG, geometry=None, min_size=None, maximized=False,
               resizable=True, transient=True, modal=False, on_close=None, close_on_escape=False):
    """A Toplevel over `parent`.
    transient: kept above it (no taskbar button of its own); modal: it takes every click
    until it is closed; on_close: what closing it (the X, or Esc with close_on_escape) does —
    destroying it when not given."""
    window = tk.Toplevel(parent)
    window.title(title)
    window.configure(bg=background)
    if geometry:
        window.geometry(geometry)
    if maximized:
        window.state(MAXIMIZED)
    if min_size:
        window.minsize(*min_size)
    if not resizable:
        window.resizable(False, False)
    if transient:
        window.transient(parent)
    close = on_close or window.destroy
    if on_close:
        window.protocol("WM_DELETE_WINDOW", close)
    if close_on_escape:
        window.bind("<Escape>", lambda event: close())
    theme.apply_icon(window)
    theme.apply_dark_title_bar(window)
    if modal:
        window.grab_set()
    return window
