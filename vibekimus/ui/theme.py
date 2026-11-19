"""Colours, fonts, spacing and widget styles shared by every window."""
import ctypes
import os
import sys
import tkinter as tk
from tkinter import ttk

BG = "#1e1e1e"
PANEL_BG = "#252526"
CARD_BG = "#2d2d30"
FG = "#e0e0e0"
FG_MUTED = "#9d9d9d"
OVERLAY_FG_MUTED = "#c8c8c8"    # second line of text on a picture's dark band
BORDER = "#3c3c3c"
BUTTON_BG = "#3a3d41"
BUTTON_HOVER = "#4a4e54"
BUTTON_PRESSED = "#2f3236"
ACCENT = "#0e7ad1"              # primary action, selection, progress
ACCENT_HOVER = "#2b8de0"
DANGER = "#c42b1c"
DANGER_HOVER = "#e0402f"
ORANGE = "#ff9800"
DISABLED_FG = "#6b6b6b"
MENU_ACTIVE_FG = "#ffffff"
BADGE_FG = "#000000"

# Status line colours
STATUS_OK = "#6ccb5f"
STATUS_BUSY = ORANGE
STATUS_ERROR = "#f1707a"
STATUS_STOPPED = STATUS_ERROR

FONT_FAMILY = "Segoe UI"
FONT_TINY_BOLD = (FONT_FAMILY, 7, "bold")
FONT_NOTE = (FONT_FAMILY, 8, "italic")
FONT_SMALL = (FONT_FAMILY, 9)
FONT_SMALL_BOLD = (FONT_FAMILY, 9, "bold")
FONT_NORMAL = (FONT_FAMILY, 10)
FONT_BOLD = (FONT_FAMILY, 10, "bold")
FONT_HEADING = (FONT_FAMILY, 11, "bold")
FONT_INPUT = (FONT_FAMILY, 12)

# Spacing, in pixels
PAD = 8                 # around panels
GAP = 6                 # between widgets in a group
GROUP_GAP = 18          # between groups of a wrapping bar
ROW_GAP = 6             # between rows of a wrapping bar
BUTTON_PADDING = (10, 5)
TREE_ROW_HEIGHT = 24    # rows of a table (Treeview)

PRIMARY_BUTTON = "Primary.TButton"
DANGER_BUTTON = "Danger.TButton"
MUTED_LABEL = "Muted.TLabel"
STATUS_BAR = "Status.TFrame"

DWMWA_USE_IMMERSIVE_DARK_MODE = 20   # Windows: draw the title bar dark
# The program's icon (made from assets/icon.png by tools/build_exe.py); inside the .exe it is
# unpacked next to the code.
_BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ICON_FILE = os.path.join(_BASE_DIR, "assets", "icon.ico")


def apply_styles():
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(".", background=PANEL_BG, foreground=FG, font=FONT_SMALL,
                    bordercolor=BORDER, lightcolor=PANEL_BG, darkcolor=PANEL_BG,
                    troughcolor=BORDER, focuscolor=ACCENT)
    style.configure("TFrame", background=PANEL_BG)
    style.configure(STATUS_BAR, background=BG)
    style.configure("TLabel", background=PANEL_BG, foreground=FG)
    style.configure(MUTED_LABEL, background=PANEL_BG, foreground=FG_MUTED)
    for name, colour, hover in (("TButton", BUTTON_BG, BUTTON_HOVER),
                                (PRIMARY_BUTTON, ACCENT, ACCENT_HOVER),
                                (DANGER_BUTTON, DANGER, DANGER_HOVER)):
        style.configure(name, background=colour, foreground=FG, padding=BUTTON_PADDING,
                        borderwidth=0, focusthickness=0, relief="flat")
        style.map(name, background=[("disabled", BUTTON_PRESSED), ("pressed", BUTTON_PRESSED), ("active", hover)],
                  foreground=[("disabled", DISABLED_FG)])
    style.configure("TMenubutton", background=BUTTON_BG, foreground=FG, padding=BUTTON_PADDING,
                    borderwidth=0, relief="flat", arrowcolor=FG)
    style.map("TMenubutton", background=[("active", BUTTON_HOVER)])
    style.configure("TCheckbutton", background=PANEL_BG, foreground=FG, indicatorbackground=BUTTON_BG,
                    indicatorforeground=FG, focusthickness=0)
    style.map("TCheckbutton", background=[("active", PANEL_BG)],
              indicatorbackground=[("selected", ACCENT), ("active", BUTTON_HOVER)])
    style.configure("TRadiobutton", background=PANEL_BG, foreground=FG, indicatorbackground=BUTTON_BG,
                    focusthickness=0)
    style.map("TRadiobutton", background=[("active", PANEL_BG)],
              indicatorbackground=[("selected", ACCENT), ("active", BUTTON_HOVER)])
    for field in ("TEntry", "TCombobox"):
        style.configure(field, fieldbackground=CARD_BG, foreground=FG, background=BUTTON_BG,
                        arrowcolor=FG, insertcolor=FG, bordercolor=BORDER, padding=(GAP, 2))
    style.map("TCombobox", fieldbackground=[("readonly", CARD_BG)], foreground=[("readonly", FG)],
              selectbackground=[("readonly", CARD_BG)], selectforeground=[("readonly", FG)])
    # The drop-down list of a combobox is a plain Tk listbox, styled through the option database.
    for option, value in (("background", CARD_BG), ("foreground", FG),
                          ("selectBackground", ACCENT), ("selectForeground", MENU_ACTIVE_FG)):
        style.master.option_add(f"*TCombobox*Listbox.{option}", value)
    style.configure("TSpinbox", fieldbackground=CARD_BG, foreground=FG, background=BUTTON_BG,
                    arrowcolor=FG, insertcolor=FG, bordercolor=BORDER, padding=(GAP, 2))
    style.configure("Horizontal.TScale", background=ACCENT, troughcolor=BORDER, sliderthickness=14)
    style.configure("Horizontal.TProgressbar", troughcolor=BORDER, background=ACCENT, thickness=10)
    style.configure("Vertical.TScrollbar", background=BUTTON_BG, troughcolor=BG, arrowcolor=FG)
    style.configure("TSeparator", background=BORDER)
    style.configure("TNotebook", background=PANEL_BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
    style.configure("TNotebook.Tab", background=BUTTON_BG, foreground=FG, padding=(GROUP_GAP, GAP), borderwidth=0)
    style.map("TNotebook.Tab", background=[("selected", PANEL_BG), ("active", BUTTON_HOVER)])
    style.configure("Treeview", background=CARD_BG, fieldbackground=CARD_BG, foreground=FG, borderwidth=0,
                    rowheight=TREE_ROW_HEIGHT)
    style.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", MENU_ACTIVE_FG)])
    style.configure("Treeview.Heading", background=BUTTON_BG, foreground=FG, relief="flat", font=FONT_SMALL)
    style.map("Treeview.Heading", background=[("active", BUTTON_HOVER)])


def apply_icon(root):
    """The icon of every window of the program (set on the main Tk window)."""
    if os.path.isfile(ICON_FILE):
        try:
            root.iconbitmap(default=ICON_FILE)
        except tk.TclError:
            pass


def apply_dark_title_bar(window):
    if os.name != "nt":
        return
    try:
        window.update()
        handle = ctypes.windll.user32.GetParent(window.winfo_id())
        enabled = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            handle, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(enabled), ctypes.sizeof(enabled))
    except Exception:
        pass
