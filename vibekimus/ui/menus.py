"""Right-click menus."""
import tkinter as tk
from dataclasses import dataclass
from typing import Callable

from vibekimus.ui import theme


@dataclass(frozen=True)
class MenuItem:
    label: str
    command: Callable
    enabled: bool = True


@dataclass(frozen=True)
class SubMenu:
    label: str
    items: tuple   # MenuItem, SubMenu or SEPARATOR


SEPARATOR = None


def _menu(parent, items):
    menu = tk.Menu(parent, tearoff=0, bg=theme.CARD_BG, fg=theme.FG,
                   activebackground=theme.ACCENT, activeforeground=theme.MENU_ACTIVE_FG,
                   relief="flat", bd=1)
    for item in items:
        if item is SEPARATOR:
            menu.add_separator()
        elif isinstance(item, SubMenu):
            menu.add_cascade(label=item.label, menu=_menu(menu, item.items))
        else:
            menu.add_command(label=item.label, command=item.command,
                             state="normal" if item.enabled else "disabled")
    return menu


def popup_menu(parent, event, items):
    """Show a context menu at the mouse position. `items` holds MenuItem, SubMenu or SEPARATOR."""
    menu = _menu(parent, items)
    try:
        menu.tk_popup(event.x_root, event.y_root)
    finally:
        menu.grab_release()
