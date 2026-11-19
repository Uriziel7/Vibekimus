"""The folders of an index: Library → New index… (a name and one or more folders) and
Library → Folders of this index… (add or remove folders)."""
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from vibekimus.scope import same_or_inside
from vibekimus.ui import theme
from vibekimus.ui.widgets import ButtonRow, scrolled_listbox
from vibekimus.ui.windows import new_window

GEOMETRY = "560x400"
WRAP = 520
ADD_TITLE = "Add a folder to the index"
ALREADY_IN = "This folder is already in the list, or inside a folder of it:\n\n{folder}"
NO_FOLDERS = "Add at least one folder."


class FoldersDialog:
    """on_done(name, folders) when confirmed; name is None when the dialog does not ask
    for one. A folder that contains folders of the list takes their place."""

    def __init__(self, parent, title, intro, folders, confirm_text, on_done, name=None):
        self._folders = [os.path.normpath(folder) for folder in folders]
        self._on_done = on_done
        self._window = window = new_window(parent, title, geometry=GEOMETRY, modal=True)
        body = ttk.Frame(window, padding=theme.PAD * 2)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=intro, wraplength=WRAP, justify="left").pack(anchor="w")
        self._name = None
        if name is not None:
            row = ttk.Frame(body)
            row.pack(fill="x", pady=(theme.PAD, 0))
            ttk.Label(row, text="Name").pack(side="left")
            self._name = tk.StringVar(value=name)
            ttk.Entry(row, textvariable=self._name).pack(side="left", fill="x", expand=True, padx=(theme.GAP, 0))
        ttk.Label(body, text="Folders", font=theme.FONT_BOLD).pack(anchor="w", pady=(theme.PAD, theme.GAP))
        holder = ttk.Frame(body)
        holder.pack(fill="both", expand=True)
        self._list = scrolled_listbox(holder)
        edit = ttk.Frame(body)
        edit.pack(fill="x", pady=(theme.GAP, 0))
        ttk.Button(edit, text="Add folder…", command=self._add).pack(side="left")
        ttk.Button(edit, text="Remove", command=self._remove).pack(side="left", padx=(theme.GAP, 0))
        buttons = ButtonRow(body)
        buttons.pack(fill="x", pady=(theme.PAD * 2, 0))
        buttons.show((confirm_text, self._confirm, theme.PRIMARY_BUTTON), ("Cancel", window.destroy, None))
        self._show()

    def _show(self):
        self._list.delete(0, tk.END)
        for folder in self._folders:
            self._list.insert(tk.END, folder)

    def _add(self):
        folder = filedialog.askdirectory(parent=self._window, title=ADD_TITLE, mustexist=True)
        if not folder:
            return
        folder = os.path.normpath(folder)
        if any(same_or_inside(folder, known) for known in self._folders):
            messagebox.showinfo(ADD_TITLE, ALREADY_IN.format(folder=folder), parent=self._window)
            return
        self._folders = [known for known in self._folders if not same_or_inside(known, folder)] + [folder]
        self._show()

    def _remove(self):
        for position in reversed(self._list.curselection()):
            del self._folders[position]
        self._show()

    def _confirm(self):
        if not self._folders:
            messagebox.showwarning(self._window.title(), NO_FOLDERS, parent=self._window)
            return
        name = self._name.get().strip() if self._name is not None else None
        self._window.destroy()
        self._on_done(name, list(self._folders))
