"""Secondary windows: folder exclusions and the 'indexing is running' close prompt."""
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from vibekimus.exclusions import ExclusionsNotSaved
from vibekimus.ui import theme
from vibekimus.ui.widgets import scrolled_listbox
from vibekimus.ui.windows import new_window

EXCLUSIONS_GEOMETRY = "540x420"
EXCLUSIONS_WRAP = 510
CLOSE_DIALOG_WIDTH = 440
CLOSE_DIALOG_HEIGHT = 170
MARGIN = 12                # the exclusions window's contents from its edges
TOP_SPACING = 10           # above its help text
LINE_SPACING = 4           # between its rows
QUESTION_SPACING = 14      # under the question of the close prompt
ADD_BUTTON_CHARS, REMOVE_BUTTON_CHARS, CLOSE_BUTTON_CHARS, CHOICE_BUTTON_CHARS = 8, 16, 10, 14
EXCLUSIONS_HELP = ("Exclude files whose path contains any pattern below, or that are inside an excluded folder.\n"
                   "Patterns: case-sensitive substring match (e.g. 'nsfw', 'temp', 'backup');\n"
                   "use forward slashes for folder separators (e.g. 'raw/originals').\n"
                   "Add folder… picks one folder (with its subfolders) in the system dialog.")
EXCLUSIONS_NOTE = "Note: Run Refresh after changing exclusions to apply them to the index."
FOLDER_ENTRY_PREFIX = "Folder: "
FOLDER_DIALOG_TITLE = "Exclude folder"
NO_LIBRARY_MESSAGE = "Create or open an index first (Library menu)."
OUTSIDE_LIBRARY_MESSAGE = "This folder is not part of the open library:\n\n{folder}"
WHOLE_LIBRARY_MESSAGE = "This is the library folder itself; excluding it would leave nothing to search."


class ExclusionsDialog:
    """Lets the user add and remove exclusion patterns and folders; every change is
    saved at once."""

    def __init__(self, parent, exclusions, scope):
        self._exclusions = exclusions
        self._scope = scope
        self._entries = []   # (remove function, value) for each line of the list
        self._window = window = new_window(parent, "Folder Exclusions", background=theme.BG,
                                           geometry=EXCLUSIONS_GEOMETRY, transient=False, modal=True)

        tk.Label(window, text=EXCLUSIONS_HELP, bg=theme.BG, fg=theme.FG, font=theme.FONT_SMALL,
                 justify="left", wraplength=EXCLUSIONS_WRAP).pack(padx=MARGIN, pady=(TOP_SPACING, LINE_SPACING),
                                                                  anchor="w")
        self._listbox = self._build_list(window)
        self._entry = self._build_entry_row(window)
        self._build_buttons(window)
        tk.Label(window, text=EXCLUSIONS_NOTE, bg=theme.BG, fg=theme.ORANGE,
                 font=theme.FONT_NOTE).pack(padx=MARGIN, pady=(0, theme.PAD), anchor="w")

    def _build_list(self, window):
        frame = tk.Frame(window, bg=theme.BG)
        frame.pack(fill="both", expand=True, padx=MARGIN, pady=LINE_SPACING)
        self._listbox = listbox = scrolled_listbox(frame)
        for pattern in self._exclusions.sorted():
            self._show_pattern(pattern)
        for folder in self._exclusions.folders():
            self._show_folder(folder)
        return listbox

    def _show_pattern(self, pattern):
        self._listbox.insert(tk.END, pattern)
        self._entries.append((self._exclusions.remove, pattern))

    def _show_folder(self, folder_prefix):
        self._listbox.insert(tk.END, FOLDER_ENTRY_PREFIX + self._scope.absolute_path(folder_prefix))
        self._entries.append((self._exclusions.remove_folder, folder_prefix))

    def _build_entry_row(self, window):
        frame = tk.Frame(window, bg=theme.BG)
        frame.pack(fill="x", padx=MARGIN, pady=LINE_SPACING)
        entry = tk.Entry(frame, bg=theme.CARD_BG, fg=theme.FG, insertbackground=theme.FG,
                         font=theme.FONT_NORMAL, relief="flat", highlightthickness=1,
                         highlightbackground=theme.BORDER, highlightcolor=theme.ACCENT)
        entry.pack(side="left", fill="x", expand=True, padx=(0, theme.GAP))
        entry.bind("<Return>", lambda event: self._add())
        ttk.Button(frame, text="Add", command=self._add, width=ADD_BUTTON_CHARS).pack(side="left")
        ttk.Button(frame, text="Add folder…", command=self._add_folder).pack(side="left", padx=(theme.GAP, 0))
        return entry

    def _build_buttons(self, window):
        frame = tk.Frame(window, bg=theme.BG)
        frame.pack(fill="x", padx=MARGIN, pady=(0, LINE_SPACING))
        ttk.Button(frame, text="Remove Selected", command=self._remove, width=REMOVE_BUTTON_CHARS,
                   style=theme.DANGER_BUTTON).pack(side="left", padx=(0, theme.PAD))
        ttk.Button(frame, text="Close", command=window.destroy, width=CLOSE_BUTTON_CHARS).pack(side="left")

    def _saved(self, change):
        """change() changes the exclusions; whether they could be saved (otherwise says why)."""
        try:
            return change()
        except ExclusionsNotSaved as error:
            messagebox.showerror("Folder Exclusions", str(error), parent=self._window)
            return None

    def _add(self):
        pattern = self._entry.get().strip()
        if pattern and self._saved(lambda: self._exclusions.add(pattern)):
            self._show_pattern(pattern)
        self._entry.delete(0, tk.END)

    def _add_folder(self):
        if self._scope is None:
            messagebox.showinfo(FOLDER_DIALOG_TITLE, NO_LIBRARY_MESSAGE, parent=self._window)
            return
        folder = filedialog.askdirectory(parent=self._window, title=FOLDER_DIALOG_TITLE,
                                         initialdir=self._scope.scan_roots[0], mustexist=True)
        if not folder:
            return
        prefix = self._scope.folder_prefix(folder)
        if prefix is None:
            messagebox.showwarning(FOLDER_DIALOG_TITLE, OUTSIDE_LIBRARY_MESSAGE.format(folder=os.path.normpath(folder)),
                                   parent=self._window)
        elif prefix == "":
            messagebox.showwarning(FOLDER_DIALOG_TITLE, WHOLE_LIBRARY_MESSAGE, parent=self._window)
        elif self._saved(lambda: self._exclusions.add_folder(prefix)):
            self._show_folder(prefix)

    def _remove(self):
        selection = self._listbox.curselection()
        if not selection:
            return
        position = selection[0]
        remove, value = self._entries[position]
        if self._saved(lambda: remove(value) or True):
            del self._entries[position]
            self._listbox.delete(position)


class CloseWhileIndexingDialog:
    """Asks what to do when the window is closed during indexing, instead of
    silently killing the run."""

    def __init__(self, parent, on_stop_and_save, on_quit):
        self._window = window = new_window(parent, "Indexing in Progress", resizable=False, transient=False,
                                           modal=True)
        self._centre_over(parent)

        ttk.Label(window, text="⚠  Indexing is currently running.", font=theme.FONT_HEADING,
                  background=theme.PANEL_BG, foreground=theme.ORANGE).pack(pady=(theme.GROUP_GAP, LINE_SPACING))
        ttk.Label(window, text="What would you like to do?",
                  background=theme.PANEL_BG, foreground=theme.FG).pack(pady=(0, QUESTION_SPACING))
        buttons = tk.Frame(window, bg=theme.PANEL_BG)
        buttons.pack()
        ttk.Button(buttons, text="Stop & Save", command=self._then(on_stop_and_save),
                   style=theme.PRIMARY_BUTTON, width=CHOICE_BUTTON_CHARS).pack(side="left", padx=theme.GAP)
        ttk.Button(buttons, text="Quit Anyway", command=self._then(on_quit),
                   style=theme.DANGER_BUTTON, width=CHOICE_BUTTON_CHARS).pack(side="left", padx=theme.GAP)
        ttk.Button(buttons, text="Cancel", command=window.destroy, width=CLOSE_BUTTON_CHARS).pack(
            side="left", padx=theme.GAP)

    def _centre_over(self, parent):
        parent.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - CLOSE_DIALOG_WIDTH) // 2
        y = parent.winfo_y() + (parent.winfo_height() - CLOSE_DIALOG_HEIGHT) // 2
        self._window.geometry(f"{CLOSE_DIALOG_WIDTH}x{CLOSE_DIALOG_HEIGHT}+{x}+{y}")

    def _then(self, action):
        def close_and_act():
            self._window.destroy()
            action()
        return close_and_act
