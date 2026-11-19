"""The main window's keys that are not typed into a field: Ctrl+C copies the selected files,
Ctrl+V searches with a pasted picture, Space previews a result, the console key opens the
console, and typing over the results goes on in the search field."""
import tkinter as tk
from tkinter import ttk

from vibekimus import settings
from vibekimus.clipboard import holds_text, may_hold_picture

TEXT_FIELDS = (tk.Entry, ttk.Entry, ttk.Combobox, ttk.Spinbox, tk.Text)   # keys there are typed
BACKSPACE_KEY = "BackSpace"
PASTE_KEYS = ("<Control-v>", "<Control-V>")   # V with Caps Lock on
HANDLED = "break"   # Tk: no other binding gets the key


class KeyRouter:
    """copy_files(paths) puts files on the clipboard; paste_picture() searches with the picture
    on the clipboard and returns whether the key was used for it (no picture: a field then
    pastes its text)."""

    def __init__(self, root, query_entry, results, console, copy_files, paste_picture):
        self._root, self._entry, self._results, self._console = root, query_entry, results, console
        self._copy_files, self._paste_picture = copy_files, paste_picture
        root.bind_all("<Control-c>", self._on_copy)
        for key in PASTE_KEYS:
            root.bind_all(key, self._on_paste_outside_fields)
            query_entry.bind(key, self._on_paste_in_search)   # before the field pastes text
        for key in settings.CONSOLE_TOGGLE_KEYS:
            root.bind_all(f"<KeyPress-{key}>", self._on_console_key)
            query_entry.bind(f"<KeyPress-{key}>", console.toggle)   # not typed into the search
        root.bind_all("<KeyPress-space>", self._on_space)
        root.bind_all("<Key>", self._on_typing_outside_fields)

    def _for_results(self, event):
        """Whether a key belongs to the results: pressed in the main window, not in a text field."""
        widget = event.widget
        return (isinstance(widget, tk.Misc) and not isinstance(widget, TEXT_FIELDS)
                and widget.winfo_toplevel() is self._root)

    def _on_copy(self, event):
        """Ctrl+C copies the selected files, except in a text field (it copies text there)."""
        if isinstance(event.widget, TEXT_FIELDS):
            return None
        if self._results.selected:
            self._copy_files(sorted(self._results.selected))
        return HANDLED

    def _on_paste_in_search(self, event):
        """Ctrl+V in the search field: text on the clipboard is pasted as usual, also when a
        picture came along with it; a picture alone is searched for (with the text in the field)."""
        if may_hold_picture() and not holds_text() and self._paste_picture():
            return HANDLED
        return None

    def _on_paste_outside_fields(self, event):
        """Ctrl+V on the results: search for the picture on the clipboard."""
        if not self._for_results(event):
            return None
        self._paste_picture()
        return HANDLED

    def _on_space(self, event):
        """Space: the large preview of the result under the mouse pointer (or the selected one)."""
        if not self._for_results(event):
            return None
        self._results.preview()
        return HANDLED

    def _on_console_key(self, event):
        """The console key opens or hides the console window; in a text field of a dialog (a
        password, say) it is typed as usual."""
        if isinstance(event.widget, TEXT_FIELDS):
            return None
        return self._console.toggle()

    def _on_typing_outside_fields(self, event):
        """Typing while the results have the keys (after a search) goes on in the search field."""
        if not self._for_results(event):
            return None
        entry = self._entry
        if event.keysym == BACKSPACE_KEY:
            entry.focus_set()
            entry.icursor(tk.END)
            if entry.get():
                entry.delete(entry.index(tk.END) - 1)
            return HANDLED
        if not event.char or not event.char.isprintable():
            return None   # Ctrl+letter, arrows, F-keys…
        entry.focus_set()
        entry.icursor(tk.END)
        entry.insert(tk.END, event.char)
        return HANDLED
