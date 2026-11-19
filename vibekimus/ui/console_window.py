"""The console window: what the program prints, as the console of Run.bat shows it (the
.exe has no console of its own). Opened and hidden with the console key
(settings.CONSOLE_TOGGLE_KEYS); it follows the output while it is open."""
import tkinter as tk
from tkinter import ttk

from vibekimus import settings
from vibekimus.ui import theme
from vibekimus.ui.windows import new_window

TITLE = "Vibekimus — console"
GEOMETRY = "1000x520"
FONT = ("Consolas", 10)
POLL_MS = 200
PARTIAL_MARK = "partial"      # where the line still being written starts
AT_BOTTOM = 0.999             # the view counts as following the output from here on
FIRST_LINE = "1.0"
LAST_CHAR = "end-1c"


class ConsoleWindow:
    def __init__(self, root, output):
        self._root, self._output = root, output
        self._window = self._text = None
        self._seen = 0            # completed lines shown
        self._poll_timer = None

    def toggle(self, event=None):
        if self._window is not None and self._window.winfo_viewable():
            self._hide()
        else:
            self._show()
        return "break"            # the key is not typed anywhere

    # ── showing and hiding ───────────────────────────────────────────────────
    def _show(self):
        if self._window is None:
            self._build()
        self._window.deiconify()
        self._window.lift()
        self._window.focus_set()
        self._update()

    def _hide(self):
        if self._poll_timer is not None:
            self._window.after_cancel(self._poll_timer)
            self._poll_timer = None
        self._window.withdraw()

    def _build(self):
        self._window = window = new_window(self._root, TITLE, background=theme.BG, geometry=GEOMETRY,
                                           transient=False, on_close=self._hide)
        scrollbar = ttk.Scrollbar(window, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        self._text = text = tk.Text(window, font=FONT, bg=theme.BG, fg=theme.FG, relief="flat", borderwidth=0,
                                    padx=theme.PAD, pady=theme.PAD, wrap="char", state="disabled",
                                    selectbackground=theme.ACCENT, yscrollcommand=scrollbar.set)
        text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text.yview)
        text.mark_set(PARTIAL_MARK, FIRST_LINE)
        text.mark_gravity(PARTIAL_MARK, "left")
        for key in settings.CONSOLE_TOGGLE_KEYS:
            window.bind(f"<KeyPress-{key}>", self.toggle)

    # ── following the output ─────────────────────────────────────────────────
    def _update(self):
        lines, partial, self._seen = self._output.since(self._seen)
        text = self._text
        following = text.yview()[1] >= AT_BOTTOM
        text.config(state="normal")
        text.delete(PARTIAL_MARK, LAST_CHAR)
        if lines:
            text.insert(LAST_CHAR, "".join(line + "\n" for line in lines))
        text.mark_set(PARTIAL_MARK, LAST_CHAR)
        text.insert(LAST_CHAR, partial)
        self._drop_oldest_lines()
        text.config(state="disabled")
        if following:
            text.see(LAST_CHAR)
        self._poll_timer = self._window.after(POLL_MS, self._update)

    def _drop_oldest_lines(self):
        extra = int(self._text.index(LAST_CHAR).split(".")[0]) - settings.CONSOLE_MAX_LINES - 1
        if extra > 0:
            self._text.delete(FIRST_LINE, f"{extra + 1}.0")
