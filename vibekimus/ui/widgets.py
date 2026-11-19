"""Small reusable widgets."""
import tkinter as tk
from tkinter import ttk

from vibekimus.ui import theme

WINDOWS_WHEEL_DELTA = 120      # event.delta of one wheel notch on Windows
MIN_WRAP = 200                 # narrowest a wrapping label gets, pixels


class ControlGroup(ttk.Frame):
    """A row of controls that always stays together (it wraps as one unit)."""

    def button(self, text, command, style="TButton", width=None):
        widget = ttk.Button(self, text=text, command=command, style=style, width=width)
        return self.add(widget)

    def checkbox(self, text, value, command=None):
        variable = tk.BooleanVar(value=value)
        self.add(ttk.Checkbutton(self, text=text, variable=variable, command=command))
        return variable

    def add(self, widget):
        is_first = len(self.winfo_children()) == 1
        widget.pack(side="left", padx=(0 if is_first else theme.GAP, 0))
        return widget


class LabeledSlider(ttk.Frame):
    """Caption, slider and the current value of `variable`, snapped to `step`.

    The slider moves its own float position; only snapped values reach `variable`
    (so an IntVar never receives a fraction). Setting `variable` from code moves
    the slider and the shown value too.
    """

    def __init__(self, parent, caption, variable, low, high, step, length, value_format):
        super().__init__(parent)
        self._variable = variable
        self._low, self._step = low, step
        self._format = value_format
        self._position = tk.DoubleVar(value=variable.get())
        ttk.Label(self, text=caption).pack(side="left")
        ttk.Scale(self, from_=low, to=high, orient="horizontal", length=length,
                  variable=self._position, command=self._on_drag).pack(side="left", padx=theme.GAP)
        self._value = ttk.Label(self, width=len(value_format.format(high)) + 1, style=theme.MUTED_LABEL)
        self._value.pack(side="left")
        variable.trace_add("write", lambda *_trace: self._show_variable())
        self._show_variable()

    def _on_drag(self, raw_value):
        steps = round((float(raw_value) - self._low) / self._step)
        snapped = self._low + steps * self._step
        snapped = int(round(snapped)) if isinstance(self._variable, tk.IntVar) else round(snapped, 6)
        if self._variable.get() != snapped:
            self._variable.set(snapped)

    def _show_variable(self):
        value = self._variable.get()
        self._position.set(value)
        self._value.config(text=self._format.format(value))


class ButtonRow(ttk.Frame):
    """Buttons on the right of a dialog, replaced as the dialog moves on: show() takes
    (text, command, style or None) tuples, the first one leftmost."""

    def show(self, *buttons):
        for child in self.winfo_children():
            child.destroy()
        for text, command, style in reversed(buttons):
            ttk.Button(self, text=text, command=command, **({"style": style} if style else {})).pack(
                side="right", padx=(theme.GAP, 0))


def scrolled_listbox(parent):
    """A dark list with a scroll bar, packed to fill `parent`."""
    frame = tk.Frame(parent, bg=theme.BG)
    frame.pack(fill="both", expand=True)
    scrollbar = tk.Scrollbar(frame, bg=theme.PANEL_BG)
    scrollbar.pack(side="right", fill="y")
    listbox = tk.Listbox(frame, bg=theme.CARD_BG, fg=theme.FG, selectbackground=theme.ACCENT,
                         font=theme.FONT_NORMAL, yscrollcommand=scrollbar.set, highlightthickness=0,
                         relief="flat", activestyle="none")
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)
    return listbox


class ScrollableFrame(ttk.Frame):
    """Contents (put them in .body) that scroll up and down when they do not fit; labels
    made with wrapping_label() wrap to the frame's width as the window is resized."""

    def __init__(self, parent):
        super().__init__(parent)
        self._canvas = canvas = tk.Canvas(self, bg=theme.PANEL_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.body = ttk.Frame(canvas, padding=theme.PAD * 2)
        self._window = canvas.create_window((0, 0), window=self.body, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        self.body.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", self._fit_width)
        self._wrapping = []        # (label, pixels it is indented by)

    def wrapping_label(self, parent, text, indent=0, **options):
        label = ttk.Label(parent, text=text, justify="left", **options)
        self._wrapping.append((label, indent))
        return label

    def scroll(self, event):
        self._canvas.yview_scroll(int(-event.delta / WINDOWS_WHEEL_DELTA), "units")

    def scroll_to_top(self):
        self._canvas.yview_moveto(0)

    def _fit_width(self, event):
        self._canvas.itemconfig(self._window, width=event.width)
        room = event.width - 2 * theme.PAD * 2
        for label, indent in self._wrapping:
            label.config(wraplength=max(MIN_WRAP, room - indent))
