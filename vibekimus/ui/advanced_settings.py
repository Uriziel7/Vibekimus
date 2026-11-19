"""The Advanced page of the Settings window: the options of advanced_options, grouped,
each with what it does and its default; "Restore defaults" puts them all back."""
import tkinter as tk
from tkinter import filedialog, ttk

from vibekimus.advanced_options import (OPTIONS, ChoiceOption, InvalidValue, NumberOption, ProgramOption,
                                        SwitchOption)
from vibekimus.ui import theme

NUMBER_FIELD_WIDTH = 10
PROGRAM_FIELD_WIDTH = 46
CHOICE_FIELD_PADDING = 2   # characters beside the longest label
NOTE_INDENT = theme.GROUP_GAP
PROGRAM_TYPES = [("Programs", "*.exe"), ("All files", "*.*")]
INTRO = "Settings for fine-tuning; the defaults suit most computers. Each applies as soon as it is saved."


class _Field:
    """The input of one option: get() gives what the user entered, set() shows a value."""

    def __init__(self, page, parent, option, value):
        self.option = option
        self._row = row = ttk.Frame(parent)
        row.pack(fill="x", pady=(theme.GAP, 0))
        self._variable = self._build(row)
        page.wrapping_label(parent, f"{option.note} Default: {option.text_of(option.default)}.", indent=NOTE_INDENT,
                            style=theme.MUTED_LABEL).pack(anchor="w", padx=(NOTE_INDENT, 0))
        self.set(value)

    def _build(self, row):
        """The option's label and input in `row`; returns the input's variable."""
        raise NotImplementedError

    def set(self, value):
        self._variable.set(self.option.text_of(value))

    def get(self):
        """The value entered. Raises InvalidValue."""
        return self.option.parse(self._variable.get())


class _SwitchField(_Field):
    def _build(self, row):
        variable = tk.BooleanVar()
        ttk.Checkbutton(row, text=self.option.label, variable=variable).pack(side="left")
        return variable

    def set(self, value):
        self._variable.set(bool(value))


class _TextField(_Field):
    """A label on the left, the input on the right."""

    def _build(self, row):
        ttk.Label(row, text=self.option.label).pack(side="left")
        variable = tk.StringVar()
        self._make_input(row, variable)
        return variable

    def _make_input(self, row, variable):
        raise NotImplementedError


class _NumberField(_TextField):
    def _make_input(self, row, variable):
        ttk.Entry(row, textvariable=variable, width=NUMBER_FIELD_WIDTH, justify="right").pack(side="right")


class _ChoiceField(_TextField):
    def _make_input(self, row, variable):
        labels = self.option.labels
        ttk.Combobox(row, textvariable=variable, state="readonly", values=labels,
                     width=max(map(len, labels)) + CHOICE_FIELD_PADDING).pack(side="right")

    def get(self):
        return self.option.parse(self.option.value_labelled(self._variable.get()))


class _ProgramField(_TextField):
    def _make_input(self, row, variable):
        ttk.Button(row, text="Browse…", command=self._browse).pack(side="right", padx=(theme.GAP, 0))
        ttk.Entry(row, textvariable=variable, width=PROGRAM_FIELD_WIDTH).pack(side="right")

    def _browse(self):
        chosen = filedialog.askopenfilename(parent=self._row, title=self.option.label, filetypes=PROGRAM_TYPES)
        if chosen:
            self._variable.set(chosen)

    def set(self, value):
        self._variable.set(value)   # the path itself; empty stays empty (text_of would say "automatic")


FIELDS = {SwitchOption: _SwitchField, NumberOption: _NumberField, ChoiceOption: _ChoiceField,
          ProgramOption: _ProgramField}   # kind of option -> its input (subclasses use their base's)


def field_class(option):
    return next(FIELDS[kind] for kind in type(option).__mro__ if kind in FIELDS)


class AdvancedSettingsPage:
    """page: a widgets.ScrollableFrame; changes: {settings name: value} saved so far."""

    def __init__(self, page, changes):
        body = page.body
        page.wrapping_label(body, INTRO, style=theme.MUTED_LABEL).pack(anchor="w")
        self._fields = []
        group = None
        for option in OPTIONS:
            if option.group != group:
                group = option.group
                ttk.Label(body, text=group, font=theme.FONT_HEADING).pack(anchor="w", pady=(theme.PAD * 2, 0))
            self._fields.append(field_class(option)(page, body, option, changes.get(option.name, option.default)))
        ttk.Button(body, text="Restore defaults", command=self.restore_defaults).pack(
            anchor="w", pady=(theme.PAD * 2, 0))

    def restore_defaults(self):
        for field in self._fields:
            field.set(field.option.default)

    def changes(self):
        """{settings name: value} of what differs from the defaults. Raises InvalidValue
        naming every option with a value it does not take."""
        changes, errors = {}, []
        for field in self._fields:
            try:
                value = field.get()
            except InvalidValue as error:
                errors.append(str(error))
                continue
            if value != field.option.default:
                changes[field.option.name] = value
        if errors:
            raise InvalidValue("\n".join(errors))
        return changes
