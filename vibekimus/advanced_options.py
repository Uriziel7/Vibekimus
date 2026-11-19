"""The settings of settings.py that the Settings window offers under "Advanced": what each
one is, which values it takes, and changing them for the running program.

The defaults are the values in settings.py (some worked out from this computer, like the
number of threads); only values that differ from them are kept with the preferences.
Every option here is read by the program when it is used, so a change applies at once.
Paths, file names, the
security limits and the media formats stay in settings.py only.

Each kind of option is its own class: it knows which values it takes (parse) and how a
value is shown (text_of). A new kind = a new subclass (and its field in ui/advanced_settings)."""
import math
from dataclasses import dataclass

from vibekimus import settings
from vibekimus.storage import StorageMode

SWITCHED_ON, SWITCHED_OFF = "on", "off"
AUTOMATIC_PROGRAM = "automatic"


class InvalidValue(ValueError):
    """A value the option does not take; the message says which ones it does."""


@dataclass(frozen=True)
class AdvancedOption:
    name: str                 # settings.NAME
    group: str
    label: str
    note: str

    @property
    def default(self):
        return DEFAULTS[self.name]

    def parse(self, value):
        """The value as this option keeps it; text from a field is accepted. Raises InvalidValue."""
        raise NotImplementedError

    def text_of(self, value):
        """How a value is shown in the window."""
        return str(value)

    def _invalid(self, allowed):
        return InvalidValue(f"{self.label}: {allowed}")


@dataclass(frozen=True)
class NumberOption(AdvancedOption):
    minimum: float
    maximum: float
    whole = False
    kind_text = "a number"

    def parse(self, value):
        try:
            number = math.nan if isinstance(value, bool) else float(str(value).strip().replace(",", "."))
        except ValueError:
            number = math.nan
        if (not math.isfinite(number) or (self.whole and number != int(number))
                or not self.minimum <= number <= self.maximum):
            raise self._invalid(f"{self.kind_text} from {self.minimum:g} to {self.maximum:g}")
        return int(number) if self.whole else number

    def text_of(self, value):
        return f"{value:g}" if isinstance(value, float) else str(value)


@dataclass(frozen=True)
class IntegerOption(NumberOption):
    whole = True
    kind_text = "a whole number"


@dataclass(frozen=True)
class SwitchOption(AdvancedOption):
    def parse(self, value):
        if not isinstance(value, bool):
            raise self._invalid(f"{SWITCHED_ON} or {SWITCHED_OFF}")
        return value

    def text_of(self, value):
        return SWITCHED_ON if value else SWITCHED_OFF


@dataclass(frozen=True)
class ChoiceOption(AdvancedOption):
    choices: tuple            # (value, label)

    @property
    def labels(self):
        return [label for _value, label in self.choices]

    def parse(self, value):
        for choice, _label in self.choices:
            if value == choice or str(value) == str(choice):
                return choice
        raise self._invalid(f"one of {', '.join(self.labels)}")

    def value_labelled(self, label):
        """The value shown as `label`; anything else as it is (parse refuses it)."""
        return next((value for value, shown in self.choices if shown == label), label)

    def text_of(self, value):
        return next((label for choice, label in self.choices if choice == value), str(value))


@dataclass(frozen=True)
class ProgramOption(AdvancedOption):
    """A program's path; empty = found automatically."""

    def parse(self, value):
        if not isinstance(value, str):
            raise self._invalid("a program's path, or empty")
        return value.strip()

    def text_of(self, value):
        return value or AUTOMATIC_PROGRAM


INDEXING, TEXT, SEARCH, ALL_DRIVES, WINDOW = ("Indexing", "Text on pictures (OCR)", "Search",
                                              "Scan of all drives", "Window")
OPTIONS = (
    ChoiceOption("STORAGE_MODE", INDEXING, "Drives",
                 "Automatic: a spinning hard disk is read one file at a time (its head would jump back and "
                 "forth otherwise), SSDs in parallel. Windows is asked about each drive.",
                 choices=((StorageMode.AUTO, "Automatic"), (StorageMode.SSD, "Every drive as an SSD"),
                          (StorageMode.HDD, "Every drive as a hard disk"))),
    IntegerOption("LOAD_WORKERS", INDEXING, "Threads reading pictures",
                  "Read and decode pictures while the graphics card encodes. More helps only up to the "
                  "number of CPU threads.", 1, 64),
    IntegerOption("VIDEO_WORKERS", INDEXING, "Threads reading videos", "Videos decoded at the same time.", 1, 16),
    IntegerOption("AUTOSAVE_SECONDS", INDEXING, "Autosave every (seconds)",
                  "While indexing, what is new is saved this often (a small extra file, cheap for an SSD).",
                  30, 3600),
    IntegerOption("VIDEO_FRAME_INTERVAL", INDEXING, "Seconds between video frames",
                  "Frames taken from each video at least this far apart; long videos are spread further.", 1, 600),
    IntegerOption("MAX_FRAMES_PER_VIDEO", INDEXING, "Frames per video at most",
                  "More frames find shorter moments, and make the video index bigger.", 1, 1000),
    IntegerOption("VIDEO_TIME_BUDGET_SECONDS", INDEXING, "Seconds per video at most",
                  "A video that reads this slowly (badly made files) keeps the frames read so far.", 1, 600),
    IntegerOption("FREE_CPU_CORES", INDEXING, "CPU cores left free",
                  "Indexing never uses these cores (the last ones) and runs below normal priority, so the computer "
                  "stays responsive. At least one core is always used.", 0, 64),
    SwitchOption("OCR_ON_GPU", TEXT, "Read text on the graphics card",
                 "About twice as fast as the CPU, the same text."),
    IntegerOption("OCR_WORKERS", TEXT, "Pictures read at the same time by each process",
                  "Each shares the CPU with the others. The number of processes follows the CPU cores and "
                  "the free graphics memory.", 1, 16),
    NumberOption("OCR_VRAM_RESERVE_GB", TEXT, "Graphics memory kept free (GB)",
                 "Reading text starts only the processes that fit above it; when other programs take the "
                 "card's memory below it, one process finishes its pictures and ends.", 0.0, 16.0),
    IntegerOption("MAX_SEARCH_RESULTS", SEARCH, "Results kept per search", "The best matches shown at most.",
                  100, 20000),
    NumberOption("COMBINED_TEXT_WEIGHT", SEARCH, "Text's share with a picture",
                 "Searching with a picture and text together: 0 = the picture alone, 1 = the text alone.", 0.0, 1.0),
    NumberOption("MOMENT_MATCH_SPREAD", SEARCH, "Matching moments of a video: closeness to its best",
                 "How close (in spreads of all scores) a moment's score must be to the video's best moment to "
                 "count as matching. Larger = more moments.", 0.0, 10.0),
    NumberOption("LIKED_WEIGHT", SEARCH, "Refine: weight of ▲ more like this",
                 "How strongly results rated ▲ pull the refined search (the search itself weighs 1).", 0.0, 10.0),
    NumberOption("DISLIKED_WEIGHT", SEARCH, "Refine: weight of ▼ less like this",
                 "How strongly results rated ▼ push the refined search away (the search itself weighs 1).",
                 0.0, 10.0),
    IntegerOption("EXPLORE_TOPICS", SEARCH, "Explore: topics at most",
                  "How many topics Explore groups the pictures into (fewer for a small library).", 2, 200),
    NumberOption("OCR_WORD_BONUS", SEARCH, "Boost for words seen on a picture",
                 "How far up a picture that shows the typed words moves (0 = not at all).", 0.0, 10.0),
    SwitchOption("GPU_SEARCH", SEARCH, "Search on the graphics card",
                 "Keeps the index on the graphics card for fast searches (otherwise in RAM, slower)."),
    SwitchOption("ALL_DRIVES_SCAN_SYSTEM_FOLDERS", ALL_DRIVES, "Scan Windows and program folders",
                 "They hold hundreds of thousands of icons and textures that are not your media."),
    IntegerOption("ALL_DRIVES_MIN_IMAGE_KB", ALL_DRIVES, "Skip pictures smaller than (KB)",
                  "Icons and thumbnails; 0 = keep everything.", 0, 10000),
    IntegerOption("DOUBLE_CLICK_MS", WINDOW, "Double click within (ms)",
                  "A single click selects a result only after this time, so it is kept short.", 50, 1000),
    ProgramOption("VIDEO_PLAYER", WINDOW, "Video player",
                  "Opens video results at the moment that matched. Empty = automatic (PotPlayer, VLC, "
                  "MPC-HC or mpv)."),
)
BY_NAME = {option.name: option for option in OPTIONS}
DEFAULTS = {option.name: getattr(settings, option.name) for option in OPTIONS}   # before any change


def valid_changes(saved):
    """The saved changes that are still options and still valid (anything else is dropped)."""
    changes = {}
    for name, value in (saved or {}).items():
        option = BY_NAME.get(name)
        if option is None:
            continue
        try:
            value = option.parse(value)
        except InvalidValue:
            continue
        if value != option.default:
            changes[name] = value
    return changes


def current():
    """Every option's value in the running program (for a worker process to apply)."""
    return {option.name: getattr(settings, option.name) for option in OPTIONS}


def apply(changes):
    """settings.py's values, with these changes, for the running program."""
    for option in OPTIONS:
        setattr(settings, option.name, changes.get(option.name, option.default))
