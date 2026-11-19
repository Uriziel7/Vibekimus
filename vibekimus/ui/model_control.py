"""The search model of the main window: loading it in the background (a newer choice makes an
older load obsolete), falling back to a model that is on this computer, and offering to
download one that is not."""
from threading import Lock, Thread
from tkinter import messagebox

from vibekimus.encoders.catalog import MODELS, create_encoder, model_by_key
from vibekimus.model_store import ModelStore
from vibekimus.runlog import log_event, safe_print
from vibekimus.ui import theme
from vibekimus.ui.model_download_dialog import ModelDownloadDialog
from vibekimus.ui.status_bar import DEVICE_PENDING

NO_MODEL_STATUS = "No search model on this computer yet"
NO_MODEL_LABEL = "No model — choose one in Settings"
NO_MODEL_MESSAGE = "There is no search model on this computer yet. Choose one in Settings to download it."
STILL_LOADING_MESSAGE = "Model is still loading..."
LOAD_FAILED_LABEL = "Load failed"
# (text found in the encoder's device name, label shown in the status line)
DEVICE_LABELS = (("CUDA", "GPU"), ("Metal", "MPS"), ("DirectML", "DirectML"))
DEFAULT_DEVICE_LABEL = "CPU"


class ModelControl:
    """preferences(): the current Preferences. on_switch(): another model was chosen — the window
    opens that model's indexes and calls start_loading(). before_change(): what belongs to the
    model being replaced is closed (Explore). schedule(action): runs it on the Tk thread."""

    def __init__(self, root, preferences, status, schedule, on_switch, before_change):
        self._root, self._preferences, self._status, self._schedule = root, preferences, status, schedule
        self._on_switch, self._before_change = on_switch, before_change
        self._store = ModelStore()
        self.encoder = None
        self._request = 0        # which load is the latest; older ones are discarded
        self._loading = Lock()   # one model loads at a time (two would need the memory of both)

    def is_available(self, key):
        return self._store.is_available(key)

    def offer_download(self, key, on_ready, on_cancel):
        """Ask whether to download a model that is not on this computer (on_ready(key) once it is)."""
        ModelDownloadDialog.confirm(self._root, self._store, key, on_ready=on_ready, on_cancel=on_cancel)

    def ensure_ready(self):
        """Whether the model is loaded; otherwise tells the user why not."""
        if self.encoder is not None:
            return True
        if not self.is_available(self._preferences().model.key):
            messagebox.showwarning("No model", NO_MODEL_MESSAGE)
        else:
            messagebox.showwarning("Wait", STILL_LOADING_MESSAGE)
        return False

    # ── choosing ─────────────────────────────────────────────────────────────
    def start_with_available(self):
        """The chosen model if it is on this computer, else another one that is; with none,
        the user picks one to download."""
        if self.is_available(self._preferences().model.key):
            self.start_loading()
            return
        available = next((spec for spec in MODELS if self.is_available(spec.key)), None)
        if available is not None:
            self.use(available.key)
            return
        self._status.set_status(NO_MODEL_STATUS, theme.STATUS_ERROR)
        ModelDownloadDialog.choose(self._root, self._store, on_ready=self.use,
                                   on_cancel=lambda: self._status.show_device(NO_MODEL_LABEL))

    def use(self, key):
        """Switch to a model that is now on this computer (and load it)."""
        model, preferences = model_by_key(key), self._preferences()
        if model == preferences.model:
            self.start_loading()
            return
        preferences.model = model
        preferences.save()
        self._on_switch()

    # ── loading ──────────────────────────────────────────────────────────────
    def start_loading(self):
        """Load the selected model in the background, replacing the current one. Choosing
        again while a model loads makes that load obsolete: it is thrown away when done."""
        self._before_change()
        previous, self.encoder = self.encoder, None   # nothing may use it while it changes
        self._request += 1
        preferences = self._preferences()
        if not self.is_available(preferences.model.key):
            if previous is not None:
                previous.unload()
            self._status.show_device(NO_MODEL_LABEL)
            self._status.set_status(NO_MODEL_STATUS, theme.STATUS_ERROR)
            return
        self._status.show_device(DEVICE_PENDING)
        Thread(target=self._load, args=(self._request, preferences.model, preferences.vram_save, previous),
               daemon=True).start()

    def _is_latest(self, request):
        return request == self._request

    def _load(self, request, spec, vram_save, previous):
        """Worker thread. The loaded model is handed to the Tk thread, which takes it only
        if no other model was chosen in the meantime."""
        with self._loading:
            if previous is not None:
                previous.unload()
            if not self._is_latest(request):
                return   # another model was chosen while this one waited for its turn
            self._schedule(lambda: self._status.set_status(f"Loading {spec.label}...", theme.STATUS_BUSY))
            try:
                encoder = create_encoder(spec, vram_save)   # imports torch — slow, so not at start-up
            except Exception as error:
                safe_print(f"[ERROR] {error}")
                log_event(f"MODEL LOAD FAILED: {spec.name}: {error}")
                self._schedule(lambda: self._failed(request, spec, error))
                return
            self._schedule(lambda: self._loaded(request, spec, encoder))

    def _loaded(self, request, spec, encoder):
        if not self._is_latest(request):
            safe_print(f"[LOAD] {spec.label} is no longer the chosen model; unloading it")
            Thread(target=encoder.unload, daemon=True).start()
            return
        self.encoder = encoder
        device = next((label for marker, label in DEVICE_LABELS if marker in encoder.device_name),
                      DEFAULT_DEVICE_LABEL)
        self._status.set_status("Ready", theme.STATUS_OK)
        self._status.show_device(f"{spec.label}  •  {device}")
        safe_print("[LOAD] Success!\n")

    def _failed(self, request, spec, error):
        if not self._is_latest(request):
            return
        self._status.set_status("Model load failed", theme.STATUS_ERROR)
        self._status.show_device(LOAD_FAILED_LABEL)
        messagebox.showerror("Error", f"Failed to load {spec.label}\n\n{error}")
