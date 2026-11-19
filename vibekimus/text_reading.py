"""Reading the text on pictures (OCR) with PP-OCRv6 small (PaddleOCR, Apache 2.0),
run by RapidOCR on ONNX Runtime: on the graphics card through DirectML when
settings.OCR_ON_GPU allows it, else on the CPU.

DirectML runs one model at a time in a process: two engines running at once crash it. So
the engines' runs on the graphics card take turns (one lock); what each engine does on the
CPU around them (cutting out lines, decoding the results) still goes on side by side.

Every engine uses the same pair of models on the graphics card, made by the first engine
and freed when the last one is closed. Two reasons: (1) with ONNX Runtime 1.24.4 freeing one
DirectML model breaks the others in the process — their next run crashes (Stop during OCR
crashed the program as each worker freed its own engine); freeing all of them together is
fine; (2) each pair keeps memory for every picture size it has seen: ~3.8 GB after a few
hundred pictures, and one pair per engine grew past 14 GB.

The models come from Hugging Face's local cache only; nothing is downloaded here.
The recognition model's ONNX file has no character list built in (RapidOCR would
then fetch one from the internet), so the list is taken from the model's own
inference.yml and handed over as a file."""
import copy
import os
import threading

import numpy as np
import yaml

from vibekimus import settings
from vibekimus.files import write_atomically
from vibekimus.runlog import safe_print

DETECTION_REPO = "PaddlePaddle/PP-OCRv6_small_det_onnx"
RECOGNITION_REPO = "PaddlePaddle/PP-OCRv6_small_rec_onnx"
MODEL_FILE, CONFIG_FILE = "inference.onnx", "inference.yml"
MODEL_NAME = "PP-OCRv6 small"
MISSING_PACKAGES = ("Reading text needs the packages rapidocr and onnxruntime.\n"
                    "Install them with:\n    pip install onnxruntime\n    pip install --no-deps rapidocr")
MISSING_MODEL = (f"The text reading model ({MODEL_NAME}, about 31 MB) is not downloaded yet.\n"
                 f"It is downloaded once from Hugging Face: {DETECTION_REPO} and {RECOGNITION_REPO}.")
LINE_SEPARATOR = "\n"
MIN_LINE_CHARS = 2
CHARACTER_FILE = "PP-OCRv6_small_characters.txt"   # in settings.MODELS_DIR
# On Windows a file being replaced while another thread or process reads or replaces it
# fails with "access denied": the program writes it before the reading processes start
# (check_available), and engines in threads of one process take turns.
_character_file_lock = threading.Lock()
DIRECTML = "DmlExecutionProvider"
GPU_NAME = "graphics card (DirectML)"
CPU_NAME = "CPU"
_gpu_turn = threading.Lock()   # making, running or freeing the models on the graphics card


class TextReaderUnavailable(RuntimeError):
    """The OCR packages or model files are missing."""


def _cached(repo, name):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        return hf_hub_download(repo, name, local_files_only=True)
    except LocalEntryNotFoundError as error:
        raise TextReaderUnavailable(MISSING_MODEL) from error


def _read_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _directml_available():
    import onnxruntime
    return DIRECTML in onnxruntime.get_available_providers()


def _gpu_wanted():
    """settings.OCR_ON_GPU, and a DirectML build of ONNX Runtime to do it."""
    try:
        return settings.OCR_ON_GPU and _directml_available()
    except ImportError:
        return False


def reading_device():
    return GPU_NAME if _gpu_wanted() else CPU_NAME


class _TakingTurns:
    """An ONNX Runtime session whose runs wait for those of every other engine."""

    def __init__(self, session):
        self._session = session

    def run(self, *args, **kwargs):
        with _gpu_turn:
            return self._session.run(*args, **kwargs)

    def close(self):
        """Free the model (this holds the only reference to it) while nothing else runs."""
        with _gpu_turn:
            self._session = None

    def __getattr__(self, name):   # inputs, outputs, metadata: those of the session
        return getattr(self._session, name)


class _SharedGpuModels:
    """The detection and recognition every engine uses on the graphics card. The first engine
    makes them; the others get copies that share the models but keep their own state of a
    call (RapidOCR's detector sets its preprocessing on itself at every picture)."""

    def __init__(self):
        self._parts = None   # (detector, recognizer)
        self._users = 0
        self._lock = threading.Lock()

    def join(self, engine):
        """Give the RapidOCR engine the shared models. False when ONNX Runtime made them on the
        CPU after all (then the engine keeps its own, as on the CPU)."""
        with self._lock:
            if self._parts is None:
                parts = self._make(engine)
                if parts is None:
                    return False
                self._parts = parts
            engine.text_det, engine.text_rec = (copy.copy(part) for part in self._parts)
            self._users += 1
            return True

    @staticmethod
    def _make(engine):
        with _gpu_turn:
            parts = (engine._load_det_model(), engine._load_rec_model())
        if not all(DIRECTML in part.session.session.get_providers() for part in parts):
            return None
        for part in parts:   # shared by the copies, so the wrapper holds the only reference
            part.session.session = _TakingTurns(part.session.session)
        return parts

    def leave(self):
        with self._lock:
            self._users -= 1
            if self._users == 0:
                for part in self._parts:
                    part.session.session.close()
                self._parts = None


_shared_gpu_models = _SharedGpuModels()


class TextReader:
    """One OCR engine; use one per thread. threads: CPU threads it may use."""

    def __init__(self, threads):
        try:
            from rapidocr import RapidOCR
        except ImportError as error:
            raise TextReaderUnavailable(MISSING_PACKAGES) from error
        detection = _read_yaml(_cached(DETECTION_REPO, CONFIG_FILE))
        characters = character_file()
        on_gpu = _gpu_wanted()
        self._engine = RapidOCR(params=self._params(detection, characters, threads, on_gpu))
        self.on_gpu = on_gpu and self._share_gpu_models()

    def _share_gpu_models(self):
        """Use the models all engines share on the graphics card (RapidOCR would make its own
        at the first picture). False when ONNX Runtime fell back to the CPU."""
        if _shared_gpu_models.join(self._engine):
            return True
        safe_print("[TEXT] The graphics card cannot read text here (DirectML); reading on the CPU")
        return False

    def close(self):
        """Done reading: the graphics card's memory goes back when the last engine is closed."""
        if self.on_gpu:
            self.on_gpu = False
            _shared_gpu_models.leave()

    @staticmethod
    def _params(detection, characters, threads, on_gpu):
        """The models' own settings (from their inference.yml) instead of RapidOCR's
        defaults, which belong to its own converted models."""
        normalize = next(op["NormalizeImage"] for op in detection["PreProcess"]["transform_ops"]
                         if "NormalizeImage" in op)
        post = detection["PostProcess"]
        return {
            "Global.use_cls": False,   # turning upside-down text would need one more model
            "Global.log_level": "error",   # it warns about every picture without text otherwise
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.use_dml": on_gpu,
            "Det.model_path": _cached(DETECTION_REPO, MODEL_FILE),
            "Det.mean": normalize["mean"], "Det.std": normalize["std"],
            "Det.thresh": post["thresh"], "Det.box_thresh": post["box_thresh"],
            "Det.unclip_ratio": post["unclip_ratio"],
            "Rec.model_path": _cached(RECOGNITION_REPO, MODEL_FILE),
            "Rec.rec_keys_path": characters,
        }

    def read(self, image):
        """The text on a PIL image, line by line ("" when there is none)."""
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        lines = self._engine(bgr).txts or ()
        # A lone character is mostly a shape taken for a letter (a circle read as "O")
        # and could never be searched for anyway.
        return LINE_SEPARATOR.join(line for line in lines if len(line.strip()) >= MIN_LINE_CHARS)


def character_file():
    """The recognition model's character list as a file in the program's data (RapidOCR reads
    it only when the first text is recognised, so it must stay)."""
    recognition = _read_yaml(_cached(RECOGNITION_REPO, CONFIG_FILE))
    return write_character_file(recognition["PostProcess"]["character_dict"])


def write_character_file(characters):
    """The file holding these characters, written only when it holds something else."""
    content = LINE_SEPARATOR.join(characters).encode("utf-8")
    path = os.path.join(settings.MODELS_DIR, CHARACTER_FILE)
    with _character_file_lock:
        if not _holds(path, content):
            os.makedirs(settings.MODELS_DIR, exist_ok=True)
            write_atomically(path, content)
    return path


def _holds(path, content):
    try:
        with open(path, "rb") as f:
            return f.read() == content
    except FileNotFoundError:
        return False


def check_available():
    """Raises TextReaderUnavailable when reading text cannot work here."""
    try:
        import rapidocr  # noqa: F401
    except ImportError as error:
        raise TextReaderUnavailable(MISSING_PACKAGES) from error
    for repo in (DETECTION_REPO, RECOGNITION_REPO):
        for name in (MODEL_FILE, CONFIG_FILE):
            _cached(repo, name)
    character_file()   # now, before reading processes start (several writing it at once fail)
    safe_print(f"[TEXT] {MODEL_NAME} found")
