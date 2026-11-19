"""Everything a user may want to tune, in one place.

Other modules read these as ``settings.NAME`` at the moment they need the value, so a
change here (or in a test) takes effect everywhere.
"""
import os
import sys

# Privacy: switch off Hugging Face telemetry. Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

# Run as the single .exe (PyInstaller): its own files are unpacked to a temporary folder,
# so everything it writes (libraries, models, preferences, logs) goes to a data folder.
FROZEN = getattr(sys, "frozen", False)
DATA_DIR_NAME = "Vibekimus"
DATA_DIR_VARIABLE = "VIBEKIMUS_DATA_DIR"   # overrides the data folder (tests, a portable copy)
if os.environ.get(DATA_DIR_VARIABLE):
    APP_DIR = os.path.abspath(os.environ[DATA_DIR_VARIABLE])
elif FROZEN:
    APP_DIR = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), DATA_DIR_NAME)
else:
    APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The .exe keeps the Hugging Face cache (model downloads) in its data folder too.
if FROZEN or os.environ.get(DATA_DIR_VARIABLE):
    os.environ.setdefault("HF_HOME", os.path.join(APP_DIR, "huggingface"))

# ─── Model ──────────────────────────────────────────────────────────────────
# The model is chosen in the program's Settings window; the models on offer are
# listed in vibekimus/encoders/catalog.py. This one is used until something is chosen.
DEFAULT_MODEL = "siglip2-so400m"

# False = the Hugging Face libraries are put in offline mode before they are even
# imported, and the app has no other code that uses the network. Set to True only
# for the one run that downloads a new model.
ALLOW_MODEL_DOWNLOAD = False
if not ALLOW_MODEL_DOWNLOAD:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ─── Media types ────────────────────────────────────────────────────────────
RAW_EXTS = (".cr2", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".pef", ".sr2")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif") + RAW_EXTS
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
              ".wmv", ".flv", ".ts", ".mpg", ".mpeg", ".3gp", ".vob")
# Formats PIL may detect from the file content when the extension is wrong. A fixed
# list of plain raster formats: PIL's EPS reader, for one, would start Ghostscript.
IMAGE_SNIFF_FORMATS = ("JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF", "PSD")
# PIL refuses anything above 2x this value (decompression-bomb guard).
MAX_IMAGE_PIXELS = 500_000_000

# ─── Video sampling ─────────────────────────────────────────────────────────
VIDEO_FRAME_INTERVAL = 5       # minimum seconds between sampled frames
MAX_FRAMES_PER_VIDEO = 50      # cap per video — the interval scales up for long videos
# Videos longer than this many seconds are not indexed (films, recordings of several
# hours), until changed in the program's Settings window. 0 = no limit. A video whose
# length cannot be read is indexed by its first minute.
VIDEO_MAX_DURATION_SECONDS = 60
# Whether Refresh and scans also index videos, until the "Index videos" box in the
# program is changed (the choice is remembered).
DEFAULT_INDEX_VIDEOS = False
# Whether Refresh and scans also read the text on pictures (OCR, model PP-OCRv6 small),
# until the "Read text (OCR)" box is changed (the choice is remembered).
DEFAULT_READ_TEXT = False
# Whether index files (paths, embeddings, file details, text read from pictures,
# exclusions) are encrypted, until changed in Library → Index encryption.
DEFAULT_ENCRYPT_INDEXES = False
MIN_PASSWORD_CHARS = 8
# Thread counts below are worked out from the CPU (logical processors; this many when unknown).
CPU_THREADS = os.cpu_count() or 8
# Pictures read at the same time by each OCR process; each uses a share of the CPU:
# one per 4 CPU threads, 1 to 8.
OCR_WORKERS = max(1, min(8, CPU_THREADS // 4))
# Text is read by several processes at once, as many as the CPU share and the free graphics
# memory allow at the time (ocr_capacity.OcrProcessBudget). In one process the graphics card
# runs one OCR model at a time (DirectML), so more processes keep it busier: measured on real
# pictures 11.8 pictures/s with 1, 20.5 with 2, 38.2 with 4.
MAX_OCR_PROCESSES = 8
# Graphics memory one OCR process keeps (its models' buffers for every picture size seen:
# 2-2.4 GB after a minute, 3.8 GB after hundreds of pictures of every size).
OCR_PROCESS_VRAM_GB = 4.0
# CPU threads each OCR process keeps busy, of those it may use (FREE_CPU_CORES left out):
# with 12 of 16 threads, 4 processes kept 62% of them busy, 5 processes 79%.
OCR_CPU_THREADS_PER_PROCESS = 3
# Graphics memory kept free while text is read: a pass starts only the processes that fit
# above it, and when the free memory drops below it (a game, another program) one process
# finishes the pictures it is reading and ends, giving its memory back. Checked this often.
OCR_VRAM_RESERVE_GB = 1.5
OCR_VRAM_CHECK_SECONDS = 2.0
# A process is added back no sooner than this after the last change (a new one takes a
# while to reach its full graphics memory).
OCR_GROW_WAIT_SECONDS = 30.0
# Read text on the graphics card (DirectML) when ONNX Runtime can:
# about twice as fast as the CPU, the same text. False = always the CPU.
OCR_ON_GPU = True
# Searching by text: a file whose picture shows all typed words gets this many standard
# deviations of the search scores on top (fewer words found, proportionally less).
OCR_WORD_BONUS = 3.0
# A video whose frames take longer than this to read (badly indexed files make every
# seek decode from the start) keeps the frames read so far and is not sampled further.
VIDEO_TIME_BUDGET_SECONDS = 10
# Video frames are shrunk to this longest side right after decoding (saves RAM).
VIDEO_FRAME_MAX_SIDE = 768

# ─── Speed ──────────────────────────────────────────────────────────────────
# How files are read while indexing: "auto" = Windows is asked about each drive, and a
# spinning hard disk is read one file at a time (its head would jump back and forth
# otherwise) while SSDs are read in parallel; "hdd" or "ssd" decide for every drive.
STORAGE_MODE = "auto"
# Threads that read + decode + resize images while the GPU encodes: all CPU threads but 2,
# 4 to 32.
LOAD_WORKERS = max(4, min(32, CPU_THREADS - 2))
# Physical CPU cores indexing never uses (with SMT both of their threads), so the computer
# stays responsive: while it indexes the program runs on the other cores at below normal
# priority (pictures, videos, and the text reading processes).
FREE_CPU_CORES = 2
# Threads that decode video frames in parallel: one per 4 CPU threads, 2 to 8 (one at a
# time per hard disk anyway).
VIDEO_WORKERS = max(2, min(8, CPU_THREADS // 4))
# Images sent to the GPU per forward pass. Larger values use more VRAM but barely
# change speed: the GPU is limited by how fast files are decoded, not by memory.
STREAM_CHUNK = 256
# Seconds between automatic saves while indexing. Each one writes only the entries
# added since the previous save (a small ".partNNNNN" file), so it is cheap for the
# SSD; the pieces are merged into the main index file when the run ends.
AUTOSAVE_SECONDS = 300

# Where files are decoded while indexing, until changed in the program's Settings
# window: "cpu", "gpu" or "cpu+gpu". Measured on an RTX 5090 + 8-core CPU:
# images — CPU fastest for camera photos (JPEGs decode at reduced size on the CPU,
# always at full size on the GPU), CPU+GPU about 5% faster on a PNG-heavy mix;
# videos — the GPU video decoder about 18% faster.
DEFAULT_IMAGE_DECODING = "cpu"
DEFAULT_VIDEO_DECODING = "gpu"

# ─── Video playback ─────────────────────────────────────────────────────────
# Program that opens a video result at the moment that matched, e.g.
# r"C:\Program Files\VideoLAN\VLC\vlc.exe". Empty = automatic: the program Windows
# uses for videos if it is PotPlayer, VLC, MPC-HC or mpv, otherwise the first of
# these that is installed.
VIDEO_PLAYER = ""

# ─── Search ─────────────────────────────────────────────────────────────────
# Keep the index on the GPU and compute similarities there (falls back to CPU).
GPU_SEARCH = True
# A search keeps at most this many best matches.
MAX_SEARCH_RESULTS = 2000
# Starting position of the Similarity Score slider. Results are always ranked best
# first; the slider only trims the tail. SigLIP text scores are much lower than the
# old CLIP model's, so the old default of 0.15 would hide nearly everything.
DEFAULT_MIN_SCORE = 0.0
# Searching with a picture and text together ("this, but at night"): the share of the
# text in the result, 0–1; the rest is how much a file looks like the picture.
COMBINED_TEXT_WEIGHT = 0.5
# A moment of a video counts as matching the search (orange on its timeline) when its
# score is at most this many standard deviations of all frames' scores below the video's
# best moment (and the moment passes Min. score and the filters).
MOMENT_MATCH_SPREAD = 1.0
# Refining a search with ratings (▲ more like this / ▼ less like this, then Refine): how
# much the liked and the disliked results weigh against the search itself (which weighs 1).
LIKED_WEIGHT = 1.0
DISLIKED_WEIGHT = 0.5
# Explore: the pictures grouped into at most this many topics (fewer when a topic would
# get under EXPLORE_MIN_TOPIC_SIZE pictures on average); grouping rounds at most.
EXPLORE_TOPICS = 36
EXPLORE_MIN_TOPIC_SIZE = 8
EXPLORE_ITERATIONS = 30
# Starting position of the Results Per Page slider.
DEFAULT_RESULTS_PER_PAGE = 30
# Thumbnail sizes (longest side, pixels) that Ctrl + mouse wheel steps through over the
# results, smallest first, and the starting one (the choice is remembered).
THUMBNAIL_ZOOM_LEVELS = (100, 120, 140, 160, 180, 220, 260, 320, 400, 500)
DEFAULT_THUMBNAIL_SIZE = 180

# VRAM Save (Settings window): the model's weights kept in 16 bits on the graphics card,
# about half the graphics memory, about 5% slower, results practically the same.
DEFAULT_VRAM_SAVE = False

# How soon (ms) a second click must follow the first to make a double click on a result.
# A single click (select) takes effect only after this time, so it is kept short.
DOUBLE_CLICK_MS = 90

# ─── Whole-computer scan ("All Drives" button) ──────────────────────────────
# Drives or folders to scan, e.g. ["D:\\", "E:\\Photos"]. Empty = every fixed local drive.
ALL_DRIVES_ROOTS = []
# The combined index is kept here, next to the program, not on the scanned drives.
ALL_DRIVES_INDEX_DIR = os.path.join(APP_DIR, "index_all_drives")
# Windows, program and app-data folders hold hundreds of thousands of icons, textures
# and thumbnails that are not your media. Set to True to scan them anyway.
ALL_DRIVES_SCAN_SYSTEM_FOLDERS = False
# Images smaller than this are skipped in an all-drives scan (icons, thumbnails). 0 = keep all.
ALL_DRIVES_MIN_IMAGE_KB = 10
# Folder names skipped only directly under a drive root (lower case):
SYSTEM_ROOT_DIRS = {
    "windows", "windows.old", "program files", "program files (x86)", "programdata",
    "recovery", "perflogs", "msocache", "$windows.~bt", "$windows.~ws", "$winreagent",
    "config.msi", "documents and settings",
}
# Folder names skipped at any depth (lower case):
SYSTEM_ANY_DIRS = {
    "$recycle.bin", "system volume information", "appdata", "node_modules",
    "site-packages", "__pycache__", ".git",
}

# ─── Files written by the program ───────────────────────────────────────────
# Run log, next to the program: configuration, speed reports and errors only.
LOG_FILE = os.path.join(APP_DIR, "vibekimus_log.txt")
# Choices made in the program's Settings window.
PREFERENCES_FILE = os.path.join(APP_DIR, "vibekimus_preferences.json")
# The index encryption key, itself protected by the Windows account or a password.
# Losing it (and its export) makes encrypted indexes unreadable: index again.
KEY_FILE = os.path.join(APP_DIR, "vibekimus_key.json")
# Compact copies of downloaded model weights (SigLIP 2: 2.9 GB instead of 4.5 GB, the same
# results); written once after the download, which is then removed from the cache.
MODELS_DIR = os.path.join(APP_DIR, "models")
# The .exe: the libraries it downloads on first start (PyTorch and the rest, ~4.9 GB).
RUNTIME_DIR = os.path.join(APP_DIR, "runtime")
# The .exe without a console: what the program prints goes here.
CONSOLE_FILE = os.path.join(APP_DIR, "vibekimus_console.txt")
# The console window (what Run.bat's console shows, also in the .exe): the keys that open and
# hide it (Tk key names: ` and ~), and how many of the last lines it keeps.
CONSOLE_TOGGLE_KEYS = ("grave", "asciitilde")
CONSOLE_MAX_LINES = 5000
INDEX_FILE_PREFIX = ".clip_cache_"
INDEX_FILE_SUFFIX = ".npz"   # numpy archive, loaded with allow_pickle=False (no code execution)
EXCLUSIONS_FILE = ".clip_exclusions.json"
# Indexes of one or more folders made in the program ("Library → New index…"): each in its
# own folder here, with its name and folders in LIBRARY_FILE. Nothing is written to the
# indexed folders.
INDEXES_DIR = os.path.join(APP_DIR, "indexes")
LIBRARY_FILE = ".vibekimus_library.json"
# Indexes kept inside a folder (made by older versions) that the program has seen, for the
# "Open saved index" list.
KNOWN_INDEXES_FILE = os.path.join(APP_DIR, "vibekimus_known_indexes.json")
# The index that was open last; the program opens it again when it starts.
LAST_INDEX_FILE = os.path.join(APP_DIR, "vibekimus_last_index.json")
# Text read from the pictures (OCR); the same for every model. No .npz extension, so the
# "Open saved index" dialog does not offer it.
TEXT_INDEX_FILE = ".vibekimus_ocr_text"
SKIPPED_IMAGES_LOG = "vibekimus_skipped_images.txt"
SKIPPED_VIDEOS_LOG = "vibekimus_skipped_videos.txt"
SKIPPED_TEXT_LOG = "vibekimus_skipped_text.txt"
