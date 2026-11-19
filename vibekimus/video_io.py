"""Reading frames from video files with OpenCV."""
import contextlib
import itertools
import os
import sys
import time
from dataclasses import dataclass, field

from PIL import Image

from vibekimus import settings
from vibekimus.decoding import DecodeMode
from vibekimus.media_info import UNKNOWN
from vibekimus.runlog import STDERR_FD, safe_print
from vibekimus.units import MILLISECONDS_PER_SECOND, SECONDS_PER_MINUTE

ASSUMED_FPS = 25.0                 # when the container does not report a frame rate
ASSUMED_DURATION_SECONDS = 60.0    # when the container does not report a frame count
# Frames darker than this (mean of 0-255) are fades and black intros — not indexed.
MIN_FRAME_BRIGHTNESS = 10
BRIGHTNESS_SAMPLE_STEP = 8         # measure brightness on every 8th pixel
OPENCV_LOG_SILENT = 0
TIMESTAMP_DECIMALS = 3
NO_TIME_STAMP = float(-2 ** 63)    # ffmpeg's AV_NOPTS_VALUE, as OpenCV reports it
HALF_FRAME = 0.5                   # OpenCV rounds a moment to the nearest frame


def load_opencv():
    """Import OpenCV configured for decoding only. Raises ImportError if not installed."""
    # ffmpeg warnings ("Missing reference picture" ...) are harmless artifacts of
    # seeking to non-keyframes; they must be silenced before any capture is opened.
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    if os.name != "nt" and sys.platform != "darwin":
        # The pip package bundles Qt libraries that clash with the system's on Linux.
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import cv2
    try:
        cv2.setLogLevel(OPENCV_LOG_SILENT)
    except Exception:
        pass
    return cv2


@contextlib.contextmanager
def silenced_stderr():
    """Send the process's stderr to nowhere for the duration of the block.

    ffmpeg's C library prints decoder errors straight to file descriptor 2,
    bypassing Python and OpenCV log levels. The descriptor is shared by the whole
    process, so this must wrap a whole run — not be entered by parallel workers.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(STDERR_FD)
    os.dup2(devnull, STDERR_FD)
    try:
        yield
    finally:
        os.dup2(saved, STDERR_FD)
        os.close(saved)
        os.close(devnull)


def open_video(cv2, path, hardware):
    """(capture, decoded on the graphics card?). hardware: try the card's video decoder first
    (H.264, HEVC...); a file it cannot take is opened for the CPU."""
    if hardware:
        capture = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        if capture.isOpened() and capture.get(cv2.CAP_PROP_HW_ACCELERATION) > 0:
            return capture, True
        capture.release()
    return cv2.VideoCapture(path), False


def read_frame_at(path, timestamp):
    """The frame at `timestamp` seconds as an RGB PIL image, or None. Seeking decodes every
    frame from the key frame before the moment, so the card's decoder is used when it can:
    a 4K HEVC game recording took ~1 s a frame on the CPU, ~0.25 s on the card. A file without
    time stamps is read with PyAV (`_frame_by_time`)."""
    cv2 = load_opencv()
    frame = _frame_at(cv2, path, timestamp, hardware=True)
    if frame is None:
        frame = _frame_at(cv2, path, timestamp, hardware=False)   # the card opened it but could not decode
    return None if frame is None else Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _frame_at(cv2, path, timestamp, hardware):
    capture, on_card = open_video(cv2, path, hardware)
    if hardware and not on_card:
        capture.release()
        return None
    try:
        if capture.grab() and capture.get(cv2.CAP_PROP_PTS) == NO_TIME_STAMP:
            frame = _frame_by_time(path, timestamp)
            if frame is not None:
                return frame
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * MILLISECONDS_PER_SECOND)
        ok, frame = capture.read()
    finally:
        capture.release()
    return frame if ok else None


def _frame_by_time(path, timestamp):
    """The frame at `timestamp` (BGR) read with PyAV, or None (also without PyAV). In a file whose frames OpenCV
    sees without time stamps (many .wmv) its every seek decodes from the first frame — up to 4 s a frame; PyAV
    seeks to the key frame before the moment (0.01-0.7 s). The moment counts from the first frame, as in OpenCV,
    so it is the same frame the index was made from."""
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            first = next(container.decode(stream)).time
            container.seek(int((first + timestamp) / stream.time_base), stream=stream, backward=True)
            before = None
            for frame in container.decode(stream):
                moment = frame.time - first
                if moment >= timestamp:
                    if before is not None and timestamp - (before.time - first) < moment - timestamp:
                        frame = before
                    return frame.to_ndarray(format="bgr24")
                before = frame
    except (av.FFmpegError, StopIteration, TypeError):   # TypeError: a frame without a time
        pass
    return None


def video_dimensions(path):
    """(width, height) of a video, or UNKNOWN for both if it cannot be opened."""
    cv2 = load_opencv()
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            return UNKNOWN, UNKNOWN
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or UNKNOWN
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or UNKNOWN
        return width, height
    finally:
        capture.release()


def sample_times(duration):
    """Moments (seconds) at which a video of the given length is sampled."""
    middle = [duration / 2.0]
    if duration < settings.VIDEO_FRAME_INTERVAL:
        return middle
    interval = max(settings.VIDEO_FRAME_INTERVAL, duration / settings.MAX_FRAMES_PER_VIDEO)
    times = []
    moment = interval   # skip t=0 — almost always a black intro or title card
    while moment < duration:
        times.append(round(moment, TIMESTAMP_DECIMALS))
        moment += interval
    return times or middle


@dataclass
class VideoSample:
    """What was read from one video."""
    frames: list = field(default_factory=list)   # [(pil_image, seconds), ...]
    too_long: bool = False   # longer than the limit, left out on purpose
    width: int = UNKNOWN
    height: int = UNKNOWN


class VideoFrameSampler:
    """Extracts the sample frames of one video; safe to run in worker threads.

    decode_mode GPU asks ffmpeg for the graphics card's video decoder (D3D11);
    HYBRID does that for every other video. A video the
    hardware decoder cannot open is decoded on the CPU.
    """

    def __init__(self, should_stop, decode_mode=DecodeMode.CPU, max_seconds=0):
        self._cv2 = load_opencv()
        self._should_stop = should_stop
        self._mode = decode_mode
        self._max_seconds = max_seconds   # longer videos are skipped; 0 = no limit
        self._opened = itertools.count()   # next() is atomic: safe from worker threads
        self.hardware_decoded = 0

    def sample(self, path, is_indexed=lambda timestamp: False):
        """Returns a VideoSample; is_indexed(seconds) tells which moments to skip."""
        name = os.path.basename(path)
        safe_print(f"[VINDEX] Analyzing: {name}")
        capture = None
        try:
            capture = self._open(path)
            if not capture.isOpened():
                safe_print(f"[VINDEX] Cannot open: {path}")
                return VideoSample()
            cv2 = self._cv2
            sample = VideoSample(width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or UNKNOWN,
                                 height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or UNKNOWN)
            duration = self._duration(capture, name)
            if duration <= 0:
                return sample
            limit = self._max_seconds
            if limit and duration > limit:
                safe_print(f"[VINDEX] Skipping {name}: {duration / SECONDS_PER_MINUTE:.1f} min "
                           f"is over the {limit}s limit")
                sample.too_long = True
                return sample
            sample.frames = self._read_frames(capture, sample_times(duration), is_indexed, name)
            return sample
        except Exception as error:   # MemoryError too
            safe_print(f"[VINDEX] Error extracting frames from {name}: {error}")
            return VideoSample()
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass

    def _wants_hardware(self):
        if self._mode is DecodeMode.GPU:
            return True
        return self._mode is DecodeMode.HYBRID and next(self._opened) % 2 == 0

    def _open(self, path):
        capture, on_card = open_video(self._cv2, path, self._wants_hardware())
        self.hardware_decoded += on_card
        return capture

    def _duration(self, capture, name):
        """Length in seconds. Frame rate and count are unreliable for some containers."""
        cv2 = self._cv2
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0:
            fps = ASSUMED_FPS
            safe_print(f"[VINDEX] fps=0 for {name}, assuming {ASSUMED_FPS:.0f}fps")
        if frame_count <= 0:
            safe_print(f"[VINDEX] frame_count=0 for {name}, assuming {ASSUMED_DURATION_SECONDS:.0f}s duration")
            return ASSUMED_DURATION_SECONDS
        return frame_count / fps

    def _read_frames(self, capture, times, is_indexed, name):
        cv2 = self._cv2
        frames = []
        brightest, brightest_level = None, -1.0
        deadline = time.monotonic() + settings.VIDEO_TIME_BUDGET_SECONDS
        seekable = True
        for moment in times:
            if self._should_stop():
                break
            if brightest is not None and time.monotonic() >= deadline:   # one frame is always read
                safe_print(f"[VINDEX] {name} is slow to seek; keeping {len(frames)} frame(s) read so far")
                break
            if is_indexed(moment):
                continue
            self._go_to(capture, moment, seekable)
            ok, frame = capture.read()
            seekable = capture.get(cv2.CAP_PROP_PTS) != NO_TIME_STAMP
            if not ok or frame is None:
                continue
            try:
                rgb = self._shrink(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except Exception:
                continue
            level = float(rgb[::BRIGHTNESS_SAMPLE_STEP, ::BRIGHTNESS_SAMPLE_STEP].mean())
            if level > brightest_level:
                brightest, brightest_level = (rgb, moment), level
            if level >= MIN_FRAME_BRIGHTNESS:
                frames.append((Image.fromarray(rgb), moment))
        if not frames and brightest is not None:
            rgb, moment = brightest
            safe_print(f"[VINDEX] All frames black for {name}, using least-dark frame at t={moment:.1f}s")
            return [(Image.fromarray(rgb), moment)]
        return frames

    def _go_to(self, capture, moment, seekable):
        """Make the next read give the frame at `moment`. OpenCV seeks exactly (it decodes forward from the key
        frame), but in a file whose frames carry no time stamps (many .wmv) every seek decodes from the very first
        frame: there the frames up to the moment are read forward instead — the same frame, a 194 s video's 38
        moments in 6 s instead of 113 s."""
        cv2 = self._cv2
        fps = capture.get(cv2.CAP_PROP_FPS)
        target = int(moment * fps + HALF_FRAME)
        if seekable or fps <= 0 or target < capture.get(cv2.CAP_PROP_POS_FRAMES):
            capture.set(cv2.CAP_PROP_POS_MSEC, moment * MILLISECONDS_PER_SECOND)
            return
        while capture.get(cv2.CAP_PROP_POS_FRAMES) < target and not self._should_stop() and capture.grab():
            pass

    def _shrink(self, rgb):
        """Shrink a decoded frame so queued frames stay small in RAM (a 4K frame is
        24 MB; the model only looks at a few hundred pixels anyway)."""
        height, width = rgb.shape[:2]
        longest = max(height, width)
        limit = settings.VIDEO_FRAME_MAX_SIDE
        if longest <= limit:
            return rgb
        scale = limit / longest
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return self._cv2.resize(rgb, size, interpolation=self._cv2.INTER_AREA)
