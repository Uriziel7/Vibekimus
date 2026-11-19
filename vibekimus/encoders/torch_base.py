"""What every PyTorch encoder needs: device and precision choice, loading weights from
the local Hugging Face cache, normalised outputs and GPU memory housekeeping."""
import gc
import os
import traceback

from vibekimus import hardware, settings
from vibekimus.encoders.base import MediaEncoder
from vibekimus.runlog import log_event, safe_print
from vibekimus.units import BYTES_PER_GB

# CUDA compute capability (major, minor) per GPU generation.
CAPABILITY_BLACKWELL = 12        # RTX 50xx
CAPABILITY_ADA = (8, 9)          # RTX 40xx — first generation where BF16 is preferred
CAPABILITY_TURING = 7            # RTX 20xx / GTX 16xx — FP16 works
MIN_TORCH_FOR_BLACKWELL = (2, 7)
BANNER_WIDTH = 60


def set_hf_offline(offline):
    """Switch the Hugging Face libraries between offline and online mode.

    Changing the flag is not enough: huggingface_hub keeps one HTTP session per
    thread, and a session created while offline keeps refusing every request
    afterwards. The sessions must be discarded so the next request builds a new
    one that matches the flag.
    """
    import huggingface_hub
    import huggingface_hub.constants
    import huggingface_hub.utils
    flag = "1" if offline else "0"
    huggingface_hub.constants.HF_HUB_OFFLINE = offline
    os.environ["HF_HUB_OFFLINE"] = flag
    os.environ["TRANSFORMERS_OFFLINE"] = flag
    for name in ("reset_sessions", "close_session"):   # the name differs between versions
        reset = getattr(huggingface_hub.utils, name, None)
        if callable(reset):
            try:
                reset()
            except Exception:
                pass
            break


class TorchEncoder(MediaEncoder):
    """Base for encoders running on PyTorch. Subclasses load their network in
    _load() and implement the encoding methods of MediaEncoder."""

    def __init__(self, spec, vram_save=False):
        import torch
        self._torch = torch
        self.spec = spec
        self.vram_save = vram_save   # weights in 16 bits on the GPU, where the encoder supports it
        self.device, self.device_name = self._select_device()
        safe_print(f"[MODEL] Using Device: {self.device_name}")
        self._enable_tf32()
        self.amp_dtype = self._select_precision()
        self._warn_if_torch_too_old_for_gpu()
        safe_print(f"[MODEL] Loading: {spec.label} ({spec.name})")
        self._load_from_cache_or_hub(self._load)

    def _load(self):
        """Load the network onto self.device (called with the right offline mode)."""
        raise NotImplementedError

    # ── device and precision ─────────────────────────────────────────────────
    @property
    def on_cuda(self):
        return self.device.type == "cuda"

    @property
    def on_mps(self):
        return self.device.type == "mps"

    @property
    def search_device(self):
        return self.device if self.on_cuda else None

    def _select_device(self):
        torch = self._torch
        try:
            if torch.cuda.is_available():
                return torch.device("cuda"), f"CUDA (GPU {torch.cuda.get_device_name(0)})"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps"), "Metal (Apple GPU)"
            if os.name == "nt":
                try:
                    import torch_directml
                    return torch_directml.device(), "DirectML (Windows GPU)"
                except ImportError:
                    pass
        except Exception:
            pass
        return torch.device("cpu"), "CPU"

    def _enable_tf32(self):
        """Free matmul speed-up on Ampere and newer."""
        if self.on_cuda:
            self._torch.backends.cuda.matmul.allow_tf32 = True
            self._torch.backends.cudnn.allow_tf32 = True

    def _select_precision(self):
        """Reduced-precision dtype for this device, or None to run in float32."""
        torch = self._torch
        dtype = None
        try:
            if self.on_cuda:
                capability = torch.cuda.get_device_capability(0)
                if capability >= CAPABILITY_ADA:
                    dtype = torch.bfloat16
                elif capability[0] >= CAPABILITY_TURING:
                    dtype = torch.float16
            elif self.on_mps:
                dtype = torch.float16
        except Exception:
            return None
        if dtype is not None:
            safe_print(f"[MODEL] Mixed precision enabled: {dtype}")
        elif self.on_cuda:
            safe_print("[MODEL] Mixed precision disabled (GPU too old)")
        return dtype

    def _warn_if_torch_too_old_for_gpu(self):
        torch = self._torch
        try:
            if not self.on_cuda or torch.cuda.get_device_capability(0)[0] < CAPABILITY_BLACKWELL:
                return
            version = tuple(int(part) for part in torch.__version__.split(".")[:2])
        except Exception:
            return
        if version >= MIN_TORCH_FOR_BLACKWELL:
            safe_print(f"[MODEL] RTX 50-series detected — PyTorch {torch.__version__} has native support ✓")
            return
        banner = "=" * BANNER_WIDTH
        safe_print(f"\n{banner}\n"
                   f"[WARNING] RTX 50-series (Blackwell) detected!\n"
                   f"[WARNING] Your PyTorch version ({torch.__version__}) may not fully support this GPU.\n"
                   f"[WARNING] If you see CUDA errors, upgrade PyTorch:\n"
                   f"[WARNING] pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
                   f"{banner}\n")

    def _ready_on_device(self, net):
        """`net` on self.device, ready to encode; in float32 on the CPU when the graphics card cannot
        take it (driver reset, exclusive GPU lock, out of memory)."""
        try:
            return net.to(self.device).eval()
        except Exception as gpu_error:
            self._fall_back_to_cpu(gpu_error)
            return net.float().to(self.device).eval()

    def _fall_back_to_cpu(self, error):
        safe_print(f"[MODEL] ⚠ GPU transfer failed ({error}), falling back to CPU")
        safe_print("[MODEL] Tip: close other GPU-heavy apps (games, other ML tools) and restart")
        self.device = self._torch.device("cpu")
        self.device_name = "CPU (GPU fallback)"
        self.amp_dtype = None

    # ── loading ──────────────────────────────────────────────────────────────
    def _load_from_cache_or_hub(self, load):
        """Local cache first; the network only if settings.ALLOW_MODEL_DOWNLOAD."""
        try:
            set_hf_offline(True)
            load()
            safe_print("[MODEL] Loaded from local cache")
            return
        except Exception as error:
            safe_print(f"[MODEL] Not in the local cache ({type(error).__name__}: {error})")
        if not settings.ALLOW_MODEL_DOWNLOAD:
            raise RuntimeError(f"Model {self.spec.label} ({self.spec.name}) is not in the local cache "
                               f"and ALLOW_MODEL_DOWNLOAD is False.")
        try:
            set_hf_offline(False)
            safe_print(f"[MODEL] Downloading {self.spec.name} (this may take a while)...")
            load()
            safe_print("[MODEL] Download complete!")
        except Exception as error:
            safe_print(f"[MODEL] Download failed: {error}")
            raise

    def log_start(self, input_description):
        vram = hardware.system_vram_bytes()
        vram_text = f"{vram / BYTES_PER_GB:.0f} GB" if vram else "unknown"
        log_event(f"START model={self.spec.name} device={self.device_name} vram={vram_text} "
                  f"precision={self.amp_dtype} input={input_description} torch={self._torch.__version__} "
                  f"batch={hardware.image_batch_size()} video_batch={hardware.video_batch_size()} "
                  f"chunk={settings.STREAM_CHUNK} load_workers={settings.LOAD_WORKERS} "
                  f"video_workers={settings.VIDEO_WORKERS} cpu_threads={os.cpu_count()} "
                  f"storage={settings.STORAGE_MODE}")

    # ── results ──────────────────────────────────────────────────────────────
    def normalized_rows(self, what, compute):
        """Run compute() without gradients and return its rows as unit-length float32 numpy."""
        torch = self._torch
        try:
            with torch.no_grad():
                features = compute().float()
            return (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()
        except Exception as error:
            safe_print(f"[ERROR] {what} failed: {error}")
            traceback.print_exc()
            raise

    # ── resources ────────────────────────────────────────────────────────────
    def release_memory(self):
        """Give back the VRAM an indexing run accumulated. Emptying the allocator
        cache is enough; the model stays where it is."""
        try:
            gc.collect()
            self.empty_cache(synchronize=True)
        except Exception as error:
            safe_print(f"[VRAM] Cleanup warning: {error}")

    def unload(self):
        """Free the model's memory before another model is loaded."""
        for name in [name for name, value in vars(self).items() if hasattr(value, "parameters")]:
            setattr(self, name, None)
        self.release_memory()

    def empty_cache(self, synchronize=False):
        torch = self._torch
        if self.on_cuda:
            torch.cuda.empty_cache()
            if synchronize:
                torch.cuda.synchronize()
        elif self.on_mps:
            torch.mps.empty_cache()

    def reset_peak_memory(self):
        if self.on_cuda:
            try:
                self._torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def peak_memory_gb(self):
        if not self.on_cuda:
            return None
        try:
            return self._torch.cuda.max_memory_allocated() / BYTES_PER_GB
        except Exception:
            return None
