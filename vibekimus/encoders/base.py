"""The interface the rest of the program uses to turn images and text into embeddings."""
from abc import ABC, abstractmethod


class MediaEncoder(ABC):
    """Maps images and text into one shared vector space (unit-length rows)."""

    device_name = "CPU"
    image_size = None   # input resolution of the vision side, in pixels

    @property
    def decode_size_hint(self):
        """Smallest side an image may be decoded at without losing anything the
        encoder would see. None = decode at full size."""
        return self.image_size

    @property
    def search_device(self):
        """Torch device able to hold the index for searching, or None for CPU search."""
        return None

    # ── encoding ─────────────────────────────────────────────────────────────
    @abstractmethod
    def prepare(self, image):
        """CPU-side preprocessing of one PIL image. Safe to call from worker threads."""

    @abstractmethod
    def encode_prepared(self, items):
        """Encode a list of prepare() results. Returns an (N, D) float32 array."""

    def create_jpeg_decoder(self):
        """A decoder that turns JPEG file contents into prepare() results on the
        accelerator (see gpu_jpeg.GpuJpegDecoder), or None if not available."""
        return None

    @abstractmethod
    def encode_images(self, images):
        """Encode a list of PIL images. Returns an (N, D) float32 array."""

    @abstractmethod
    def encode_text(self, texts):
        """Encode a list of strings. Returns an (N, D) float32 array."""

    # ── resource hooks (optional) ────────────────────────────────────────────
    def activate(self):
        """Called before a run of encoding work; re-acquire anything released earlier."""

    def release_memory(self):
        """Called after an indexing run; give back memory that is no longer needed."""

    def unload(self):
        """Called before another model is loaded in its place; free everything."""

    def empty_cache(self, synchronize=False):
        """Return cached accelerator memory to the system."""

    def reset_peak_memory(self):
        """Start measuring peak accelerator memory from now."""

    def peak_memory_gb(self):
        """Peak accelerator memory since reset_peak_memory(), or None if unknown."""
        return None
