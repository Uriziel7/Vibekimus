"""Encoder for open_clip models (SigLIP 2 by default): separate image and text towers.

Weights: after the first download they are rewritten once as a compact file
(open_clip_weights). VRAM Save keeps them in 16 bits on the GPU; should that ever produce
an invalid vector, the encoder goes back to full weights for the rest of the session."""
import contextlib

import numpy as np

from vibekimus.encoders.gpu_jpeg import GpuJpegDecoder
from vibekimus.encoders.open_clip_weights import OpenClipWeights
from vibekimus.encoders.pixels import PixelPipeline
from vibekimus.encoders.torch_base import TorchEncoder
from vibekimus.runlog import log_event, safe_print

DEFAULT_IMAGE_SIZE = 384
PREPROCESS_KEYS = {"image_mean": "mean", "image_std": "std", "image_interpolation": "interpolation",
                   "image_resize_mode": "resize_mode"}   # open_clip argument -> key of the weights' config


class OpenClipEncoder(TorchEncoder):
    def __init__(self, spec, vram_save=False):
        self._weights = OpenClipWeights(spec)
        self._weights_16bit = False   # VRAM Save is on
        super().__init__(spec, vram_save)
        self._pixels = PixelPipeline(self._preprocess, self.device)
        self.image_size = self._read_image_size()
        safe_print(f"[MODEL] Input resolution: {self.image_size}px")
        self.log_start(f"{self.image_size}px")
        safe_print("[MODEL] Ready!\n")

    def _load(self):
        import open_clip
        net = self._from_compact_file()
        if net is None:
            net = self._create(self.spec.pretrained)
            self._weights.use_download(net, self.amp_dtype if self.on_cuda else None)
        self._tokenizer = open_clip.get_tokenizer(self.spec.name)
        self._net = self._ready_on_device(self._in_vram_save_precision(net))

    def _fall_back_to_cpu(self, error):
        super()._fall_back_to_cpu(error)
        self._weights_16bit = False   # float32 on the CPU

    def _create(self, pretrained, dtype=None):
        """The network with weights from a tag or a file; the preprocessing always that
        of the tag's weights (a file alone would get open_clip's defaults). dtype: the
        16-bit type of VRAM Save, None for float32.

        On a graphics card the network is built right there, in its final type: its random
        starting values (overwritten by the weights at once) take about 6 s to fill in on
        the CPU and a fraction of a second on the GPU, and VRAM Save never holds a float32
        copy. The weights end up exactly the same."""
        import open_clip
        config = open_clip.get_pretrained_cfg(self.spec.name, self.spec.pretrained)
        overrides = {argument: config[key] for argument, key in PREPROCESS_KEYS.items() if key in config}
        if self.on_cuda:
            try:
                return self._built_on(self.device, dtype, pretrained, overrides)
            except self._torch.cuda.OutOfMemoryError:
                self.empty_cache()
                safe_print("[MODEL] Not enough free graphics memory to build the model there; building it in RAM")
        return self._built_on(self._torch.device("cpu"), None, pretrained, overrides)

    def _built_on(self, device, dtype, pretrained, overrides):
        import open_clip
        with device, self._parameters_created_as(dtype):
            net, _, self._preprocess = open_clip.create_model_and_transforms(
                self.spec.name, pretrained=pretrained, device=device, **overrides)
        return net

    @contextlib.contextmanager
    def _parameters_created_as(self, dtype):
        """torch's default type, for the whole process: nothing else creates tensors while
        the model loads (searching and indexing wait for it)."""
        torch = self._torch
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype or previous)
        try:
            yield
        finally:
            torch.set_default_dtype(previous)

    def _from_compact_file(self):
        """The network built from the compact weights, or None when there are none (or they cannot be read)."""
        path = self._weights.compact_file()
        if path is None:
            return None
        try:
            net = self._create(path, self._vram_save_dtype())
            self._weights.use_compact(self.amp_dtype)
        except Exception as error:
            safe_print(f"[MODEL] Compact weights unreadable ({error}); using the downloaded weights")
            return None
        return net

    # ── VRAM Save ────────────────────────────────────────────────────────────
    def _vram_save_dtype(self):
        """The 16-bit type of the weights under VRAM Save; None when they stay float32."""
        return self.amp_dtype if self.vram_save and self.on_cuda else None

    def _in_vram_save_precision(self, net):
        """The network in VRAM Save's type (already so when it was built in it)."""
        if not self.vram_save:
            return net
        dtype = self._vram_save_dtype()
        if dtype is None:
            safe_print("[MODEL] VRAM Save needs a graphics card that computes in 16 bits (RTX 20 or newer)")
            return net
        self._weights_16bit = True
        safe_print(f"[MODEL] VRAM Save: weights in {dtype}")
        return net.to(dtype)

    def _leave_vram_save(self):
        """Full weights again, exactly as without VRAM Save."""
        from open_clip.factory import load_checkpoint
        safe_print("[MODEL] ⚠ VRAM Save gave an invalid vector: full precision for the rest of this session")
        log_event(f"VRAM SAVE LEFT: invalid vector from {self.spec.name}")
        self._weights_16bit = False
        self._net = self._net.float()
        if self._weights.path is not None:
            load_checkpoint(self._net, self._weights.path, strict=True)

    def _read_image_size(self):
        try:
            size = self._net.visual.image_size
            return int(size[0] if isinstance(size, (tuple, list)) else size)
        except Exception:
            return DEFAULT_IMAGE_SIZE

    # ── encoding ─────────────────────────────────────────────────────────────
    def prepare(self, image):
        # Without this every worker thread starts a full PyTorch thread pool and the
        # workers together oversubscribe the CPU.
        self._torch.set_num_threads(1)
        return self._pixels.from_image(image)

    def create_jpeg_decoder(self):
        if not self.on_cuda or not self._pixels.split:
            return None
        return GpuJpegDecoder(self.device, self._pixels.tensor_geometry)

    def encode_prepared(self, items):
        return self._encode_pixels(items)

    def encode_images(self, images):
        return self._encode_pixels([self._pixels.from_image(image) for image in images])

    def encode_text(self, texts):
        tokens = self._tokenizer(texts)
        if isinstance(tokens, dict):
            tokens = {name: value.to(self.device) for name, value in tokens.items()}
            return self._encode("Text encoding", lambda: self._net.encode_text(**tokens), autocast_on_mps=False)
        tokens = tokens.to(self.device)
        return self._encode("Text encoding", lambda: self._net.encode_text(tokens), autocast_on_mps=False)

    def _encode_pixels(self, tensors):
        # Pinned memory allows a non-blocking copy, so the GPU is not stalled by it.
        pixels = self._pixels.batch_on_device(tensors, pin_memory=self.on_cuda)
        return self._encode("Image encoding", lambda: self._net.encode_image(pixels))

    def _encode(self, what, compute, autocast_on_mps=True):
        """compute() reads self._net when called, so a repeat uses the full weights."""
        def under_autocast():
            with self._autocast(autocast_on_mps):
                return compute()
        rows = self.normalized_rows(what, under_autocast)
        if self._weights_16bit and not np.isfinite(rows).all():
            self._leave_vram_save()
            rows = self.normalized_rows(what, under_autocast)
        return rows

    def _autocast(self, allow_mps):
        if self.amp_dtype is not None:
            if self.on_cuda:
                return self._torch.autocast(device_type="cuda", dtype=self.amp_dtype)
            if self.on_mps and allow_mps:
                return self._torch.autocast(device_type="mps", dtype=self.amp_dtype)
        return contextlib.nullcontext()
