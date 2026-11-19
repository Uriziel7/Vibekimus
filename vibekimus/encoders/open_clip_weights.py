"""Where the weights of an open_clip model are read from: its compact file (see compact_weights)
when there is one; otherwise the download, which is rewritten as the compact file once and then
removed (same results, 2.9 instead of 4.5 GB for SigLIP 2)."""
import os
import shutil

from vibekimus import settings
from vibekimus.encoders.compact_weights import CompactWeights, bytes_needed, compact_file_for
from vibekimus.encoders.hf_cache import cached_file, remove_cached_file
from vibekimus.runlog import log_event, safe_print
from vibekimus.units import BYTES_PER_DECIMAL_GB

DISK_SPACE_MARGIN = 1.1   # free space wanted for the compact file, relative to its size


class OpenClipWeights:
    """The weight files of one catalogue model (ModelSpec)."""

    def __init__(self, spec):
        self._spec = spec
        self._compact = CompactWeights(compact_file_for(spec))
        self.path = None   # where the full weights can be read again; None = nowhere known

    def compact_file(self):
        """The compact file to build the network from, or None when there is none yet."""
        return self._compact.path if self._compact.exists() else None

    def use_compact(self, amp_dtype):
        """The network was built from the compact file; amp_dtype: the type this device computes in
        (None = float32). Raises when the file cannot be read."""
        rounded_to = self._compact.rounded_to()
        self.path = self._compact.path
        if amp_dtype is None or rounded_to != str(amp_dtype).replace("torch.", ""):
            safe_print(f"[MODEL] Compact weights were made for {rounded_to} and this device computes in "
                       f"{amp_dtype or 'float32'}: results differ very slightly from the original weights")
        safe_print("[MODEL] Weights: compact file")

    def use_download(self, net, rounding):
        """The network was built from the downloaded weights. rounding: the 16-bit type the graphics card
        computes in, None where it does not (there the compact file would change the results)."""
        self.path = self._downloaded_file()
        if rounding is not None and self.path is not None:
            self._make_compact(net, rounding)

    def _downloaded_file(self):
        """The downloaded weights file in the Hugging Face cache, or None."""
        import open_clip
        from open_clip.pretrained import HF_SAFE_WEIGHTS_NAME, HF_WEIGHTS_NAME
        repo = open_clip.get_pretrained_cfg(self._spec.name, self._spec.pretrained).get("hf_hub", "").strip("/")
        return cached_file(repo, (HF_SAFE_WEIGHTS_NAME, HF_WEIGHTS_NAME)) if repo else None

    def _make_compact(self, net, rounding):
        """Once, after the download: write the compact file, then remove the download."""
        compact, spec = self._compact, self._spec
        os.makedirs(settings.MODELS_DIR, exist_ok=True)
        if shutil.disk_usage(settings.MODELS_DIR).free < bytes_needed(net) * DISK_SPACE_MARGIN:
            safe_print("[MODEL] Not enough free disk space for the compact weights; keeping the downloaded ones")
            return
        safe_print("[MODEL] Writing compact weights (once)...")
        try:
            compact.write(net, rounding, source=f"{spec.name}/{spec.pretrained}")
            freed = remove_cached_file(self.path)
        except Exception as error:
            safe_print(f"[MODEL] Compact weights not written ({error}); keeping the downloaded ones")
            log_event(f"COMPACT WEIGHTS FAILED: {spec.name}: {error}")
            return
        self.path = compact.path
        message = (f"Compact weights written ({os.path.getsize(compact.path) / BYTES_PER_DECIMAL_GB:.2f} GB), "
                   f"download removed ({freed / BYTES_PER_DECIMAL_GB:.2f} GB)")
        safe_print(f"[MODEL] {message}")
        log_event(f"COMPACT WEIGHTS {spec.name}: {message}")
