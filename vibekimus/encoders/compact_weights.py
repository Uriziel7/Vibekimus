"""A model's weights saved once in a smaller file, without changing what it computes.

Under mixed precision (autocast) the GPU rounds the weights and biases of linear,
convolution and attention projections to 16 bits right before using them; everything
else (normalisation, embeddings such as the word table, position embeddings, scales)
is used in float32. The compact file stores the first group already rounded to the
GPU's 16-bit type and the rest in float32. Loaded into a float32 model, the network
computes bit for bit what it computed from the original file, on a GPU of the same
precision. About 2.9 GB instead of 4.5 GB for SigLIP 2 SO400M."""
import os

from vibekimus import settings

COMPACT_SUFFIX = "_compact.safetensors"
FORMAT_KEY = "vibekimus_compact"
FORMAT_VERSION = "1"
ROUNDED_TO_KEY = "rounded_to"
SOURCE_KEY = "source"
BYTES_PER_16BIT_VALUE = 2
TEMPORARY_SUFFIX = ".tmp"
# Parameters that MultiheadAttention passes to linear() itself (autocast rounds them).
ATTENTION_PROJECTIONS = ("in_proj_weight", "in_proj_bias", "q_proj_weight", "k_proj_weight", "v_proj_weight")


def compact_file_for(spec):
    """Where the compact weights of a catalogue model (ModelSpec) are kept."""
    return os.path.join(settings.MODELS_DIR, spec.index_label + COMPACT_SUFFIX)


class CompactWeightsError(Exception):
    """The compact file could not be written, or does not match what was written."""


def rounded_names(net):
    """Names (as in net.state_dict()) of the tensors autocast rounds before use."""
    import torch.nn as nn
    names = set()
    for module_name, module in net.named_modules():
        prefix = f"{module_name}." if module_name else ""
        if isinstance(module, (nn.Linear, nn.modules.conv._ConvNd)):
            names.update(f"{prefix}{name}" for name, value in module.named_parameters(recurse=False)
                         if value is not None)
        elif isinstance(module, nn.MultiheadAttention):
            names.update(f"{prefix}{name}" for name in ATTENTION_PROJECTIONS if getattr(module, name, None) is not None)
    return names


class CompactWeights:
    """The compact weights file of one model."""

    def __init__(self, path):
        self.path = path

    def exists(self):
        return os.path.isfile(self.path)

    def rounded_to(self):
        """The 16-bit type the file was rounded to ("bfloat16" / "float16"), or None."""
        from safetensors import safe_open
        with safe_open(self.path, framework="pt") as file:
            return (file.metadata() or {}).get(ROUNDED_TO_KEY)

    def write(self, net, dtype, source):
        """Save net (a float32 model on any device) rounded for `dtype` (torch.bfloat16
        or torch.float16). The file is read back and compared before it replaces anything."""
        import torch
        from safetensors.torch import save_file
        rounded = rounded_names(net)
        tensors = {}
        for name, value in net.state_dict().items():
            value = value.detach().cpu()
            if name in rounded:
                value = value.to(dtype)
            elif value.is_floating_point():
                value = value.float()
            tensors[name] = value.contiguous()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        temporary = self.path + TEMPORARY_SUFFIX
        metadata = {FORMAT_KEY: FORMAT_VERSION, ROUNDED_TO_KEY: str(dtype).replace("torch.", ""), SOURCE_KEY: source}
        try:
            save_file(tensors, temporary, metadata=metadata)
            self._check(temporary, tensors)
            os.replace(temporary, self.path)
        except Exception:
            if os.path.exists(temporary):
                os.remove(temporary)
            raise

    @staticmethod
    def _check(path, expected):
        import torch
        from safetensors import safe_open
        with safe_open(path, framework="pt") as file:
            if set(file.keys()) != set(expected):
                raise CompactWeightsError("the written file has other tensors than the model")
            for name in file.keys():
                if not torch.equal(file.get_tensor(name), expected[name]):
                    raise CompactWeightsError(f"tensor {name} reads back differently")


def bytes_needed(net):
    """Size of the compact file for net (an estimate good to a few kilobytes)."""
    rounded = rounded_names(net)
    return sum(value.numel() * (BYTES_PER_16BIT_VALUE if name in rounded else value.element_size())
               for name, value in net.state_dict().items())
