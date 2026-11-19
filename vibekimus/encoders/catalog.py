"""The models the program offers, and how to create the encoder for one of them.

To offer another model, add a ModelSpec to MODELS. A model of a new family also
needs an encoder class (a TorchEncoder subclass) registered in ENCODER_CLASSES.
Every model has its own index files, so switching models never mixes embeddings.
"""
import importlib
from dataclasses import dataclass

OPEN_CLIP = "open_clip"
# family -> (module, class). Imported only when needed: they pull in torch.
ENCODER_CLASSES = {
    OPEN_CLIP: ("vibekimus.encoders.open_clip_encoder", "OpenClipEncoder"),
}


@dataclass(frozen=True)
class ModelSpec:
    key: str            # stored in the preferences file
    label: str          # shown in the Settings window and the title bar
    description: str    # one line for the Settings window
    family: str         # which encoder class runs it
    name: str           # open_clip model name
    pretrained: str = ""   # open_clip weights tag
    # Cosine similarity from which two images count as the same picture (re-saved,
    # resized) and as very similar (burst shots, small edits). Every model has its
    # own scale; measured on re-saved copies vs. different images of a real library
    # (lower "very similar" values grouped hundreds of merely alike screenshots).
    same_picture: float = 0.97
    very_similar: float = 0.95

    @property
    def index_label(self):
        """The part of the index file names that identifies the model."""
        return f"{self.name}_{self.pretrained.upper()}"

    def made(self, meta):
        """Whether index metadata says the index was made with this model."""
        return meta.get("model") == self.name and meta.get("pretrained", "") == self.pretrained


SIGLIP2_SO400M = ModelSpec(
    key="siglip2-so400m",
    label="SigLIP 2 SO400M",
    description="Fast (about 235 images/s on an RTX 5090), multilingual. Text in pictures is read by OCR.",
    family=OPEN_CLIP,
    name="ViT-SO400M-16-SigLIP2-384",
    pretrained="webli",
)
MODELS = (SIGLIP2_SO400M,)


def model_by_key(key, default=SIGLIP2_SO400M):
    return next((spec for spec in MODELS if spec.key == key), default)


def model_for_meta(meta):
    """The catalogue entry an index was made with, or None."""
    return next((spec for spec in MODELS if spec.made(meta)), None)


def create_encoder(spec, vram_save=False):
    """Load the model (slow: imports torch and reads gigabytes of weights). vram_save:
    keep its weights in 16 bits on the graphics card."""
    module_name, class_name = ENCODER_CLASSES[spec.family]
    return getattr(importlib.import_module(module_name), class_name)(spec, vram_save)
