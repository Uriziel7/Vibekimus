"""Where media files are decoded while indexing: on the CPU, the GPU, or both."""
from dataclasses import dataclass
from enum import Enum


class DecodeMode(Enum):
    CPU = "cpu"
    GPU = "gpu"          # images: JPEGs on the GPU, other formats still on the CPU
    HYBRID = "cpu+gpu"   # both at once, sharing the work

    @property
    def uses_gpu(self):
        return self is not DecodeMode.CPU

    @classmethod
    def parse(cls, value, default):
        try:
            return cls(value)
        except ValueError:
            return default


@dataclass(frozen=True)
class DecodingOptions:
    images: DecodeMode = DecodeMode.CPU
    videos: DecodeMode = DecodeMode.CPU
