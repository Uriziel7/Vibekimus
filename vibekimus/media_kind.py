"""What a search result is: a picture, or one moment of a video."""
from enum import StrEnum


class MediaKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
