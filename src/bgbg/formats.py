"""The image formats bg-be-gone opens and saves, in one place.

The GTK frontend (system Python) and the worker (its own venv) both import
this module, so it uses only the standard library. The file dialog filters,
drag and drop, the batch folder scan and the save dialog all read from here.

The worker decodes every format in READ_FORMATS with Pillow, plus pi-heif for
HEIC/HEIF. The frontend previews with GdkPixbuf and asks the worker for a PNG
when no GdkPixbuf loader is installed for the format.
"""
import os
from dataclasses import dataclass
from enum import Enum


class Animation(Enum):
    """Whether an image has more than one frame, or a format can hold them."""

    STILL = "still"
    ANIMATED = "animated"


@dataclass(frozen=True)
class ImageFormat:
    name: str              # shown in dialogs and messages
    suffixes: tuple        # lower case, with the dot; the first is the one we write
    mime_types: tuple
    animation: Animation

    @property
    def suffix(self):
        return self.suffixes[0]


PNG = ImageFormat("PNG", (".png",), ("image/png",), Animation.STILL)
JPEG = ImageFormat("JPEG", (".jpg", ".jpeg", ".jpe"), ("image/jpeg",), Animation.STILL)
WEBP = ImageFormat("WebP", (".webp",), ("image/webp",), Animation.ANIMATED)
AVIF = ImageFormat("AVIF", (".avif",), ("image/avif",), Animation.STILL)
HEIF = ImageFormat("HEIF", (".heic", ".heif"), ("image/heic", "image/heif"), Animation.STILL)
TIFF = ImageFormat("TIFF", (".tif", ".tiff"), ("image/tiff",), Animation.STILL)
BMP = ImageFormat("BMP", (".bmp",), ("image/bmp",), Animation.STILL)
GIF = ImageFormat("GIF", (".gif",), ("image/gif",), Animation.ANIMATED)

READ_FORMATS = (PNG, JPEG, WEBP, AVIF, HEIF, TIFF, BMP, GIF)
# Batch cuts out still images only. A GIF there would lose all but its first
# frame, so GIFs stay on the Single page, which processes every frame.
BATCH_FORMATS = tuple(f for f in READ_FORMATS if f is not GIF)
READ_SUFFIXES = tuple(s for f in READ_FORMATS for s in f.suffixes)
BATCH_SUFFIXES = tuple(s for f in BATCH_FORMATS for s in f.suffixes)
MIME_TYPES = tuple(m for f in READ_FORMATS for m in f.mime_types)


def format_of(path):
    """The format a file name says it is, or None for one we don't read."""
    suffix = os.path.splitext(path)[1].lower()
    return next((f for f in READ_FORMATS if suffix in f.suffixes), None)
