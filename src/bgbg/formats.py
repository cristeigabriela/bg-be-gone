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
from typing import Optional


class Animation(Enum):
    """Whether an image has more than one frame, or a format can hold them."""

    STILL = "still"
    ANIMATED = "animated"


class Alpha(Enum):
    """Whether a result has a transparent background to keep."""

    TRANSPARENT = "transparent"
    FLATTENED = "flattened"


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


def is_readable(path):
    return format_of(path) is not None


def can_animate(path):
    fmt = format_of(path)
    return fmt is not None and fmt.animation is Animation.ANIMATED


def save_formats(alpha):
    """What a still result can be saved as. JPEG has no alpha channel, so it is
    offered only once the background is flattened."""
    if alpha is Alpha.TRANSPARENT:
        return (PNG, WEBP)
    return (PNG, WEBP, JPEG)


def default_save_format(source_path, choices):
    """The source's own format when it is one of `choices`, else the first."""
    source = format_of(source_path or "")
    return source if source in choices else choices[0]


class SuffixChange(Enum):
    """What save_target did to the name typed in the save dialog."""

    KEPT = "kept"            # its suffix names one of the choices
    ADDED = "added"          # no image suffix: the default's was appended
    REPLACED = "replaced"    # an image format this result can't be saved as


@dataclass(frozen=True)
class SaveTarget:
    path: str
    format: ImageFormat
    change: SuffixChange
    typed: Optional[ImageFormat]     # the format the typed suffix named


def save_target(path, choices, default):
    """Where and how to save, from the name typed in the save dialog.

    A suffix that names one of `choices` picks that format. A suffix naming
    another format is replaced by the default's, so `cutout.jpg` for a
    transparent result becomes `cutout.png`, not a JPEG without the
    transparency. Any other name gets the default's suffix appended.
    """
    typed = format_of(path)
    if typed in choices:
        return SaveTarget(path, typed, SuffixChange.KEPT, typed)
    if typed is not None:
        stem = os.path.splitext(path)[0]
        return SaveTarget(stem + default.suffix, default, SuffixChange.REPLACED, typed)
    return SaveTarget(path + default.suffix, default, SuffixChange.ADDED, None)


def overwrites_unasked(target):
    """Whether saving would replace a file nobody agreed to replace.

    The dialog asks before overwriting the name that was typed. When
    save_target changed that name, the file it now names wasn't asked about.
    """
    return target.change is not SuffixChange.KEPT and os.path.exists(target.path)
