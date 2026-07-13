"""The pixels the engine refers to only by handle.

The engine never holds a texture: it names images (`"src"`, an object id, a
`("clip", i)`) and this store turns a name back into something GSK can draw. The
browser backend will do exactly the same with ImageBitmaps, which is why the
engine can stay ignorant of both.

Decoding lives here too — PNG -> Gdk.Texture for a mask, PNG -> raw buffer for a
lookup map. The engine indexes the bytes but never parses them.
"""
import gi
gi.require_version("Gdk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gdk, GdkPixbuf, GLib  # noqa: E402

from engine.hittest import PixelMap  # noqa: E402

SRC = "src"          # the source image
POINT = "point"      # the point-mode object mask
CLIP = "clip"        # ("clip", i) — the i-th mask of the cutout preview


def load_texture(path):
    """A mask PNG -> Gdk.Texture, or None if it will not decode."""
    try:
        return Gdk.Texture.new_from_filename(path)
    except GLib.Error:
        return None


def load_pixel_map(path):
    """A lookup-map PNG -> a raw buffer the engine can index, or None.

    Decoding is the shell's job (GdkPixbuf here, an ImageBitmap on the web); the
    engine only ever indexes the bytes.
    """
    if not path:
        return None
    try:
        pb = GdkPixbuf.Pixbuf.new_from_file(path)
    except GLib.Error:
        return None
    return PixelMap(pb.get_pixels(), pb.get_rowstride(), pb.get_n_channels(),
                    pb.get_width(), pb.get_height())


class TextureStore:
    def __init__(self):
        self.source = None        # Gdk.Texture of the image being shown
        self.masks = {}           # object id -> Gdk.Texture
        self.point = None         # point-mode mask
        self.clip = []            # the cutout preview's masks, in order

    def resolve(self, handle):
        """Display-list image handle -> Gdk.Texture (or None)."""
        if handle == SRC:
            return self.source
        if handle == POINT:
            return self.point
        if isinstance(handle, tuple) and handle[0] == CLIP:
            i = handle[1]
            return self.clip[i] if i < len(self.clip) else None
        return self.masks.get(handle)

    def clear_seg(self):
        self.masks = {}
        self.point = None
