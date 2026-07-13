"""Hit-testing against the per-pixel lookup maps. Stdlib only.

The compute side emits three maps alongside the per-object masks:

  ``label``    the most *specific* object at each pixel (smallest painted last)
  ``general``  the most *general* object at each pixel (largest painted last)
  ``depth``    how many objects overlap at each pixel

Object ids are packed into an RGB image as ``R + G*256`` (0 = background), so up
to 65535 objects. ``depth`` is a single-channel count.

The engine does not decode PNGs — that is the shell's job (GdkPixbuf on the
desktop, an ImageBitmap/ImageData on the web). The shell hands over a raw pixel
buffer and its geometry; the engine just indexes it. That keeps this file free of
any image library and makes it trivially mirrorable in TypeScript over a
``Uint8Array``.
"""


class PixelMap:
    """A raw, shell-provided pixel buffer the engine can index directly.

    `data` is any bytes-like supporting integer indexing (bytes, memoryview,
    Uint8Array in the TS mirror). `channels` is the interleave, `stride` the row
    length in bytes.

    Each map carries its OWN width/height. The pre-extraction code bounds-checked
    the general and depth maps against the *label* map's dimensions, which is
    only correct because all three happen to be emitted at full image size.
    """

    __slots__ = ("data", "stride", "channels", "width", "height")

    def __init__(self, data, stride, channels, width, height):
        self.data = data
        self.stride = int(stride)
        self.channels = int(channels)
        self.width = int(width)
        self.height = int(height)

    def contains(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def _offset(self, x, y):
        return y * self.stride + x * self.channels

    def value_at(self, x, y):
        """First channel (R, or the grey level of an L image loaded as RGB)."""
        if not self.contains(x, y):
            return 0
        return self.data[self._offset(x, y)]

    def id_at(self, x, y):
        """Unpack an object id: ``R + G*256``. 0 means background."""
        if not self.contains(x, y):
            return 0
        i = self._offset(x, y)
        if self.channels >= 2:
            return self.data[i] + (self.data[i + 1] << 8)
        return self.data[i]


class Hit:
    """What is under a point, in one shot."""

    __slots__ = ("specific", "general", "depth")

    def __init__(self, specific=0, general=0, depth=0):
        self.specific = specific
        self.general = general
        self.depth = depth

    @property
    def stacked(self):
        """True when several objects overlap here (whole vs part is meaningful)."""
        return self.depth >= 2 and self.general != self.specific

    def __repr__(self):
        return "Hit(specific=%d, general=%d, depth=%d)" % (
            self.specific, self.general, self.depth)


class HitMaps:
    """The three lookup maps for one segmented image."""

    __slots__ = ("label", "general", "depth")

    def __init__(self, label=None, general=None, depth=None):
        self.label = label          # PixelMap | None
        self.general = general      # PixelMap | None
        self.depth = depth          # PixelMap | None

    def clear(self):
        self.label = self.general = self.depth = None

    @property
    def loaded(self):
        return self.label is not None

    def specific_at(self, x, y):
        return self.label.id_at(x, y) if self.label is not None else 0

    def general_at(self, x, y):
        return self.general.id_at(x, y) if self.general is not None else 0

    def depth_at(self, x, y):
        return self.depth.value_at(x, y) if self.depth is not None else 0

    def hit(self, x, y):
        return Hit(self.specific_at(x, y),
                   self.general_at(x, y),
                   self.depth_at(x, y))
