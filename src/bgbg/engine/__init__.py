"""bg-be-gone engine — the headless core.

STDLIB ONLY. This package must import with no `gi`, no numpy, no PIL and no
onnxruntime, because the GTK shell runs on the *system* Python (it spawns a venv
for the heavy work). That constraint is what keeps the engine portable: it is the
same seam the browser needs, so a TypeScript mirror of this package can be written
against the shared schema + golden corpus in `spec/`.

Anything that needs pixels or a model belongs in `bgbg.compute`, behind a port.
"""
from .geometry import (          # noqa: F401
    MIN_ZOOM, MAX_ZOOM, clamp, lerp, ease_out, ease_out_back,
    polygon_area_abs, resample_closed, align_ring,
)
from .pane import Pane                            # noqa: F401
from .hittest import PixelMap, HitMaps, Hit       # noqa: F401

PROTOCOL_VERSION = 1
