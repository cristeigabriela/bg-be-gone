#!/usr/bin/env python3
"""The engine must import with NOTHING but the standard library.

This is a load-bearing constraint, not a style rule:

  * the GTK shell runs on the *system* Python and deliberately has no numpy/PIL
    (it spawns a venv for the heavy work), so an engine that imported numpy would
    simply fail to load in the real app;
  * the same seam is what the browser needs — anything that reaches for numpy,
    PIL, gi or onnxruntime cannot be mirrored into the TypeScript core.

So we import the engine with those modules forcibly unavailable and assert it
still works. Run: python tests/test_engine_stdlib_only.py
"""
import os
import sys
import importlib

BANNED = ("gi", "numpy", "PIL", "onnxruntime", "cv2", "scipy", "rembg")


class _Blocker:
    """A meta-path finder that makes the banned modules look uninstalled."""

    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in BANNED:
            raise ImportError(
                "%r is not importable from bgbg.engine (stdlib only)" % root)
        return None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "src", "bgbg"))

    # drop anything already imported so the engine re-imports under the blocker
    for mod in list(sys.modules):
        if mod.split(".")[0] in BANNED or mod.split(".")[0] == "engine":
            del sys.modules[mod]

    sys.meta_path.insert(0, _Blocker())
    try:
        engine = importlib.import_module("engine")
        # exercise the public surface, not just the import
        p = engine.Pane()
        p.set_image_size(100, 80)
        p.set_view_size(400, 300)
        assert p.view_to_image(200, 150) is not None
        assert engine.polygon_area_abs([(0, 0), (2, 0), (2, 2), (0, 2)]) == 4.0
        assert abs(engine.ease_out(1.0) - 1.0) < 1e-12
        assert len(engine.resample_closed([(0, 0), (4, 0), (4, 4), (0, 4)], 16)) == 16
        m = engine.PixelMap(bytes([1, 1, 0]), 3, 3, 1, 1)   # id = 1 + 1*256
        assert engine.HitMaps(label=m).specific_at(0, 0) == 257
        assert engine.PROTOCOL_VERSION >= 1
    finally:
        sys.meta_path.pop(0)

    # and prove the blocker actually blocks (otherwise this test proves nothing)
    sys.meta_path.insert(0, _Blocker())
    try:
        try:
            importlib.import_module("numpy")
        except ImportError:
            pass
        else:
            raise AssertionError("blocker is inert — the test is vacuous")
    finally:
        sys.meta_path.pop(0)

    loaded = sorted(m for m in sys.modules
                    if m.startswith("engine") and "." in m)
    print("engine imported with %s unavailable" % ", ".join(BANNED))
    print("  submodules: %s" % ", ".join(loaded))
    print("STDLIB-ONLY OK")


if __name__ == "__main__":
    main()
