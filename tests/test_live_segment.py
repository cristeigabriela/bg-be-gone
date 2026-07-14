#!/usr/bin/env python3
"""End-to-end: real worker, real SAM model, real masks, real render.

The engine refactor changed the on-disk mask format (L -> alpha coverage), which
touches the worker (save_objects / load_union / composite_extract), the texture
upload, and the renderer. The golden corpus covers the render half with a
synthetic scene; this covers the whole pipeline with the actual worker.

Needs the venv + a cached SAM model. Run: python tests/test_live_segment.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "src", "bgbg")
sys.path.insert(0, SRC)
os.environ.setdefault("BGBG_VENV_PYTHON",
                      os.path.expanduser("~/.local/share/bg-be-gone/venv/bin/python"))
os.environ.setdefault("BGBG_WORKER", os.path.join(SRC, "compute", "service.py"))
os.environ["BGBG_START_PAGE"] = "segment"
os.chdir(SRC)

import gi  # noqa: E402
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import GLib, Gio  # noqa: E402

import app as A  # noqa: E402

IMG = os.path.join(ROOT, "spec", "assets", "scene", "source.png")
R = {}
wins = []
_orig = A.Window.__init__


def _patched(self, app):
    _orig(self, app)
    wins.append(self)


A.Window.__init__ = _patched

app = A.App()
app.set_flags(app.get_flags() | Gio.ApplicationFlags.NON_UNIQUE)
loop = GLib.MainLoop()
app.quit = lambda *a: loop.quit()
state = {"step": 0, "n": 0}


def tick():
    state["n"] += 1
    if state["n"] > 300:
        R["timeout"] = True
        loop.quit()
        return False
    if not wins:
        return True
    w = wins[0]

    if state["step"] == 0:
        if not getattr(w, "_seg_available", False):
            return True
        if not w.source_path:
            w._load_source(IMG)
            return True
        names = [w.seg_model_row.get_model().get_string(i)
                 for i in range(w.seg_model_row.get_model().get_n_items())]
        pick = [i for i, n in enumerate(names) if "Tiny" in n or "Mobile" in n]
        if pick:
            w.seg_model_row.set_selected(pick[0])
        w.seg_detail_row.set_selected(0)
        w._on_seg_everything()
        state["step"] = 1
        return True

    if state["step"] == 1:
        if w.seg_busy or not w.seg_objects:
            return True
        R["n_objects"] = len(w.seg_objects)
        # the worker must now be writing alpha-coverage masks
        from PIL import Image as _I  # only to inspect; the shell never needs PIL
        modes = {_I.open(o["mask"]).mode for o in w.seg_objects}
        R["mask_modes"] = sorted(modes)
        # textures loaded into the shell, objects into the engine
        R["textures_loaded"] = (
            len(w.seg_panel.view.textures.masks) == len(w.seg_objects)
            and len(w.seg_panel.view.session.objects.masks) == len(w.seg_objects))
        # select two objects -> exercises the mask UNION path end to end
        ids = [o["id"] for o in w.seg_objects][:2]
        R["selected"] = ids
        w.seg_panel.view.set_seg_selection(ids)
        w._update_seg_selection_ui()
        state["step"] = 2
        state["t"] = state["n"]
        return True

    if state["step"] == 2:
        if state["n"] - state["t"] < 12:      # let the debounced prerender land
            return True
        R["extract_ok"] = bool(w.seg_result_output
                               and os.path.exists(w.seg_result_output))
        R["save_enabled"] = w.seg_save_btn.get_sensitive()
        loop.quit()
        return False
    return True


app.register(None)
app.activate()
GLib.timeout_add(300, tick)
GLib.timeout_add_seconds(120, lambda: (loop.quit(), False)[1])
loop.run()

checks = [
    ("segmentation produced objects", R.get("n_objects", 0) > 0),
    ("worker writes alpha-coverage masks", R.get("mask_modes") == ["RGBA"]),
    ("all mask textures loaded", R.get("textures_loaded")),
    ("multi-select extract produced a file", R.get("extract_ok")),
    ("save enabled", R.get("save_enabled")),
    ("did not time out", not R.get("timeout")),
]
ok = True
for label, got in checks:
    ok &= bool(got)
    print("  %-38s %s" % (label, "PASS" if got else "FAIL (%r)" % got))
print("  (%d objects, masks=%s, selected=%s)"
      % (R.get("n_objects", 0), R.get("mask_modes"), R.get("selected")))
print("LIVE SEGMENT " + ("OK" if ok else "FAILED"))
sys.stdout.flush()
os._exit(0 if ok else 1)
