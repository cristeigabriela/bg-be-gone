#!/usr/bin/env python3
"""The app <-> engine seam, in the real window: hover, badge, click, sidebar.

The interaction gate (test_engine_interaction.py) proves the engine turns events
into the right effects. This proves the other half — that the GTK shell actually
*applies* them: that a real pointer motion over a real Adw window reaches
`Window._on_seg_hover` and writes the stacked-objects hint, that a real click
reaches `Window._on_seg_click` and updates the selection sidebar, and that the
selection goes on to reach the compute layer.

That chain (GTK controller -> engine event -> effect -> app callback -> worker
request) is exactly what step 6 rewired, and none of it is exercised by the
goldens.

**No ML, and no worker subprocess.** The worker is stubbed, so this runs in a
harness that cannot spawn one alongside GTK; the objects are injected exactly as
a real `seg_objects` reply would deliver them. The real worker pipeline is
covered by tests/test_live_segment.py, which needs the venv and a cached model.

Run: python tests/test_live_app_segment.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "src", "bgbg")
sys.path.insert(0, SRC)
os.environ["BGBG_START_PAGE"] = "segment"

import gi  # noqa: E402
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import GLib, Gio  # noqa: E402

import app as A  # noqa: E402

SCENE = os.path.join(ROOT, "spec", "assets", "scene")
NESTED = (46, 52)          # image px: object 3 (the core) nested inside object 1
R = {}


class StubWorker:
    """The compute port, without the subprocess. Records what the app asks for."""

    def __init__(self, on_message):
        self.ok = True
        self.error = None
        self.sent = []

    def send(self, req):
        self.sent.append(req)
        return 1

    def shutdown(self):
        pass


A.Worker = StubWorker

wins = []
_orig = A.Window.__init__


def _patched(self, app):
    _orig(self, app)
    wins.append(self)


A.Window.__init__ = _patched


def load_scene():
    with open(os.path.join(SCENE, "meta.json")) as f:
        meta = json.load(f)
    objs = [dict(o, mask=os.path.join(SCENE, o["mask"])) for o in meta["objects"]]
    maps = {k: os.path.join(SCENE, v) for k, v in meta["maps"].items()}
    return meta, objs, maps


class FakeGesture:
    def get_current_event_state(self):
        return 0


def main():
    app = A.App()
    app.set_flags(app.get_flags() | Gio.ApplicationFlags.NON_UNIQUE)
    loop = GLib.MainLoop()
    app.quit = lambda *a: loop.quit()
    app.register(None)
    app.activate()
    w = wins[0]
    meta, objs, maps = load_scene()
    v = w.seg_panel.view

    def phase1():
        # Put the scene on the Segment page and deliver the objects exactly as a
        # real worker reply would — this is app.py's own code path.
        v.load_file(os.path.join(SCENE, meta["source"]))
        w._on_worker_message({
            "type": "seg_objects", "objects": objs, "count": len(objs),
            "label_map": maps["label"], "general_map": maps["general"],
            "depth_map": maps["depth"],
        })
        R["objects_loaded"] = len(w.seg_objects) == len(objs)
        R["textures_loaded"] = len(v.textures.masks) == len(objs)
        R["engine_loaded"] = len(v.session.objects.masks) == len(objs)
        GLib.timeout_add(150, phase2)
        return False

    def phase2():
        # A real pointer motion over the stacked point, through the same handler
        # GTK's motion controller calls.
        wx, wy = v.image_to_widget(*NESTED)
        v._on_leave = lambda *a: None      # no phantom leave mid-dwell
        v._on_motion(None, wx, wy)
        R["hover_reached_app"] = "2 objects" in w.seg_status.get_text()
        R["status_after_hover"] = w.seg_status.get_text()

        # ... and a real click: press + release at the same point.
        v._on_primary_pressed(FakeGesture(), 1, wx, wy)
        v._on_primary_released(FakeGesture(), 1, wx, wy)
        R["selection"] = v.get_seg_selection()
        R["click_reached_app"] = "1 selected" in w.seg_status.get_text()
        R["status_after_click"] = w.seg_status.get_text()
        GLib.timeout_add(400, phase3)     # > the 150ms preview debounce
        return False

    def phase3():
        # The selection must reach the compute layer — a prerender for exactly
        # the object we clicked.
        req = [r for r in w.worker.sent if r.get("op") == "seg_extract"]
        R["prerender_sent"] = bool(req)
        R["prerender_ids"] = req[-1].get("ids") if req else None
        loop.quit()
        return False

    GLib.timeout_add(300, phase1)
    GLib.timeout_add_seconds(30, lambda: (loop.quit(), False)[1])
    loop.run()

    sel = R.get("selection") or []
    checks = [
        ("objects reach the app", R.get("objects_loaded")),
        ("mask textures load in the shell", R.get("textures_loaded")),
        ("the engine sees every object", R.get("engine_loaded")),
        ("hover -> app writes the stacked hint", R.get("hover_reached_app")),
        ("click -> exactly one object selected", len(sel) == 1),
        ("click -> app updates the sidebar", R.get("click_reached_app")),
        ("selection reaches the compute layer", R.get("prerender_sent")),
        ("... for the object that was clicked",
         R.get("prerender_ids") == sel),
    ]
    ok = True
    for label, got in checks:
        ok &= bool(got)
        print("  %-44s %s" % (label, "PASS" if got else "FAIL (%r)" % got))
    print("  hover:  %r" % (R.get("status_after_hover"),))
    print("  click:  %r" % (R.get("status_after_click"),))
    print("LIVE APP SEGMENT " + ("OK" if ok else "FAILED"))
    # os._exit skips GTK teardown (which can SIGABRT offscreen) but it also skips
    # flushing stdio — do that by hand or the report is lost.
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
