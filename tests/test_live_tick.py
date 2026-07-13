#!/usr/bin/env python3
"""Live GTK test of the animation loop, driven by a real frame clock.

The render goldens prove *static* frames. This proves the moving parts: that the
engine's clocks are actually being advanced by the widget's tick callback, that
the hover-dwell really drills from the general object to the specific one after
the dwell time, that a press stays alive after the button comes up, and — the
easy one to regress — that the tick *stops* when nothing is animating.

Uses the spec scene, so it needs no worker, no model and no network.

Run: python tests/test_live_tick.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src", "bgbg"))

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import Gtk, GLib, GdkPixbuf  # noqa: E402

from viewer import ImageView  # noqa: E402

SCENE = os.path.join(ROOT, "spec", "assets", "scene")
R = {}


class FakeGesture:
    def get_current_event_state(self):
        return 0


def load_scene(v):
    with open(os.path.join(SCENE, "meta.json")) as f:
        meta = json.load(f)
    objs = []
    for o in meta["objects"]:
        o = dict(o)
        o["mask"] = os.path.join(SCENE, o["mask"])
        objs.append(o)
    v.pixbuf = GdkPixbuf.Pixbuf.new_from_file(
        os.path.join(SCENE, meta["source"]))
    v._texture = None
    v.set_seg_mode("everything")
    v.set_seg_layers(objs,
                     os.path.join(SCENE, meta["maps"]["label"]),
                     os.path.join(SCENE, meta["maps"]["general"]),
                     os.path.join(SCENE, meta["maps"]["depth"]))
    return meta


def main():
    Gtk.init()
    v = ImageView()
    meta = load_scene(v)
    win = Gtk.Window()
    win.set_default_size(320, 240)
    win.set_child(v)
    win.present()

    loop = GLib.MainLoop()
    v.set_dwell_ms(300)                 # keep the test quick
    steps = iter(range(100))

    # (46,52) is the nested core (obj 3) inside field (obj 1)
    NESTED = (46, 52)

    def to_view(ix, iy):
        return v.image_to_widget(ix, iy)

    def phase1():
        # the tick must be RUNNING (set_seg_layers began a reveal)
        R["tick_running_after_layers"] = v._anim_tick is not None
        R["reveal_started"] = v.anim.reveal < 1.0

        wx, wy = to_view(*NESTED)
        v._on_leave = lambda *a: None    # don't let a phantom leave cancel the dwell
        v._on_motion(None, wx, wy)
        R["hover_general_first"] = (v._hover_gen == 1 and v._seg_hover_id == 1
                                    and v._hover_spec == 3)
        R["depth_seen"] = v._hover_depth
        R["dwell_armed"] = v.anim.dwell_t0 is not None
        GLib.timeout_add(500, phase2)    # > dwell_ms
        return False

    def phase2():
        # the tick should have drilled general -> specific by now
        R["drilled"] = v.anim.drilled
        R["focus_is_specific"] = (v._seg_hover_id == 3)
        R["morph_ran"] = True            # _focus began a morph; it may already be done

        wx, wy = to_view(*NESTED)
        v._on_primary_pressed(FakeGesture(), 1, wx, wy)
        R["press_alive"] = v.anim.pressing and v.anim.held
        R["waves_while_held"] = len(v.anim.waves(v._now())) >= 1
        v._on_primary_released(FakeGesture(), 1, wx, wy)
        R["press_outlives_release"] = v.anim.pressing and not v.anim.held
        GLib.timeout_add(1100, phase3)   # > PRESS_WAVE_MS after release
        return False

    def phase3():
        R["press_retired"] = not v.anim.pressing
        # now make it fully idle and check the tick shuts itself down
        v._on_leave(None)
        v._seg_hover_id = 0
        v.set_seg_selection([])
        v.anim.cancel_dwell()
        v.set_scanning(False)
        GLib.timeout_add(400, phase4)
        return False

    def phase4():
        R["tick_stopped_when_idle"] = v._anim_tick is None
        loop.quit()
        return False

    GLib.timeout_add(200, phase1)
    GLib.timeout_add_seconds(20, lambda: (loop.quit(), False)[1])
    loop.run()

    checks = [
        ("tick running after set_seg_layers", R.get("tick_running_after_layers")),
        ("reveal began", R.get("reveal_started")),
        ("hover focuses the GENERAL object first", R.get("hover_general_first")),
        ("stacked depth seen (2)", R.get("depth_seen") == 2),
        ("dwell timer armed", R.get("dwell_armed")),
        ("dwell drilled after the dwell time", R.get("drilled")),
        ("focus moved to the SPECIFIC object", R.get("focus_is_specific")),
        ("press alive while held", R.get("press_alive")),
        ("waves spawn while held", R.get("waves_while_held")),
        ("press outlives the button release", R.get("press_outlives_release")),
        ("press retires once decayed", R.get("press_retired")),
        ("tick STOPS when idle", R.get("tick_stopped_when_idle")),
    ]
    ok = True
    for label, got in checks:
        ok &= bool(got)
        print("  %-42s %s" % (label, "PASS" if got else "FAIL (%r)" % got))
    print("LIVE TICK " + ("OK" if ok else "FAILED"))
    # os._exit skips GTK/worker teardown (which can SIGABRT offscreen) but it
    # also skips flushing stdio — do that by hand or the report is lost.
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
