#!/usr/bin/env python3
"""Freeze/check the SIDEBAR as a golden — the regression gate for step 7.

Step 7 moves the settings out of app.py's hardcoded `SUBJECTS`/`MODELS`/`BGS`/
`SEG_*` tables and into the engine, and rebuilds the sidebar from the engine's
`describe()`. The gate is that the user sees *exactly* the same sidebar: the same
groups, the same rows, in the same order, with the same titles, subtitles,
options, defaults — and the same reactive behaviour (the exact-model row appears
only for "Custom model…", the blur row only for "Blur background", the colour row
is sensitive only for "Custom…").

So this walks the real widget tree of the real Adw window and dumps it, across
several pages *and* several settings states. Freeze it before the refactor, check
it after: a row that moved, lost a subtitle, or changed its option list shows up
as a diff on that row instead of as a vague "the sidebar looks off".

The worker is stubbed — no subprocess, no model. See tests/test_live_app_segment.py.

    python spec/tools/uidump.py --freeze
    python spec/tools/uidump.py --check
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.dirname(HERE)
ROOT = os.path.dirname(SPEC)
SRC = os.path.join(ROOT, "src", "bgbg")
sys.path.insert(0, SRC)

import gi  # noqa: E402
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Gio, Adw  # noqa: E402

import app as A  # noqa: E402
from engine.ports import RecordingPort  # noqa: E402

GOLDEN = os.path.join(SPEC, "goldens", "sidebar.json")

# What the worker would report for the SAM ladder — pinned here so the dynamic
# "Model" options under Segment > Advanced are part of the golden too.
SEG_MODELS = [
    {"rung": "tiny", "label": "SAM 2.1 Tiny", "vram": 1500},
    {"rung": "small", "label": "SAM 2.1 Small", "vram": 2600},
    {"rung": "mobile", "label": "MobileSAM", "vram": 900},
]


class StubPort(RecordingPort):
    def __init__(self, python, script, on_event, log=None):
        super().__init__(on_event)


A.WorkerPort = StubPort

_wins = []
_orig = A.Window.__init__


def _patched(self, app):
    _orig(self, app)
    _wins.append(self)


A.Window.__init__ = _patched


# ---------------------------------------------------------------- the walk ---
def _strings(row):
    """The option labels of an Adw.ComboRow, in order."""
    m = row.get_model()
    if m is None:
        return []
    return [m.get_string(i) for i in range(m.get_n_items())]


def _row(w):
    d = {"kind": type(w).__name__, "title": w.get_title()}
    sub = w.get_subtitle() if hasattr(w, "get_subtitle") else None
    if sub:
        d["subtitle"] = sub
    if not w.get_visible():
        d["visible"] = False
    if not w.get_sensitive():
        d["sensitive"] = False
    if isinstance(w, Adw.ComboRow):
        d["options"] = _strings(w)
        d["selected"] = w.get_selected()
    elif isinstance(w, Adw.SwitchRow):
        d["active"] = w.get_active()
    elif isinstance(w, Adw.SpinRow):
        a = w.get_adjustment()
        d["value"] = a.get_value()
        d["range"] = [a.get_lower(), a.get_upper()]
    return d


def _walk(w, out):
    """Depth-first over the real widget tree, recording only the rows a user
    sees. Everything else (boxes, labels, Adw's internal plumbing) is skipped but
    still descended into."""
    child = w.get_first_child()
    while child is not None:
        if isinstance(child, Adw.PreferencesGroup):
            grp = {"group": child.get_title(), "rows": []}
            if not child.get_visible():
                grp["visible"] = False
            out.append(grp)
            _walk(child, grp["rows"])
        elif isinstance(child, Adw.PreferencesRow):
            row = _row(child)
            out.append(row)
            if isinstance(child, Adw.ExpanderRow):
                row["rows"] = []
                _walk(child, row["rows"])
                child = child.get_next_sibling()
                continue
        else:
            _walk(child, out)
        child = child.get_next_sibling()
    return out


def dump(w):
    """The sidebar across the pages and the states whose rules step 7 moves into
    the schema."""
    states = []

    def snap(name):
        # let GTK settle the visibility changes we just triggered
        while Gtk.events_pending() if hasattr(Gtk, "events_pending") else False:
            Gtk.main_iteration()
        states.append({"state": name, "sidebar": _walk(w.sidebar_root, [])})

    w.stack.set_visible_child_name("single")
    w._sync_sidebar_page()
    snap("single/defaults")

    n = w.subject_row.get_model().get_n_items()
    w.subject_row.set_selected(n - 1)              # "Custom model…"
    snap("single/subject=custom-model")
    w.subject_row.set_selected(0)

    w.bg_row.set_selected(1)                       # "Blur background"
    snap("single/bg=blur")
    w.bg_row.set_selected(5)                       # "Custom…"
    snap("single/bg=custom")
    w.bg_row.set_selected(0)

    w.stack.set_visible_child_name("segment")
    w._sync_sidebar_page()
    snap("segment/seg-unavailable")               # the group stays hidden

    w._seg_available = True                       # the worker reported SAM
    w._sync_sidebar_page()
    snap("segment/available")

    w._seg_models = SEG_MODELS                    # ... and its model ladder
    w._apply_seg_models()
    snap("segment/models-loaded")

    w.stack.set_visible_child_name("batch")
    w._sync_sidebar_page()
    snap("batch/defaults")
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    app = A.App()
    app.set_flags(app.get_flags() | Gio.ApplicationFlags.NON_UNIQUE)
    app.register(None)
    app.activate()
    w = _wins[0]

    got = json.dumps(dump(w), indent=1, sort_keys=True)

    rc = 0
    if args.freeze:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            f.write(got + "\n")
        n = sum(1 for _ in json.loads(got))
        print("froze %s (%d states, %d B)" % (GOLDEN, n, len(got)))
    elif args.check:
        if not os.path.exists(GOLDEN):
            print("MISSING GOLDEN %s" % GOLDEN)
            rc = 1
        else:
            with open(GOLDEN) as f:
                want = f.read().rstrip("\n")
            if want == got:
                print("sidebar matches (%d states)" % len(json.loads(got)))
            else:
                with open(GOLDEN + ".actual", "w") as f:
                    f.write(got + "\n")
                print("SIDEBAR DIFF — wrote %s.actual" % GOLDEN)
                _diff(json.loads(want), json.loads(got))
                rc = 1
    # os._exit skips GTK teardown (it can SIGABRT offscreen); flush first.
    sys.stdout.flush()
    os._exit(rc)


def _diff(want, got):
    """Point at the state that drifted, rather than dumping the whole tree."""
    for i in range(max(len(want), len(got))):
        a = want[i] if i < len(want) else None
        b = got[i] if i < len(got) else None
        if a != b:
            name = (b or a).get("state")
            print("  state %r differs:" % name)
            print("    want: %s" % json.dumps((a or {}).get("sidebar"))[:400])
            print("    got:  %s" % json.dumps((b or {}).get("sidebar"))[:400])


if __name__ == "__main__":
    main()
