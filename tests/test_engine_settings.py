#!/usr/bin/env python3
"""The settings schema: the UiSchema snapshot, the rules, and the resolvers.

`spec/tools/uidump.py` proves the GTK sidebar renders the schema correctly. This
proves the schema itself — with no GTK at all — because it is the *schema* that a
TypeScript sidebar will be built from, not the widgets.

Three things are pinned:

  1. **The UiSchema snapshot.** `describe()` is the whole contract between the
     engine and any UI. Frozen as JSON, so a setting that quietly loses its
     subtitle, changes its default or reorders its options shows up as a diff.
  2. **The reactive rules**, which used to be signal handlers wiring one widget
     to another and are now `visible_when` / `sensitive_when` data.
  3. **The resolvers** — the mappings from what the user picked to what the
     worker is actually asked for. "Subject: Person" -> `birefnet-portrait`,
     "Background: Custom…" -> the chosen colour, "Detail: Auto" -> *no* override.
     These are the quiet, silently-wrong-output kind of logic, so both cores get
     to check themselves against one golden.

    python tests/test_engine_settings.py            # check
    python tests/test_engine_settings.py --freeze   # re-freeze the snapshot
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src", "bgbg"))

from engine import interactables as IX                          # noqa: E402

GOLDEN = os.path.join(ROOT, "spec", "goldens", "ui_schema.json")

FAILED = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILED.append(label)
    print("  %-52s %s" % (label, "PASS" if ok else "FAIL"))
    if not ok:
        print("      want: %r" % (want,))
        print("      got:  %r" % (got,))


# ------------------------------------------------------------- the snapshot ---
def snapshot(freeze=False):
    """describe() is the engine's whole UI contract — freeze it."""
    st = IX.new_settings()
    got = json.dumps(st.describe(), indent=1, sort_keys=True)
    if freeze:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            f.write(got + "\n")
        print("froze %s (%d B)" % (GOLDEN, len(got)))
        return
    if not os.path.exists(GOLDEN):
        FAILED.append("ui_schema golden missing")
        print("  MISSING GOLDEN %s" % GOLDEN)
        return
    with open(GOLDEN) as f:
        want = f.read().rstrip("\n")
    if want == got:
        print("  %-52s %s" % ("UiSchema snapshot", "PASS"))
    else:
        FAILED.append("UiSchema snapshot")
        with open(GOLDEN + ".actual", "w") as f:
            f.write(got + "\n")
        print("  %-52s %s" % ("UiSchema snapshot", "FAIL (wrote .actual)"))


# ----------------------------------------------------------------- the rules ---
def test_rules():
    """What used to be _on_subject_changed / _sync_color, as data."""
    st = IX.new_settings()

    check("by default the exact-model row is hidden", st.visible("model"), False)
    st.set("subject", "custom")
    check("... and Custom model… reveals it", st.visible("model"), True)
    st.set("subject", "birefnet-general")

    check("the blur row is hidden unless the bg is Blur",
          st.visible("blur"), False)
    st.set("bg", "blur")
    check("... and Blur background reveals it", st.visible("blur"), True)

    check("the colour row is insensitive unless the bg is Custom…",
          st.sensitive("bg_color"), False)
    st.set("bg", "custom")
    check("... and Custom… makes it sensitive", st.sensitive("bg_color"), True)
    check("... which also hides the blur row", st.visible("blur"), False)

    # page / worker context
    check("the Model group shows on the Single page",
          st.group_visible("model"), True)
    check("the Segmentation group needs the segment page AND a worker with SAM",
          st.group_visible("seg"), False)

    st.set_context(page="segment")
    check("... on the segment page alone, still not (no SAM yet)",
          st.group_visible("seg"), False)
    check("... and the Model group hides there", st.group_visible("model"), False)
    check("... as does Alpha matting", st.visible("alpha"), False)

    st.set_context(seg_available=True)
    check("... with SAM available, it shows", st.group_visible("seg"), True)


# ------------------------------------------------------------- the resolvers ---
def test_resolvers():
    st = IX.new_settings()

    check("subject General -> the birefnet-general model",
          IX.model_id(st), "birefnet-general")
    st.set("subject", "birefnet-portrait")
    check("subject Person -> birefnet-portrait", IX.model_id(st),
          "birefnet-portrait")

    st.set("subject", "custom")
    st.set("model", "silueta")
    check("subject Custom model… -> the exact picker wins",
          IX.model_id(st), "silueta")

    check("background Transparent -> 'transparent'",
          IX.background(st), "transparent")
    st.set("bg", "#00b140")
    check("background Green screen -> its hex", IX.background(st), "#00b140")
    st.set("bg", "blur")
    check("background Blur -> 'blur'", IX.background(st), "blur")
    st.set("bg", "custom")
    st.set("bg_color", "#123456")
    check("background Custom… -> the colour that was picked",
          IX.background(st), "#123456")

    check("blur strength is an int", IX.blur_strength(st), 20)
    check("alpha matting is off by default", IX.alpha_matting(st), False)

    # The one that would silently change behaviour if a port got it wrong:
    # "Auto" is 0, and 0 must become *no* points_per_side override, not 0 points.
    check("detail Auto -> None (no override), NOT 0",
          IX.seg_detail(st), None)
    st.set("seg_detail", 32)
    check("detail Fine -> 32 points per side", IX.seg_detail(st), 32)

    check("focus speed defaults to Normal (1600ms)", IX.dwell_ms(st), 1600)
    st.set("seg_focus", 800)
    check("... Quick is 800ms", IX.dwell_ms(st), 800)


# ----------------------------------------------- the SAM ladder, at runtime ---
def test_seg_model_ladder():
    """The Model options only exist once the worker says what the GPU can hold."""
    st = IX.new_settings()
    check("before the worker reports, the only option is Auto",
          [c.label for c in st.options("seg_model")], ["Auto"])
    check("... and it resolves to 'auto'", IX.seg_model(st), "auto")

    st.set_options("seg_model", IX.seg_model_choices([
        {"rung": "tiny", "label": "SAM 2.1 Tiny", "vram": 1500},
        {"rung": "mobile", "label": "MobileSAM", "vram": 900},
        {"rung": "cpu", "label": "MobileSAM (CPU)", "vram": 0},
    ]))
    check("the ladder arrives, with VRAM labels",
          [c.label for c in st.options("seg_model")],
          ["Auto — best for your GPU", "SAM 2.1 Tiny · ~1.5 GB",
           "MobileSAM · ~0.9 GB", "MobileSAM (CPU) · CPU · low VRAM"])
    check("... Auto stays selected across the reload", IX.seg_model(st), "auto")
    check("... and the status line can name it",
          st.label("seg_model"), "Auto — best for your GPU")

    st.set("seg_model", "tiny")
    check("picking a rung resolves to the rung", IX.seg_model(st), "tiny")

    # A ladder that no longer contains the choice must not leave it dangling.
    st.set_options("seg_model", IX.seg_model_choices([]))
    check("a ladder without the chosen rung falls back to Auto",
          IX.seg_model(st), "auto")


def main():
    if "--freeze" in sys.argv:
        snapshot(freeze=True)
        return 0
    print("UiSchema")
    snapshot()
    print("the reactive rules (were signal handlers, now data)")
    test_rules()
    print("the resolvers (pick -> what the worker is asked for)")
    test_resolvers()
    print("the SAM ladder, replaced at runtime")
    test_seg_model_ladder()
    print()
    if FAILED:
        print("SETTINGS FAILED (%d): %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("SETTINGS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
