"""Everything a user can touch, declared once. Stdlib only.

This is the catalogue that used to be `SUBJECTS` / `MODELS` / `BGS` / `SEG_MODES`
/ `SEG_DETAIL` / `SEG_FOCUS` in app.py, hardcoded next to the GTK rows that read
them. Now it is data, and both UIs build themselves from `Settings.describe()`.

It also owns the *resolvers*: the small mappings from what the user picked to what
the compute side is actually asked for ("Subject: Person" -> `birefnet-portrait`,
"Background: Custom…" -> the colour they chose, "Detail: Auto" -> no override).
Those live here rather than in the shell because they are exactly the sort of
quiet mapping that a second implementation gets subtly wrong — and here, one
golden pins them for both cores.
"""
from .settings import (
    Choice, Setting, Group, Schema, Settings,
    CHOICE, BOOL, INT, COLOR, EXPANDER, PAGE,
    eq, ne, truthy,
)

# ---------------------------------------------------------------- the values --
SUBJECTS = [
    Choice("birefnet-general", "General (objects, scenes)"),
    Choice("birefnet-portrait", "Person / portrait"),
    Choice("isnet-anime", "Anime / illustration"),
    Choice("u2net", "Fast (lower quality)"),
    # Reveals the exact-model row below, the same way Background's "Custom…"
    # reveals the colour picker. No hidden mode flag.
    Choice("custom", "Custom model…"),
]

MODELS = [
    Choice("birefnet-general", "BiRefNet — General"),
    Choice("birefnet-general-lite", "BiRefNet — General Lite"),
    Choice("birefnet-massive", "BiRefNet — Massive"),
    Choice("birefnet-portrait", "BiRefNet — Portrait"),
    Choice("birefnet-hrsod", "BiRefNet — HRSOD"),
    Choice("birefnet-dis", "BiRefNet — DIS"),
    Choice("isnet-general-use", "ISNet — General"),
    Choice("isnet-anime", "ISNet — Anime"),
    Choice("u2net", "U2Net"),
    Choice("u2net_human_seg", "U2Net — Human Seg"),
    Choice("silueta", "Silueta"),
]

BACKGROUNDS = [
    Choice("transparent", "Transparent"),
    Choice("blur", "Blur background"),
    Choice("#ffffff", "White"),
    Choice("#000000", "Black"),
    Choice("#00b140", "Green screen"),
    Choice("custom", "Custom…"),
]

SEG_MODES = [
    Choice("everything", "Everything"),
    Choice("point", "Click to select"),
]

#: Grid density for "segment everything" — points per side (0 = let the worker pick).
SEG_DETAIL = [
    Choice(0, "Auto"),
    Choice(16, "Fast"),
    Choice(24, "Balanced"),
    Choice(32, "Fine"),
    Choice(44, "Maximum"),
]

#: How long to hover before drilling from the whole object to the part, in ms.
SEG_FOCUS = [
    Choice(800, "Quick"),
    Choice(1600, "Normal"),
    Choice(2600, "Relaxed"),
]

AUTO_MODEL = Choice("auto", "Auto")


# ---------------------------------------------------------------- the schema --
def schema():
    """The sidebar, declared. Group order here is the order on screen."""
    return Schema([
        Group("model", "Model", visible_when=[ne("page", "segment")], settings=[
            Setting("subject", CHOICE, "Subject",
                    subtitle="What the image mainly contains",
                    options=SUBJECTS, default="birefnet-general"),
            Setting("model", CHOICE, "Model",
                    subtitle="Exact rembg model",
                    options=MODELS, default="birefnet-general",
                    visible_when=[eq("subject", "custom")]),
        ]),
        Group("seg", "Segmentation",
              visible_when=[eq("page", "segment"), truthy("seg_available")],
              settings=[
                  Setting("seg_mode", CHOICE, "Mode", options=SEG_MODES,
                          default="everything", surface=PAGE),
                  Setting("seg_focus", CHOICE, "Focus speed",
                          subtitle="Hover this long to focus a sub-object",
                          options=SEG_FOCUS, default=1600),
                  Setting("seg_advanced", EXPANDER, "Advanced",
                          subtitle="Model and detail", children=[
                              Setting("seg_model", CHOICE, "Model",
                                      subtitle="Auto — by GPU / VRAM",
                                      options=[AUTO_MODEL], default="auto"),
                              Setting("seg_detail", CHOICE, "Detail",
                                      subtitle="Finer finds more, but is slower",
                                      options=SEG_DETAIL, default=0),
                          ]),
              ]),
        Group("output", "Output", settings=[
            Setting("bg", CHOICE, "Background",
                    options=BACKGROUNDS, default="transparent"),
            Setting("bg_color", COLOR, "Custom colour", default="#00b140",
                    sensitive_when=[eq("bg", "custom")]),
            Setting("blur", INT, "Blur strength", default=20,
                    minimum=2, maximum=80, step=1,
                    visible_when=[eq("bg", "blur")]),
            Setting("alpha", BOOL, "Alpha matting",
                    subtitle="Cleaner edges, a little slower", default=False,
                    visible_when=[ne("page", "segment")]),
        ]),
    ])


def new_settings():
    return Settings(schema(), context={"page": "single", "seg_available": False})


# ------------------------------------------------------------- the resolvers --
def model_id(st):
    """Which rembg model to actually run. "Custom model…" defers to the exact
    picker; every other subject *is* a model id."""
    subject = st.get("subject")
    return st.get("model") if subject == "custom" else subject


def background(st):
    """What the worker should composite behind the cutout: "transparent",
    "blur", or an #rrggbb colour. "Custom…" resolves to the chosen colour."""
    bg = st.get("bg")
    return st.get("bg_color") if bg == "custom" else bg


def blur_strength(st):
    return int(st.get("blur"))


def alpha_matting(st):
    return bool(st.get("alpha"))


def seg_detail(st):
    """Points per side, or None to let the worker choose."""
    return st.get("seg_detail") or None


def seg_model(st):
    """The SAM rung, or "auto"."""
    return st.get("seg_model") or "auto"


def dwell_ms(st):
    return st.get("seg_focus")


def seg_mode(st):
    return st.get("seg_mode")


# --------------------------------------------------- the runtime SAM ladder --
def vram_label(mb):
    if not mb:
        return "CPU · low VRAM"
    return ("~%.1f GB" % (mb / 1000)).replace(".0", "")


def seg_model_choices(models):
    """The Model options once the worker has reported what this GPU can hold.

    `models` is the worker's ladder: [{"rung", "label", "vram"}].
    """
    out = [Choice("auto", "Auto — best for your GPU")]
    for m in models:
        out.append(Choice(m["rung"], "%s · %s"
                          % (m["label"], vram_label(m.get("vram", 0)))))
    return out
