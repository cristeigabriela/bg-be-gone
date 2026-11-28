#!/usr/bin/env python3
"""bg-be-gone — GTK/libadwaita frontend.

Runs on the system Python. Image processing happens in a persistent worker
subprocess (worker.py) inside the bundled virtualenv, which keeps the model
resident on the GPU. Frontend and worker talk over line-delimited JSON.
"""
import os
import sys
import enum
import json
import math
import logging
import secrets
import shutil
import tempfile
import threading
import subprocess
from dataclasses import dataclass
from typing import Optional

import plat
plat.frontend_starting()   # before GTK initialises: it reads GDK_DEBUG then

import gi  # noqa: E402
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Gdk, GdkPixbuf, Gio, GLib, Adw  # noqa: E402

from viewer import (  # noqa: E402
    ImageView,
    Loaded,
    Transform,
    transform_pixbuf,
)
import background  # noqa: E402
import batchnames  # noqa: E402
import formats  # noqa: E402
from gi.repository import GObject, Graphene, Gsk, Pango  # noqa: E402
import modelinfo  # noqa: E402
from modelinfo import Device, Pick  # noqa: E402
from engines import selection  # noqa: E402  (standard library only)
import offmain  # noqa: E402
import settings  # noqa: E402
from widgets import action_bar, pill_button, primary_button, save_button  # noqa: E402

APP_ID = "io.github.cristeigabriela.BgBeGone"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PY = os.environ.get("BGBG_VENV_PYTHON") or plat.default_worker_python()
WORKER = os.environ.get("BGBG_WORKER", os.path.join(APP_DIR, "worker.py"))
LOG = plat.worker_log_path()
APP_VERSION = os.environ.get("BGBG_VERSION", "1.1.0")
_log = logging.getLogger("bgbg")


def _alpha_of(bg):
    """Whether a result made with background setting `bg` is transparent."""
    if bg == "transparent":
        return formats.Alpha.TRANSPARENT
    return formats.Alpha.FLATTENED


class Job(enum.Enum):
    """What a request sent to the worker is for.

    Every reply carries its request's id. Window keeps the requests still in
    flight by id, so a reply for one that no longer applies (its image was
    replaced, or the page unloaded its model) is dropped instead of landing
    on new work.
    """

    OPEN = "open"                # decode a new source, or count its frames
    SINGLE = "single"            # Generate, on a still or an animation
    BATCH = "batch"
    SAVE = "save"                # re-encode a result as WebP or JPEG for Save
    SEG_LOAD = "seg_load"        # load SAM and encode the image
    SEG_RUN = "seg_run"          # Segment everything, or a click in point mode
    SEG_PREVIEW = "seg_preview"  # the live composite in Segment's result panel
    SEG_SAVE = "seg_save"        # the full-size render behind Save selection
    RECOMPOSITE = "recomposite"  # the shown result, put onto a changed background


# Reply types that belong to one request; any other reply is window-wide state.
_JOB_REPLIES = {
    "loading", "done_single", "gif_progress", "gif_done", "canceled", "progress",
    "done_batch", "converted", "frames", "seg_ready", "seg_download", "seg_step",
    "seg_progress", "seg_objects", "seg_mask", "seg_extracted", "seg_canceled", "error",
    "download", "recomposited",
}
# Reply types that end their request.
_JOB_ENDS = {
    "done_single", "gif_done", "canceled", "done_batch", "converted", "frames",
    "seg_ready", "seg_objects", "seg_mask", "seg_extracted", "seg_canceled", "error",
    "recomposited",
}
# How long a background setting rests before the shown result follows it.
_RECOMPOSITE_DELAY_MS = 150
# Replies after which a Generate's kept folder belongs to the shown result.
_KEPT_ADOPTED = {"done_single", "gif_done"}
_SEG_JOBS = {Job.SEG_LOAD, Job.SEG_RUN, Job.SEG_PREVIEW, Job.SEG_SAVE}


class Abandoned(enum.Enum):
    """Whether Window._abandon_jobs found any of the jobs it was given."""

    SOME = "some"
    NONE = "none"


class Prep(enum.Enum):
    """Whose input file is being encoded, before its request can be sent."""

    SINGLE = "single"            # Generate's input
    SEGMENT = "segment"          # Segment's model load


_PATTERN_HINT = ("<small>Tokens: <tt>{name}</tt> original filename, "
                 "<tt>{n}</tt> index. Output is PNG in the output folder; a name "
                 "that's taken gets a number.</small>")
# Failed or skipped files a batch summary lists by name, so the dialog fits on screen.
_MAX_LISTED = 15


class Quit(enum.Enum):
    """Whether a quit is waiting for something before the window closes."""

    NOT_ASKED = "not_asked"
    AFTER_SAVES = "after_saves"      # once the saves being written are done
    AFTER_BATCH = "after_batch"      # once the cancelled batch has stopped


# How long "Stop and quit" waits for the cancelled batch to stop: the file in
# progress finishes first, and the worker deletes its own unfinished file.
_BATCH_STOP_WAIT_MS = 3000


class SaveOutcome(enum.Enum):
    WRITTEN = "written"
    FAILED = "failed"


class WorkerState(enum.Enum):
    UP = "up"          # started; requests go through
    DOWN = "down"      # failed to start or exited: nothing would answer


@dataclass
class _PendingOpen:
    """A source the worker is decoding or probing before the app adopts it."""

    path: str
    preview: Optional[str]   # what GdkPixbuf already shows, or None until the worker's PNG


@dataclass
class _PendingSave:
    """A result the worker is writing as WebP or JPEG."""

    target: formats.SaveTarget
    choices: tuple
    staging: str             # our temporary copy of the result, deleted afterwards
    part: str                # the temp file beside the target the worker writes


# PNG options for render(path, png_options) in Window._save_result: the saved
# file gets GdkPixbuf's defaults; a hand-off to the worker, read once, is fast.
_PNG_FINAL = ([], [])
_PNG_HANDOFF = (["compression"], ["1"])


def _part_beside(target):
    """An empty temp file next to `target`, named in its format, to save into.

    Beside it, so os.replace() swaps it in whole; in its format's suffix, so
    the worker, which picks the format from the name, writes the right one.
    Not mkstemp: its files are private (0600), and this one becomes the
    saved image, which gets the usual permissions the umask allows.
    """
    folder, name = os.path.split(target.path)
    while True:
        part = os.path.join(folder, ".%s.%s.part%s" % (
            name, secrets.token_hex(4), target.format.suffix))
        try:
            os.close(os.open(part, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
            return part
        except FileExistsError:
            continue


@dataclass
class _InFlight:
    """A request sent to the worker, kept by its id until its last reply."""

    job: Job
    # SINGLE and SEG_SAVE: whether the result is transparent, which decides
    # if JPEG is offered. Recorded when sent, never read from the settings later.
    alpha: Optional[formats.Alpha] = None
    source: Optional[_PendingOpen] = None    # OPEN: the source being opened
    save: Optional[_PendingSave] = None      # SAVE: where the re-encoded result goes
    input: Optional[str] = None              # SINGLE: its input, deleted if we encoded it
    kept: Optional["_Kept"] = None           # SINGLE: what it keeps, the result's once shown
    loads: Optional[tuple] = None            # SEG_LOAD: the (source, model) it loads


@dataclass
class _Kept:
    """What a result was made from, so its background can change without the model.

    The worker writes it into `folder`, in the window's temp folder; the
    window deletes the folder when the result goes (_forget_kept), and the
    whole temp folder goes when the window closes.
    """

    folder: str                       # the picture and cut-out, or a cut-out per frame
    bg: str                           # the background the shown result has
    blur: int
    # An animation's source and its baked rotate/flip, which recomposite
    # reads again; None for a still.
    animation: Optional[dict] = None


# A remembered window side outside this range is not a size to restore.
_MIN_WINDOW_SIDE, _MAX_WINDOW_SIDE = 200, 16384


def _window_dimension(value):
    """A remembered window width or height, or None when it isn't a sane one."""
    if isinstance(value, int) and not isinstance(value, bool) \
            and _MIN_WINDOW_SIDE <= value <= _MAX_WINDOW_SIDE:
        return value
    return None


def _existing_folder(value):
    """A remembered folder, or None when it isn't a path to one that exists."""
    try:
        return value if isinstance(value, str) and os.path.isdir(value) else None
    except ValueError:              # a path with a NUL in it
        return None


class PasteWriteError(OSError):
    """The pasted image couldn't be written to the temp folder."""

    def __init__(self, path):
        super().__init__(f"could not write the pasted image to {path}")
        self.path = path


def plain_reason(exc):
    """The OS's own words for an OSError, else the exception's first line."""
    if isinstance(exc, OSError) and exc.strerror:
        return exc.strerror
    if isinstance(exc, GLib.Error):
        return exc.message
    return (str(exc).splitlines() or [type(exc).__name__])[0]


def _remove_quietly(path):
    """Delete a temporary file of ours; one already gone is fine."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _saved_notice(target, choices):
    """The toast after a save, saying why when the typed format wasn't used."""
    name = os.path.basename(target.path)
    if target.change is not formats.SuffixChange.REPLACED:
        return "Saved " + name
    if target.typed is formats.JPEG and choices == formats.save_formats(formats.Alpha.TRANSPARENT):
        reason = "JPEG can't keep the transparency"
    else:
        reason = "this result can't be saved as %s" % target.typed.name
    return "Saved %s as %s: %s." % (name, target.format.name, reason)


# Start-up check for CI: with BGBG_SELFTEST set to a file path, the app writes
# the worker's "ready" message (or why it never came) there as JSON, then quits.
SELFTEST = os.environ.get("BGBG_SELFTEST")
# The app is single-instance per application id, so a test launch with the
# real id would open its window inside a bg-be-gone the user already runs.
# BGBG_APP_ID gives a test its own instance; the icon keeps APP_ID.
INSTANCE_ID = os.environ.get("BGBG_APP_ID", APP_ID)
_SELFTEST_TIMEOUT_S = 300   # generous: ready took 9 s locally; CI runners are slower


def _finish_selftest(win, result):
    result = dict(result, opened=win.get_application().opened)
    # Written aside and renamed, so a watcher never reads half a file.
    with open(SELFTEST + ".part", "w", encoding="utf-8") as f:
        json.dump(result, f)
    os.replace(SELFTEST + ".part", SELFTEST)
    win.close()        # close-request shuts the worker down, then the app exits
    return False       # one-shot when used as a GLib timeout

SUBJECTS = [
    ("General (objects, scenes)", "birefnet-general"),
    ("Person / portrait", "birefnet-portrait"),
    ("Anime / illustration", "isnet-anime"),
    ("Fast (lower quality)", "u2net"),
]
CUSTOM_SUBJECT = "custom"      # the Subject entry that reveals the Model row
# Model > Advanced > Engine: what runs BiRefNet. Automatic leaves it to the
# install's profile, then ONNX Runtime (engines/selection.py).
ENGINES = [
    ("Automatic", selection.Engine.AUTO),
    ("ONNX Runtime", selection.Engine.ORT),
    ("PyTorch", selection.Engine.TORCH),
]
# The Output button's short name for each background.
BG_SHORT_NAMES = {
    "transparent": "Transparent",
    "blur": "Blur",
    "#ffffff": "White",
    "#000000": "Black",
    "#00b140": "Green screen",
    "custom": "Custom",
}
BGS = [
    ("Transparent", "transparent"),
    ("Blur background", "blur"),
    ("White", "#ffffff"),
    ("Black", "#000000"),
    ("Green screen", "#00b140"),
    ("Custom…", "custom"),
]
# Segment's model row: its Automatic entry, and seg_ready's mode -> how the model was picked.
SEG_AUTO = "auto"
SEG_PICKS = {p.value: p for p in (Pick.AUTO, Pick.MANUAL, Pick.FALLBACK)}


def _seg_reason(seg_ready):
    """What Auto picked by, in a few words: "11.2 GB free", or None.

    ``vram_free`` (MiB) in seg_ready is the free VRAM the worker measured when
    Auto picked; it's null when nothing was probed. It stays true when the
    model then fell back to the CPU, which ``gpu`` says separately.
    """
    free = seg_ready.get("vram_free")
    return modelinfo.free_text(free) if free is not None else None


SEG_MODES = [
    ("Everything — pick layers", "everything"),
    ("Click to select", "point"),
]
# Grid density for "segment everything": label -> points-per-side (0 = auto).
SEG_DETAIL = [
    ("Auto", 0),
    ("Fast", 16),
    ("Balanced", 24),
    ("Fine", 32),
    ("Maximum", 44),
]
# Hover-dwell before drilling general -> specific: label -> milliseconds.
SEG_FOCUS = [
    ("Quick", 800),
    ("Normal", 1600),
    ("Relaxed", 2600),
]
PROVIDER_LABELS = {
    "CUDAExecutionProvider": "NVIDIA (CUDA)",
    "ROCMExecutionProvider": "AMD (ROCm)",
    "MIGraphXExecutionProvider": "AMD (MIGraphX)",
    "DmlExecutionProvider": "DirectML",
    "CPUExecutionProvider": "CPU",
}

CSS = """
/* bg-be-gone — playful, Apple-ish polish */
.view-frame {
  background: #17171a;
  border-radius: 16px;
  box-shadow: 0 10px 30px alpha(black, 0.35);
}
.zoom-badge {
  background: alpha(black, 0.5); color: white; border-radius: 999px;
  padding: 2px 10px; margin: 10px; font-size: 0.78em;
}
.cursor-badge {
  background: alpha(black, 0.72); color: white; border-radius: 999px;
  padding: 3px 11px; font-size: 0.8em;
  box-shadow: 0 3px 10px alpha(black, 0.35);
}
.panel-title { font-weight: 700; opacity: 0.85; letter-spacing: 0.2px; }

.statusbar { min-height: 26px; font-size: 0.9em; }
.statusbar .dev-dot { min-width: 10px; min-height: 10px; }

.action-bar {
  padding: 8px;
  border-radius: 18px;
  background: alpha(@window_bg_color, 0.6);
  box-shadow: 0 1px 2px alpha(black, 0.15), inset 0 1px 0 alpha(white, 0.03);
}
.status-label { font-size: 0.92em; }

button.pill {
  border-radius: 999px;
  padding: 6px 16px;
  transition: transform 120ms ease, box-shadow 150ms ease;
}
button.pill:hover { box-shadow: 0 3px 10px alpha(black, 0.22); }
button.pill:active { transform: scale(0.96); }
button.suggested-action.pill { box-shadow: 0 4px 14px alpha(@accent_bg_color, 0.45); }

.empty-title { font-size: 1.25em; font-weight: 700; }
.empty-hint { opacity: 0.62; }
.tip-banner { font-size: 0.94em; }

/* device indicator dot — vendor coloured */
.dev-dot {
  min-width: 11px; min-height: 11px; border-radius: 999px;
  background: #8a8a8a; box-shadow: 0 0 0 2px alpha(black, 0.15);
}
.dev-nvidia { background: #76b900; box-shadow: 0 0 8px alpha(#76b900, 0.7); }
.dev-amd    { background: #ed1c24; box-shadow: 0 0 8px alpha(#ed1c24, 0.6); }
.dev-dml    { background: #0a84ff; box-shadow: 0 0 8px alpha(#0a84ff, 0.6); }
.dev-cpu    { background: #9aa0a6; }

/* footer progress (ProgressMotion): a sheen sweeps the filled part while work
   runs, when GTK draws on the GPU; with no fraction yet, a pulse sweeps the
   whole bar. Accent only. */
progressbar.footer-progress > trough > progress { background-repeat: no-repeat; }
progressbar.footer-progress.running.sheen > trough > progress {
  background-image: linear-gradient(90deg, alpha(white, 0), alpha(white, 0.45), alpha(white, 0));
  background-size: 48px 100%;
  animation: footer-sheen 1.6s ease-in-out infinite;
}
@keyframes footer-sheen {
  from { background-position: -48px 0; }
  to   { background-position: 200px 0; }
}
progressbar.footer-progress.unknown > trough > progress {
  background-color: alpha(@accent_bg_color, 0.3);
  background-image: linear-gradient(90deg, alpha(@accent_bg_color, 0), @accent_bg_color,
                                    alpha(@accent_bg_color, 0));
  background-size: 64px 100%;
  animation: footer-pulse 1.2s ease-in-out infinite alternate;
}
@keyframes footer-pulse {
  from { background-position: -24px 0; }
  to   { background-position: 110px 0; }
}

/* Source | Result: the handle is the gap between the two cards, with no line.
   It keeps a 12 px grab area and the resize cursor; hovering or dragging it
   shows a short grip, a shape rather than a colour change. */
paned.card-split > separator {
  min-width: 12px;
  background: none;
  box-shadow: none;
  border: none;
}
paned.card-split > separator:hover,
paned.card-split > separator:active {
  background-image: image(alpha(@window_fg_color, 0.35));
  background-size: 4px 40px;
  background-position: center;
  background-repeat: no-repeat;
}
"""


def model_picker(title, sections, describe):
    """An Adw.ComboRow of models, in titled sections, two lines per model.

    The row shows a model's short name. The open list shows its full name and
    a dim line under it, and never ellipsizes, so it grows as wide as its
    longest line instead of cutting names off.

    Args:
        title: the row's title.
        sections: ``[(header or None, [key, ...]), ...]`` in picker order.
        describe: ``key -> (short, name, subtitle)``, read each time a model is shown.

    Returns:
        ``(row, refresh)``: the row, whose selected item is a Gtk.StringObject
        holding the key, and a function that redraws every entry, for when
        what ``describe`` says has changed (the engine, and so the VRAM).
    """
    stores = Gio.ListStore(item_type=Gio.ListModel)
    headers = {}            # first key of a section -> that section's header
    for header, keys in sections:
        stores.append(Gtk.StringList.new(keys))
        if header and keys:
            headers[keys[0]] = header
    # A flattened list of lists is a section model: one section per list.
    model = Gtk.FlattenListModel.new(stores)

    def setup_value(_factory, item):
        # The row may ellipsize; the open list shows the whole name.
        item.set_child(Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.END))

    def bind_value(_factory, item):
        item.get_child().set_label(describe(item.get_item().get_string())[0])

    def setup_entry(_factory, item):
        text = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        text.append(Gtk.Label(xalign=0))
        sub = Gtk.Label(xalign=0)
        sub.add_css_class("dim-label")
        sub.add_css_class("caption")
        text.append(sub)
        # A custom list factory drops libadwaita's checkmark, so draw our own.
        check = Gtk.Image(icon_name="object-select-symbolic")
        entry = Gtk.Box(spacing=12)
        entry.append(text)
        entry.append(check)
        item.set_child(entry)

    # The check marks the row's choice. Not the list item's own selected
    # state: the open list selects whatever the pointer is over.
    checks = {}             # each bound entry's check -> the key it shows

    def show_check(check):
        chosen = row.get_selected_item()
        mark = chosen is not None and checks.get(check) == chosen.get_string()
        check.set_opacity(1 if mark else 0)

    def bind_entry(_factory, item):
        key = item.get_item().get_string()
        _short, name, subtitle = describe(key)
        text = item.get_child().get_first_child()
        text.get_first_child().set_label(name)
        text.get_last_child().set_label(subtitle)
        check = text.get_next_sibling()
        checks[check] = key
        show_check(check)

    def unbind_entry(_factory, item):
        checks.pop(item.get_child().get_last_child(), None)

    def setup_header(_factory, header):
        # Indented to line up with the entries' text: list headers get no row padding.
        label = Gtk.Label(xalign=0, margin_start=16, margin_top=6, margin_bottom=2)
        label.add_css_class("caption-heading")
        label.add_css_class("dim-label")
        header.set_child(label)

    header_labels = set()   # every header label made, to find a section's on screen

    def bind_header(_factory, header):
        label = header.get_child()
        label.set_label(headers.get(header.get_item().get_string(), ""))
        header_labels.add(label)

    def factory(setup, bind, unbind=None):
        f = Gtk.SignalListItemFactory()
        f.connect("setup", setup)
        f.connect("bind", bind)
        if unbind is not None:
            f.connect("unbind", unbind)
        return f

    row = Adw.ComboRow(title=title, model=model)

    def refresh():
        # New factories rebind every entry, and the row's own value.
        row.set_factory(factory(setup_value, bind_value))
        row.set_list_factory(factory(setup_entry, bind_entry, unbind_entry))
    refresh()
    row.connect("notify::selected", lambda *_: [show_check(c) for c in list(checks)])
    # Section headers need libadwaita 1.6; older ones list the same models unheaded.
    if not headers or not hasattr(row, "set_header_factory"):
        return row, refresh
    row.set_header_factory(factory(setup_header, bind_header))

    # The section each position is in: (its first position, its header).
    section_of, at = [], 0
    for header, keys in sections:
        section_of += [(at, header)] * len(keys)
        at += len(keys)

    # The list opens on the selected model with its section's header above it.
    # libadwaita keeps the list where it was, and GTK, scrolling to a row,
    # stops with the row at the edge and its header out of sight.
    def scroll_to_selected(listview, scrolled):
        selected = max(row.get_selected(), 0)
        listview.scroll_to(selected, Gtk.ListScrollFlags.NONE, None)
        _after_next_frame(scrolled, show_header, listview, scrolled, selected, 1)

    def show_header(listview, scrolled, selected, retries):
        start, header = section_of[selected]
        label = next((w for w in header_labels
                      if w.get_mapped() and w.get_label() == header), None)
        if label is None:
            # Too far up to have a widget: bring the section's first row in, then look again.
            if retries:
                listview.scroll_to(start, Gtk.ListScrollFlags.NONE, None)
                _after_next_frame(scrolled, show_header, listview, scrolled, selected,
                                  retries - 1)
            return
        ok, at = label.get_parent().compute_point(scrolled, Graphene.Point())
        if ok and at.y < 0:
            # Sections are shorter than the list, so the selection stays in view.
            adj = scrolled.get_vadjustment()
            adj.set_value(0.0 if start == 0 else max(0.0, adj.get_value() + at.y))

    popover = next((w for w in _descendants(row) if isinstance(w, Gtk.Popover)), None)
    inside = list(_descendants(popover)) if popover is not None else []
    listview = next((w for w in inside if isinstance(w, Gtk.ListView)), None)
    scrolled = next((w for w in inside if isinstance(w, Gtk.ScrolledWindow)), None)
    if listview is not None and scrolled is not None:
        popover.connect("map", lambda *_: _after_next_frame(
            scrolled, scroll_to_selected, listview, scrolled))
    return row, refresh


def _after_next_frame(widget, callback, *args):
    """Call `callback(*args)` once, after `widget`'s next frame is laid out and drawn.

    An idle callback can run before the frame it waits for: the frame clock
    paces frames to the display, and idles run whenever the loop is free.
    """
    clock = widget.get_frame_clock()
    if clock is None:
        return
    handler = None

    def run(_clock):
        clock.disconnect(handler)
        callback(*args)
    handler = clock.connect("after-paint", run)
    clock.request_phase(Gdk.FrameClockPhase.AFTER_PAINT)


class Motion(enum.Enum):
    """What a progress bar's motion says; each is a CSS class with its own animation."""

    RUNNING = "running"        # a known fraction: a sheen sweeps the filled part, on the GPU
    UNKNOWN = "unknown"        # no fraction yet: a pulse sweeps the whole bar


class ProgressMotion:
    """How a progress bar moves: eased steps, a sheen on the GPU while work runs, a fade-out.

    Every motion runs on the frame clock, as CSS or libadwaita animations,
    never as an idle or timeout per frame: idle callbacks once starved the
    worker's replies. Both kinds follow gtk-enable-animations, so with
    animations off the bar jumps and nothing moves. A hidden bar runs nothing.
    """

    EASE_MS = 250
    FADE_MS = 300
    SHEEN = "sheen"                 # the CSS class that adds the sweep to RUNNING

    def __init__(self, bar):
        self._bar = bar
        self._mode = None               # the Motion shown, or None while hidden
        self._target = 0.0              # the fraction the bar shows or eases to
        self._last_step_us = 0          # when move_to last ran, monotonic microseconds
        self._ease = Adw.TimedAnimation.new(
            bar, 0.0, 0.0, self.EASE_MS,
            Adw.CallbackAnimationTarget.new(lambda value: bar.set_fraction(value)))
        self._ease.set_easing(Adw.Easing.EASE_OUT_CUBIC)
        self._fade = Adw.TimedAnimation.new(
            bar, 1.0, 0.0, self.FADE_MS, Adw.PropertyAnimationTarget.new(bar, "opacity"))
        self._fade.connect("done", self._on_faded)

    def move_to(self, fraction):
        """Show the bar heading to `fraction`; it never eases backwards."""
        # A new run, one after the pulse, or one that started over: jump there.
        restart = self._mode is not Motion.RUNNING or fraction < self._target
        # Steps closer together than an ease already look smooth, and easing
        # each of them would redraw every frame for as long as they keep coming.
        now = GLib.get_monotonic_time()
        frequent = now - self._last_step_us < self.EASE_MS * 1000
        self._last_step_us = now
        self._show(Motion.RUNNING)
        if restart or frequent:
            self._ease.reset()
            self._bar.set_fraction(fraction)
        else:
            self._ease.set_value_from(self._bar.get_fraction())
            self._ease.set_value_to(fraction)
            self._ease.play()
        self._target = fraction

    def unknown(self):
        """Show work with no fraction yet: a soft pulse across the whole bar."""
        self._show(Motion.UNKNOWN)
        self._ease.reset()
        self._bar.set_fraction(1.0)

    def fade_out(self):
        if self._mode is None:
            return
        self._mode = None
        for motion in Motion:
            self._bar.remove_css_class(motion.value)
        self._fade.play()

    def _show(self, mode):
        self._fade.reset()              # a run that starts while the last one fades out
        self._bar.set_opacity(1.0)
        for motion in Motion:
            self._bar.remove_css_class(motion.value)
        self._bar.add_css_class(mode.value)
        # Cairo paints the whole window in software for every frame of the
        # sweep (a quarter of a core on Windows), so it only runs on the GPU.
        if self._drawn_on_gpu():
            self._bar.add_css_class(self.SHEEN)
        else:
            self._bar.remove_css_class(self.SHEEN)
        self._bar.set_visible(True)
        self._mode = mode

    def _drawn_on_gpu(self):
        native = self._bar.get_native()
        renderer = native.get_renderer() if native is not None else None
        return renderer is not None and not isinstance(renderer, Gsk.CairoRenderer)

    def _on_faded(self, *_):
        self._bar.set_visible(False)
        self._bar.set_opacity(1.0)
        self._ease.reset()
        self._target = 0.0


def _descendants(widget):
    """Every widget under `widget`, depth first."""
    child = widget.get_first_child()
    while child is not None:
        yield child
        yield from _descendants(child)
        child = child.get_next_sibling()


def _exit_code_text(code):
    """An exit code as people look it up: a negative one names the signal,
    and a Windows crash code (above 255) reads in hex."""
    if code < 0:
        return "%d (killed by signal %d)" % (code, -code)
    if code > 255:
        return "0x%08X" % code
    return str(code)


class Worker:
    """Owns the venv subprocess and dispatches replies to the main loop.

    When the process ends on its own (a crash, a failed start), the window
    gets a ``{"type": "worker_exited", "code": N}`` message, as if the worker
    had sent it.
    """

    def __init__(self, on_message, first_id=1):
        self._on_message = on_message
        # A restarted worker continues the old one's ids, so a late reply or
        # a request still tracked from before can never match a new request.
        self._next_id = first_id
        self._closing = False
        self.ok = False
        self.error = None
        # Requests sent before the worker's "ready", in order; None once it came.
        self._queued = []
        try:
            self._logf = open(LOG, "a")
            self.proc = subprocess.Popen(
                [VENV_PY, WORKER], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self._logf, text=True, bufsize=1,
                creationflags=plat.subprocess_flags(), env=plat.worker_env())
            threading.Thread(target=self._reader, daemon=True).start()
            self.ok = True
        except OSError as e:
            self.error = plain_reason(e)
            _log.error("starting the worker %s %s failed", VENV_PY, WORKER, exc_info=e)

    def _reader(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue             # a stray print from a library, not a reply
            # Above redraw priority: at the default idle priority, a reply could
            # wait for as long as the Segment outlines kept animating, which on
            # Windows starved it until the next input event.
            GLib.idle_add(self._on_message, msg, priority=GLib.PRIORITY_DEFAULT)
        # stdout closes when the process ends: say so, unless we asked it to.
        # From here on nothing is sent: no reply would ever come.
        code = self.proc.wait()
        self.ok = False
        if not self._closing:
            GLib.idle_add(self._on_message, {"type": "worker_exited", "code": code},
                          priority=GLib.PRIORITY_DEFAULT)

    @property
    def next_id(self):
        return self._next_id

    def send(self, req):
        """Send `req` with an id of its own; return the id, or -1 when the
        worker isn't there to answer (never started, exited, or its pipe broke)."""
        if not self.ok:
            return -1
        rid = self._next_id
        self._next_id += 1
        req["id"] = rid
        self._post(req)
        if not self.ok:
            return -1                   # writing it found the pipe broken
        return rid

    def cancel(self, rid):
        """Ask the worker to stop request `rid`: running, or skipped when it
        comes up. Written as is, since send() would give it an id of its own,
        and here the id names the target."""
        if not self.ok:
            return
        self._post({"op": "cancel", "id": rid})

    def _post(self, message):
        """Write `message` now, or hold it until "ready", in order: a cancel
        sent that early waits behind the request it names."""
        if self._queued is not None:
            self._queued.append(message)
        else:
            self._write(message)

    def _write(self, req):
        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError):   # ValueError: the pipe is already closed
            self.ok = False             # the reader reports the exit itself

    def started(self):
        """The worker sent "ready": send what waited for it."""
        queued, self._queued = self._queued or [], None
        for req in queued:
            self._write(req)

    def shutdown(self):
        if not self.ok:
            return
        self._closing = True
        try:
            self.proc.stdin.write('{"op":"shutdown"}\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass


class Panel:
    """Titled interactive image view with per-side transform buttons."""

    def __init__(self, title, on_change=None, transforms=True,
                 empty_icon="image-x-generic-symbolic",
                 empty_title="Drop an image here", empty_hint="or click Open"):
        self._ext_change = on_change
        self._last_pct = -1
        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.box.set_hexpand(True)

        header = Gtk.CenterBox()
        lbl = Gtk.Label(label=title)
        lbl.add_css_class("panel-title")
        header.set_start_widget(lbl)
        tools = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
        # Rotate/flip re-orient the image; the Segment page hides them because
        # its overlays register to un-rotated image pixels.
        buttons = []
        if transforms:
            buttons += [
                ("object-rotate-left-symbolic", "Rotate left",
                 lambda *_: self.view._rotate(-1)),
                ("object-rotate-right-symbolic", "Rotate right",
                 lambda *_: self.view._rotate(1)),
                ("object-flip-horizontal-symbolic", "Flip horizontal",
                 lambda *_: self.view._flip(True)),
                ("object-flip-vertical-symbolic", "Flip vertical",
                 lambda *_: self.view._flip(False)),
            ]
        buttons.append(("zoom-fit-best-symbolic", "Reset view", self._reset))
        for icon, tip, cb in buttons:
            b = Gtk.Button(icon_name=icon)
            b.set_tooltip_text(tip)
            b.add_css_class("flat")
            b.connect("clicked", cb)
            tools.append(b)
        header.set_end_widget(tools)
        self._tools = tools
        self.box.append(header)

        self.view = ImageView(on_change=self._changed)
        self.view.on_paint = self._update_zoom

        frame = Gtk.Frame()
        frame.add_css_class("view-frame")
        frame.set_child(self.view)
        frame.set_vexpand(True)

        overlay = Gtk.Overlay()
        overlay.set_child(frame)

        self.zoom_badge = Gtk.Label(label="")
        self.zoom_badge.add_css_class("zoom-badge")
        self.zoom_badge.set_halign(Gtk.Align.END)
        self.zoom_badge.set_valign(Gtk.Align.END)
        self.zoom_badge.set_can_target(False)
        self.zoom_badge.set_visible(False)
        overlay.add_overlay(self.zoom_badge)

        self.placeholder = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.placeholder.set_halign(Gtk.Align.CENTER)
        self.placeholder.set_valign(Gtk.Align.CENTER)
        self.placeholder.set_can_target(False)
        ph_icon = Gtk.Image.new_from_icon_name(empty_icon)
        ph_icon.set_pixel_size(56)
        ph_icon.add_css_class("dim-label")
        ph_title = Gtk.Label(label=empty_title)
        ph_title.add_css_class("empty-title")
        ph_hint = Gtk.Label(label=empty_hint)
        ph_hint.add_css_class("empty-hint")
        self.placeholder.append(ph_icon)
        self.placeholder.append(ph_title)
        self.placeholder.append(ph_hint)
        overlay.add_overlay(self.placeholder)

        # floating badge that trails the cursor over stacked objects (Segment)
        self.cursor_badge = Gtk.Label(label="")
        self.cursor_badge.add_css_class("cursor-badge")
        self.cursor_badge.set_halign(Gtk.Align.START)
        self.cursor_badge.set_valign(Gtk.Align.START)
        self.cursor_badge.set_can_target(False)
        self.cursor_badge.set_visible(False)
        overlay.add_overlay(self.cursor_badge)

        self.box.append(overlay)

    def show_cursor_badge(self, text, wx, wy):
        self.cursor_badge.set_text(text)
        self.cursor_badge.set_margin_start(max(4, int(wx) + 14))
        self.cursor_badge.set_margin_top(max(4, int(wy) + 16))
        self.cursor_badge.set_visible(True)

    def hide_cursor_badge(self):
        self.cursor_badge.set_visible(False)

    def add_tool(self, widget):
        """Put `widget` first among the header's tools."""
        self._tools.prepend(widget)

    def _reset(self, *_):
        self.view.reset_view()

    def _changed(self):
        has = self.view.has_image()

        def apply():
            self.placeholder.set_visible(not has)
            self.zoom_badge.set_visible(has)
            return False
        # Still deferred past the current snapshot, but not starved by an
        # animation like the worker replies above.
        GLib.idle_add(apply, priority=GLib.PRIORITY_DEFAULT)
        if self._ext_change:
            self._ext_change()

    def _update_zoom(self, percent):
        # Called from the paint callback; defer the label mutation to idle so we
        # never relayout mid-snapshot.
        if percent != self._last_pct:
            self._last_pct = percent
            GLib.idle_add(self.zoom_badge.set_text, "%d%%" % percent,
                          priority=GLib.PRIORITY_DEFAULT)


class Window(Adw.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app)
        self.set_title("bg-be-gone")
        self.set_default_size(1300, 820)

        self.tmpdir = os.environ.get("BGBG_TMPDIR") or tempfile.mkdtemp(
            prefix="bg-be-gone-")
        os.makedirs(self.tmpdir, exist_ok=True)
        self.source_path = None
        self._source_animation = formats.Animation.STILL
        self._pasted = 0               # images pasted so far, to name their files
        self._open_gen = 0             # bumped by every open, drop or paste (_new_open)
        self.result_output = None
        # Whether the shown result's pixels are transparent, which decides if
        # JPEG is offered. Recorded per request (_InFlight.alpha) and adopted
        # with the result, never read from the settings at save time.
        self._result_alpha = formats.Alpha.TRANSPARENT
        # Every request to the worker until its last reply: id -> _InFlight.
        # A request given up on moves to _abandoned, where its late replies are
        # dropped; its temporary input is still deleted once it ends.
        self._jobs = {}
        self._abandoned = {}
        self._preparing = {}           # Prep -> token of the input encode under way
        self._inputs_made = 0          # encoded inputs so far, to name each one's file
        self._animations_made = 0      # animation results so far, to name each one's file
        self._animations = set()       # animation results in the temp folder
        self._kept = None              # what the shown result was made from (_Kept)
        self._kept_made = 0            # kept folders so far, to name each one
        self._live_results = set()     # still results re-composited, deleted when replaced
        self._recomposite_timer = 0    # the debounce before a background change applies
        self._live_made = 0            # re-composited stills so far, to name each one's file
        self._animation_saves = {}     # animation result -> saves still reading it
        self._encoded_inputs = set()   # paths _prepare_input wrote, the only ones it deletes
        self._saves_writing = 0        # saves being rendered off the UI thread
        self._quit = Quit.NOT_ASKED
        self._quit_timer = 0           # the wait for a stopping batch, while one runs
        self.busy = False
        self.batch_input = None
        self.batch_output = None
        self._gpu_notified = False
        # segmentation state
        self._seg_available = False
        self._seg_models = []          # [{"rung","label"}] from the worker
        self.seg_loaded_for = None     # (source path, model asked for) the worker holds
        self.seg_pending = None        # callback to run once seg_ready arrives
        self._seg_in_use = None        # the seg_ready that loaded Segment's model, or None
        # The engine that runs BiRefNet, from the worker's "ready": the model
        # rows show its download size and VRAM.
        self._bg_engine = selection.Engine.ORT
        self._engine_used = {}         # model -> the engine its session was made on
        self.seg_busy = False
        self.seg_objects = []
        self.seg_points = []           # accumulated [x, y, label] in point mode
        self.seg_point_mask = None     # last point-mode mask path
        self._seg_point_contour = None  # its outline, to show it again
        self._seg_maps = None          # (label, general, depth) map paths of seg_objects
        self._seg_shown_mode = "everything"
        self._seg_stash = {}           # mode -> what it found, while the other mode shows
        self._seg_input = None         # the file Segment's masks were made from
        self._seg_layers_loading = None  # token of the objects being decoded, if any
        self.seg_result_output = None
        self._seg_rows = {}            # object id -> layer-list row widgets
        self._settings_path = None     # the worker's settings.json, from its "ready"
        self._engine_switch = None     # an Engine label waiting for a new worker, or None
        self._start_worker()
        self._worker_state = WorkerState.UP if self.worker.ok else WorkerState.DOWN

        self.toasts = Adw.ToastOverlay()
        self.set_content(self.toasts)
        tv = Adw.ToolbarView()
        self.toasts.set_child(tv)

        header = Adw.HeaderBar()
        self.stack = Adw.ViewStack()
        switcher = Adw.ViewSwitcher(stack=self.stack,
                                    policy=Adw.ViewSwitcherPolicy.WIDE)
        header.set_title_widget(switcher)
        open_btn = Gtk.Button(icon_name="document-open-symbolic")
        open_btn.set_tooltip_text("Open image")
        open_btn.connect("clicked", self._on_open_image)
        header.pack_start(open_btn)
        help_btn = Gtk.Button(icon_name="help-about-symbolic")
        help_btn.set_tooltip_text("How to use bg-be-gone")
        help_btn.connect("clicked", lambda *_: self._on_help())
        header.pack_end(help_btn)
        menu = Gio.Menu()
        menu.append("How to use…", "app.help")
        menu.append("Open log", "win.open-log")
        menu.append("About bg-be-gone", "app.about")
        open_log = Gio.SimpleAction.new("open-log", None)
        open_log.connect("activate", self._on_open_log)
        self.add_action(open_log)
        menu_btn = Gtk.MenuButton(icon_name="open-menu-symbolic",
                                  menu_model=menu)
        header.pack_end(menu_btn)
        tv.add_top_bar(header)
        # Stays up while there is no worker, since nothing would answer a request.
        self.worker_banner = Adw.Banner()
        self.worker_banner.set_button_label("Restart")
        self.worker_banner.connect("button-clicked", lambda *_: self._restart_worker())
        tv.add_top_bar(self.worker_banner)

        body = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        # libadwaita's sidebar surface sets it off from the pages, with no line.
        sidebar = self._build_sidebar()
        sidebar.add_css_class("sidebar-pane")
        # As wide as its rows want, and no wider: otherwise it takes a share of
        # any spare width (659 px of a maximized window), which a page needs.
        sidebar.set_propagate_natural_width(True)
        sidebar.set_hexpand(False)
        body.append(sidebar)
        self.stack.set_hexpand(True)
        body.append(self.stack)
        tv.set_content(body)
        tv.add_bottom_bar(self._build_statusbar())
        tv.set_bottom_bar_style(Adw.ToolbarStyle.RAISED_BORDER)

        self.single_page = self.stack.add_titled_with_icon(
            self._build_single(), "single", "Single", "image-x-generic-symbolic")
        self.batch_page = self.stack.add_titled_with_icon(
            self._build_batch(), "batch", "Batch", "view-grid-symbolic")
        self.seg_page = self.stack.add_titled_with_icon(
            self._build_segment(), "segment", "Segment", "edit-select-all-symbolic")
        self.stack.connect("notify::visible-child-name", self._on_page_changed)
        # mirror each page's status label into the shared footer
        self._mirror_status(self.status, "single")
        self._mirror_status(self.seg_status, "segment")
        self._mirror_status(self.batch_status, "batch")
        if os.environ.get("BGBG_START_PAGE"):
            self.stack.set_visible_child_name(os.environ["BGBG_START_PAGE"])
        self._sync_sidebar_page()
        self._update_footer()
        self._cur_page = self.stack.get_visible_child_name()

        # window-wide drag and drop
        drop = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop.connect("drop", self._on_drop)
        self.add_controller(drop)

        # Keyboard shortcuts; the help dialog lists the same ones. A focused text
        # field gets Ctrl+V first, so pasting into the rename pattern still works.
        sc = Gtk.ShortcutController()
        sc.set_scope(Gtk.ShortcutScope.MANAGED)
        for trigger, action in (
                ("<Control>o", lambda: self._on_open_image()),
                ("<Control>v", lambda: self._on_paste()),
                ("<Control>s", self._accel_save),
                ("<Control>Return", self._accel_primary),
                ("Escape", self._accel_cancel),
                ("<Control>q", lambda: self.close()),
                ("<Control>w", lambda: self.close())):
            # An action that returns False hands the key on to whatever else wants it.
            sc.add_shortcut(Gtk.Shortcut.new(
                Gtk.ShortcutTrigger.parse_string(trigger),
                Gtk.CallbackAction.new(lambda *_, act=action: act() is not False)))
        self.add_controller(sc)

        self.connect("close-request", self._on_close)
        self._restore_settings()
        if not self.worker.ok:
            self._show_worker_down("Could not start the background process (%s)."
                                   % self.worker.error)

    # ---------- remembered settings ----------
    def _restore_settings(self):
        """Put back last launch's choices, then save each one as it changes.

        The change handlers connect only after the restore, so restoring
        writes nothing.
        """
        self._settings_error = None    # the first failed save, shown once
        try:
            self._settings = settings.load()
        except settings.SettingsError as e:
            _log.error("reading the settings failed", exc_info=e)
            self._settings = {}
        # Every value is checked before use: the file is hand-editable, and a bad
        # one falls back to its default instead of keeping the window from opening.
        s = self._settings
        width, height = _window_dimension(s.get("window_width")), \
            _window_dimension(s.get("window_height"))
        if width and height:
            self.set_default_size(width, height)
        if s.get("window_maximized") is True:
            self.maximize()
        subjects = [v for _, v in SUBJECTS]
        if s.get("subject") in subjects:
            self.subject_row.set_selected(subjects.index(s["subject"]))
        elif s.get("subject") == "custom":
            self.subject_row.set_selected(len(SUBJECTS))
        models = list(modelinfo.BG_MODELS)
        if s.get("model") in models:
            self.model_row.set_selected(models.index(s["model"]))
        backgrounds = [v for _, v in BGS]
        if s.get("background") in backgrounds:
            self.bg_row.set_selected(backgrounds.index(s["background"]))
        colour = Gdk.RGBA()
        if isinstance(s.get("custom_colour"), str) and colour.parse(s["custom_colour"]):
            self.color_btn.set_rgba(colour)
        # "blur_strength", not the old "blur": that one was a radius in pixels,
        # which means something else on this scale, so it's left unread.
        blur = s.get("blur_strength")
        if isinstance(blur, (int, float)) and not isinstance(blur, bool) and math.isfinite(blur):
            self.blur_row.set_value(blur)          # the row clamps it to its range
        for key, row in (("batch_input", self.in_row), ("batch_output", self.out_row)):
            folder = _existing_folder(s.get(key))
            if folder:
                setattr(self, key, folder)
                row.set_subtitle(folder)
        self._sync_batch()

        self.subject_row.connect("notify::selected", lambda *_: self._remember(
            subject=("custom" if self.subject_row.get_selected() >= len(SUBJECTS)
                     else SUBJECTS[self.subject_row.get_selected()][1])))
        self.model_row.connect("notify::selected", lambda *_: self._remember(
            model=self.model_row.get_selected_item().get_string()))
        self.bg_row.connect("notify::selected", lambda *_: self._remember(
            background=BGS[self.bg_row.get_selected()][1]))
        self.color_btn.connect("notify::rgba", lambda *_: self._remember(
            custom_colour=self.color_btn.get_rgba().to_string()))
        self.blur_row.connect("notify::value", lambda *_: self._remember(
            blur_strength=int(self.blur_row.get_value())))

    def _remember(self, **values):
        """Save the `values` that changed; a failure is logged once and shown once."""
        changed = {k: v for k, v in values.items() if self._settings.get(k) != v}
        if not changed:
            return
        self._settings.update(changed)
        try:
            settings.update(**changed)
        except settings.SettingsError as e:
            if self._settings_error is None:
                self._settings_error = e
                self._report_failure("saving the settings", e,
                                     "Could not save your settings: %s." % e.reason)

    def _remembered_folder(self, key):
        """A remembered folder as a Gio.File, if it still exists."""
        folder = _existing_folder(self._settings.get(key))
        return Gio.File.new_for_path(folder) if folder else None

    def _source_folder(self):
        """The folder of the opened image, unless it's one of ours (a paste)."""
        if not self.source_path or self._in_tmpdir(self.source_path):
            return None
        return Gio.File.new_for_path(os.path.dirname(os.path.abspath(self.source_path)))

    # ---------- sidebar ----------
    def _build_sidebar(self):
        scroller = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER)
        # The Output rows used to widen it; without them, it keeps room for the
        # Subject row's value and its cost.
        scroller.set_size_request(360, -1)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        box.set_margin_top(16)
        box.set_margin_bottom(16)
        box.set_margin_start(12)
        box.set_margin_end(12)
        scroller.set_child(box)

        # One coherent picker: friendly Subject presets, plus a "Custom model…"
        # entry that reveals the exact-model list (same pattern as the Background
        # "Custom…" colour reveal). No hidden mode flag.
        model_grp = Adw.PreferencesGroup(title="Model")
        self.subject_row, self._refresh_subject_entries = model_picker(
            "Subject", [(None, [s[1] for s in SUBJECTS] + [CUSTOM_SUBJECT])],
            self._describe_subject)
        self.subject_row.connect("notify::selected", self._on_subject_changed)
        model_grp.add(self.subject_row)

        self.model_row, self._refresh_model_entries = model_picker(
            "Model",
            [(family.value, keys) for family, keys in modelinfo.by_family(modelinfo.BG_MODELS)],
            self._describe_bg_model)
        self.model_row.set_visible(False)
        self.model_row.connect("notify::selected", self._on_bg_model_changed)
        model_grp.add(self.model_row)
        self._refresh_bg_subtitles()

        # It changes the cut-out itself, so it's part of what runs.
        self.alpha_row = Adw.SwitchRow(
            title="Alpha matting", subtitle="Cleaner edges, a little slower")
        model_grp.add(self.alpha_row)

        # Shown only when the worker says it can run PyTorch, so an install
        # without it looks as it always did.
        self.model_adv = Adw.ExpanderRow(title="Advanced")
        self.engine_row = Adw.ComboRow(
            title="Engine", model=Gtk.StringList.new([e[0] for e in ENGINES]))
        self.engine_row.set_subtitle("What runs BiRefNet")
        self._engine_handler = self.engine_row.connect("notify::selected", self._on_engine_changed)
        self._engine_saved = self.engine_row.get_selected()   # the row's last saved choice
        self.model_adv.add_row(self.engine_row)
        self.model_adv.set_visible(False)
        model_grp.add(self.model_adv)
        self.model_grp = model_grp
        box.append(model_grp)
        box.append(self._build_seg_group())

        self._build_output_rows()
        return scroller

    def _build_output_rows(self):
        """Background, Custom colour and Blur strength, in the list the Result
        headers' Output popovers show (_output_button)."""
        self.output_list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        self.output_list.add_css_class("boxed-list")
        self.output_list.set_size_request(320, -1)
        self.bg_row = Adw.ComboRow(
            title="Background",
            model=Gtk.StringList.new([b[0] for b in BGS]))
        self.bg_row.connect("notify::selected", lambda *_: self._sync_color())
        self.output_list.append(self.bg_row)

        self.color_row = Adw.ActionRow(title="Custom colour")
        self.color_btn = Gtk.ColorDialogButton.new(Gtk.ColorDialog())
        rgba = Gdk.RGBA()
        rgba.parse("#00b140")
        self.color_btn.set_rgba(rgba)
        self.color_btn.connect("notify::rgba", self._on_output_changed)
        self.color_btn.set_valign(Gtk.Align.CENTER)
        self.color_row.add_suffix(self.color_btn)
        self.color_row.set_sensitive(False)
        self.output_list.append(self.color_row)

        # A share of the picture's size (background.py), so it looks the same on any photo.
        self.blur_row = Adw.SpinRow(
            title="Blur strength", subtitle="The same look on any photo size",
            adjustment=Gtk.Adjustment(
                value=background.STRENGTH_DEFAULT, lower=background.STRENGTH_MIN,
                upper=background.STRENGTH_MAX, step_increment=1, page_increment=5))
        self.blur_row.set_visible(False)
        self.blur_row.connect("notify::value", self._on_output_changed)
        self.output_list.append(self.blur_row)
        self._output_buttons = []
        self._output_owner = None       # the popover holding the rows, if any

    def _output_button(self):
        """The Output button for a Result header: the current background, as
        text, and a popover with its settings, shared by every page."""
        content = Adw.ButtonContent(icon_name="preferences-color-symbolic")
        button = Gtk.MenuButton(child=content)
        button.add_css_class("flat")
        popover = Gtk.Popover()
        button.set_popover(popover)

        def show_rows(*_):
            # One set of rows: they move into whichever popover opens. Their
            # parent is the popover's own contents widget, so the popover
            # that holds them is kept here to hand them over.
            owner = self._output_owner
            if owner is not popover:
                if owner is not None:
                    owner.set_child(None)
                popover.set_child(self.output_list)
                self._output_owner = popover
        popover.connect("show", show_rows)
        self._output_buttons.append((button, content))
        self._sync_output_buttons()
        return button

    def _sync_output_buttons(self):
        """Name the background on every Output button: in its label, tooltip
        and accessible name, so it reads without the colour swatch."""
        name = BG_SHORT_NAMES[BGS[self.bg_row.get_selected()][1]]
        for button, content in getattr(self, "_output_buttons", []):
            content.set_label(name)
            button.set_tooltip_text("Background: %s" % name)
            button.update_property([Gtk.AccessibleProperty.LABEL],
                                   ["Output background: %s" % name])

    def _build_statusbar(self):
        """Persistent window footer: device (left), the active page's status/hint
        (centre), and page context (right). Consolidates what used to be a
        sidebar device row plus scattered per-page labels."""
        bar = Gtk.CenterBox()
        bar.add_css_class("statusbar")
        bar.set_margin_start(12)
        bar.set_margin_end(12)
        bar.set_margin_top(4)
        bar.set_margin_bottom(4)

        dev = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        dev.set_valign(Gtk.Align.CENTER)
        self.device_dot = Gtk.Box()
        self.device_dot.add_css_class("dev-dot")
        self.device_dot.set_valign(Gtk.Align.CENTER)
        self.device_lbl = Gtk.Label(xalign=0, label="Detecting device…")
        self.device_lbl.add_css_class("dim-label")
        dev.append(self.device_dot)
        dev.append(self.device_lbl)
        bar.set_start_widget(dev)

        centre = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        centre.set_valign(Gtk.Align.CENTER)
        self.footer_status = Gtk.Label(label="")
        self.footer_status.add_css_class("status-label")
        self.footer_status.set_ellipsize(3)          # PANGO_ELLIPSIZE_END
        centre.append(self.footer_status)
        # Shared progress bar: long, frame-wise or step-wise work (GIF removal,
        # "segment everything") reports here so it's always visible in the footer.
        self.footer_progress = Gtk.ProgressBar()
        self.footer_progress.add_css_class("footer-progress")
        self.footer_progress.set_valign(Gtk.Align.CENTER)
        self.footer_progress.set_size_request(150, -1)
        self.footer_progress.set_visible(False)
        self._footer_motion = ProgressMotion(self.footer_progress)
        centre.append(self.footer_progress)
        # The bar's count ("12 / 30") sits beside it, not drawn above it, so
        # the bar's middle lines up with the text's at any font size. It
        # shows the bar's own text, and comes and goes (and fades) with it.
        count = Gtk.Label(valign=Gtk.Align.CENTER)
        count.add_css_class("status-label")
        count.add_css_class("numeric")
        for prop in ("text", "visible", "opacity"):
            self.footer_progress.bind_property(
                prop, count, "label" if prop == "text" else prop,
                GObject.BindingFlags.SYNC_CREATE)
        centre.append(count)
        bar.set_center_widget(centre)

        self.footer_context = Gtk.Label(label="")
        self.footer_context.add_css_class("dim-label")
        self.footer_context.add_css_class("status-label")
        bar.set_end_widget(self.footer_context)
        return bar

    def _footer_progress(self, done, total):
        """Show frame/step progress in the shared footer bar (GIF + segmentation)."""
        total = max(int(total), 1)
        self._footer_motion.move_to(min(1.0, done / total))
        self.footer_progress.set_text("%d / %d" % (done, total))

    @staticmethod
    def _download_text(msg):
        """A download reply as text, like "312 of 973 MB (32%)"; only the MB
        when the server gave no size."""
        mb = 1024 * 1024
        if not msg.get("total"):
            return "%d MB" % (msg["done"] // mb)
        return "%d of %d MB (%d%%)" % (msg["done"] // mb, msg["total"] // mb,
                                       100 * msg["done"] // msg["total"])

    def _footer_download(self, msg):
        mb = 1024 * 1024
        if msg.get("total"):
            self._footer_progress(msg["done"] // mb, msg["total"] // mb)
            self.footer_progress.set_text("%d / %d MB" % (msg["done"] // mb, msg["total"] // mb))
        if msg.get("total") and msg["done"] >= msg["total"]:
            self._footer_loading()             # loading the model comes next

    def _footer_loading(self):
        """Pulse the footer bar while a model loads: there's no fraction to show."""
        self.footer_progress.set_text("")
        self._footer_motion.unknown()

    def _hide_footer_progress(self):
        if getattr(self, "footer_progress", None) is not None:
            self._footer_motion.fade_out()

    def _mirror_status(self, label, page):
        """Reflect a per-page status label into the footer while its page is
        active — no call-site changes, just watch the label's text property."""
        def cb(*_):
            if self.stack.get_visible_child_name() == page:
                self.footer_status.set_text(label.get_text())
        label.connect("notify::label", cb)

    def _update_footer(self):
        page = self.stack.get_visible_child_name()
        cur = {"single": getattr(self, "status", None),
               "segment": getattr(self, "seg_status", None),
               "batch": getattr(self, "batch_status", None)}.get(page)
        if cur is not None:
            self.footer_status.set_text(cur.get_text())
        try:
            self.footer_context.set_text(self._footer_context())
        except Exception:
            self.footer_context.set_text("")

    def _footer_context(self):
        page = self.stack.get_visible_child_name()
        if page == "segment":
            # The model actually loaded, once there is one: Auto, or a step-down
            # after a failure, may differ from the row's choice.
            model = self._seg_model_name()
            mode = "Click" if self._seg_mode() == "point" else "Everything"
            return "Segment · %s · %s" % (mode, model) if model else "Segment · %s" % mode
        if page == "batch":
            return "Batch"
        i = self.subject_row.get_selected()
        name = (modelinfo.BG_MODELS[self._bg_model_key()].short if i >= len(SUBJECTS)
                else SUBJECTS[i][0])
        return "Single · %s" % name

    def _set_device(self, provider, label):
        self.device_lbl.set_text(label)
        cls = {"CUDAExecutionProvider": "dev-nvidia",
               "ROCMExecutionProvider": "dev-amd",
               "MIGraphXExecutionProvider": "dev-amd",
               "DmlExecutionProvider": "dev-dml"}.get(provider, "dev-cpu")
        for c in ("dev-nvidia", "dev-amd", "dev-dml", "dev-cpu"):
            self.device_dot.remove_css_class(c)
        self.device_dot.add_css_class(cls)
        self.device_dot.set_tooltip_text(label)

    def _start_worker(self, first_id=1):
        """Start this window's worker; replies from one it replaced are dropped.

        A replacement starts its ids at `first_id`, the old worker's next one,
        so no request id is ever used twice.
        """
        def deliver(msg):
            if self.worker is worker:
                self._on_worker_message(msg)

        worker = Worker(deliver, first_id=first_id)
        self.worker = worker
        self._forget_seg_model()           # a new worker holds no Segment model

    def _offer_engines(self, on_offer, settings_path):
        """Show Model > Advanced > Engine when the worker can run PyTorch,
        set to what the worker's own settings file says."""
        self._settings_path = settings_path
        shown = "torch" in on_offer and bool(settings_path)
        if shown:
            current = selection.read_setting(settings_path)
            index = next(i for i, e in enumerate(ENGINES) if e[1] is current)
            with self.engine_row.handler_block(self._engine_handler):
                self.engine_row.set_selected(index)
        self._engine_saved = self.engine_row.get_selected()
        self.model_adv.set_visible(shown)

    def _on_engine_changed(self, *_):
        if self.engine_row.get_selected() == self._engine_saved:
            return          # nothing new to save: e.g. the row put back below
        label, engine = ENGINES[self.engine_row.get_selected()]
        try:
            selection.save_setting(engine, self._settings_path)
        except settings.SettingsError as err:
            self._toast("Couldn't save the engine setting: %s" % err.reason)
            # Back to what is saved and running: the row mustn't claim a choice
            # that neither the file nor the worker has.
            self.engine_row.set_selected(self._engine_saved)
            return
        self._engine_saved = self.engine_row.get_selected()
        # A new worker reads the setting at start-up, so nothing of the old
        # engine (a PyTorch start-up in flight, its GPU context) carries over.
        self._engine_switch = label
        if self._job_in_flight():
            self._toast("Engine: %s, once the current job is done." % label)
        self._replace_worker_when_idle()
        self._mark_stale()

    def _job_in_flight(self):
        """Whether a request is out to the worker, or an input is being made for one."""
        return bool(self._jobs or self._preparing)

    def _replace_worker_when_idle(self):
        """Swap in a new worker for an Engine change once no job is in flight.

        Called again whenever a job ends, so the active one finishes first.
        """
        if self._engine_switch is None:
            return
        if self._job_in_flight():
            return
        if self._worker_state is WorkerState.DOWN:
            return          # Restart starts the new worker, and that applies the change
        label, self._engine_switch = self._engine_switch, None
        old = self.worker
        self.seg_loaded_for = None     # the new worker holds no encoded image
        # The old worker's abandoned requests will never answer now.
        for req in self._abandoned.values():
            self._remove_temp_input(req.input)
        self._abandoned.clear()
        self._start_worker(first_id=old.next_id)
        threading.Thread(target=old.shutdown, daemon=True).start()
        self._toast("Engine: %s." % label)

    def _on_subject_changed(self, *_):
        # The exact-model picker only appears for the "Custom model…" entry.
        custom = self.subject_row.get_selected() >= len(SUBJECTS)
        self.model_row.set_visible(custom)
        self._on_bg_model_changed()

    def _on_bg_model_changed(self, *_):
        self._refresh_bg_subtitles()
        self._mark_stale()
        if getattr(self, "footer_context", None) is not None:
            self._update_footer()

    def _bg_model_key(self):
        """The background-removal model the Model group picks."""
        i = self.subject_row.get_selected()
        if i >= len(SUBJECTS):                      # "Custom model…"
            return self.model_row.get_selected_item().get_string()
        return SUBJECTS[i][1]

    def _describe_subject(self, key):
        if key == CUSTOM_SUBJECT:
            return "Custom model…", "Custom model…", "any model, by name"
        label = next(s[0] for s in SUBJECTS if s[1] == key)
        info = modelinfo.BG_MODELS[key]
        return label, label, modelinfo.row_subtitle(Pick.PRESET, info, self._engine_of(key))

    def _describe_bg_model(self, key):
        info = modelinfo.BG_MODELS[key]
        return info.short, info.name, modelinfo.entry_subtitle(info, self._engine_of(key))

    def _engine_of(self, key):
        """The engine that runs model `key`: the one its session was made on,
        else the one the worker chose at start-up."""
        return self._engine_used.get(key, self._bg_engine)

    def _set_bg_engine(self, engine):
        """A worker is ready: BiRefNet runs on `engine`, and no model has a session yet."""
        self._bg_engine = engine
        self._engine_used.clear()
        self._refresh_bg_costs()

    def _note_engine_used(self, model, engine):
        """A session for `model` was made on `engine`, which can differ from the
        start-up choice: PyTorch failed for it, say, and ONNX Runtime took over."""
        if self._engine_used.get(model) is engine:
            return
        self._engine_used[model] = engine
        self._refresh_bg_costs()

    def _refresh_bg_costs(self):
        """Show every model's cost on the engine that runs it."""
        self._refresh_subject_entries()
        self._refresh_model_entries()
        self._refresh_bg_subtitles()

    def _refresh_bg_subtitles(self):
        """Name the model in use and what it costs, under the Model group's rows."""
        info = modelinfo.BG_MODELS[self._bg_model_key()]
        if self.subject_row.get_selected() >= len(SUBJECTS):
            self.subject_row.set_subtitle("The model chosen below")
            self.model_row.set_subtitle(
                modelinfo.row_subtitle(Pick.MANUAL, info, self._engine_of(info.key)))
        else:
            self.subject_row.set_subtitle(
                modelinfo.row_subtitle(Pick.PRESET, info, self._engine_of(info.key)))

    def _sync_color(self):
        bg = BGS[self.bg_row.get_selected()][1]
        self.color_row.set_sensitive(bg == "custom")
        self.blur_row.set_visible(bg == "blur")
        self._sync_output_buttons()
        self._on_output_changed()

    def _on_output_changed(self, *_):
        """A background setting changed: the shown result follows, live."""
        self._seg_bg_changed()
        # A change still running is for settings that no longer hold, even
        # when this one goes back to what's shown: drop it now, not at the send.
        if self._abandon_jobs({Job.RECOMPOSITE}) is Abandoned.SOME:
            self._recomposite_over()
        if self._kept is None:
            self._mark_stale()           # a result made before it could be kept
            return
        self._schedule_recomposite()

    def _schedule_recomposite(self):
        """Apply the background settings to the shown result, once they rest."""
        if getattr(self, "_kept", None) is None:    # also before the window is built
            return
        if self._recomposite_timer:
            GLib.source_remove(self._recomposite_timer)
        # Debounced: dragging the strength or a colour sends one job, not dozens.
        self._recomposite_timer = GLib.timeout_add(_RECOMPOSITE_DELAY_MS, self._recomposite)

    def _recomposite(self):
        """Put the shown result onto the current background, without the model.

        One job at a time, the latest wins: one still out is abandoned, and
        its reply dropped. Waits while a Generate or a batch runs: _set_busy applies it after.
        """
        self._recomposite_timer = 0
        kept = self._kept
        _model, bg, _alpha, blur = self.get_settings()
        if kept is None or self.busy or (bg, blur) == (kept.bg, kept.blur):
            return False
        self._abandon_jobs({Job.RECOMPOSITE})
        req = {"op": "recomposite", "keep_dir": kept.folder, "bg": bg, "blur": blur}
        if kept.animation is None:
            self._live_made += 1
            req["output"] = os.path.join(self.tmpdir, "live-%d.png" % self._live_made)
        else:
            self._animations_made += 1
            req["output"] = os.path.join(
                self.tmpdir, "result-%d%s" % (self._animations_made, kept.animation["suffix"]))
            req.update({k: kept.animation[k] for k in ("input", "rot", "fh", "fv")})
            self.gen_cancel_btn.set_sensitive(True)
            self.status.set_text("Changing the background…")
        self._send_job(Job.RECOMPOSITE, req, alpha=_alpha_of(bg),
                       kept=_Kept(kept.folder, bg, blur, kept.animation))
        return False

    def _on_recomposited(self, req, msg):
        """Show a re-composited result; Save and its JPEG offer follow it."""
        self._recomposite_over()
        output = msg["output"]
        if self._kept is None or req.kept.folder != self._kept.folder:
            _remove_quietly(output)       # for a result that's gone since
            return
        if msg["frames"] > 1:
            self._animations.add(output)
            if not self.res_panel.view.load_file(output):
                self.res_panel.view.load_file(msg["preview"])
        else:
            self._live_results.add(output)
        self._replace_result(output)
        self._result_alpha = req.alpha
        self._kept = req.kept
        self.status.set_text("Background: %s." % BG_SHORT_NAMES.get(req.kept.bg, "Custom"))
        if msg["frames"] > 1:
            self.save_btn.set_sensitive(True)
            return
        self.save_btn.set_sensitive(False)

        def decoded(loaded):
            if loaded is Loaded.SHOWN and self.result_output == output \
                    and not self._has_job(Job.OPEN):
                self.save_btn.set_sensitive(True)
        self.res_panel.view.load_file_async(output, done=decoded)

    def _recomposite_over(self):
        self.gen_cancel_btn.set_sensitive(self.busy and self._has_job(Job.SINGLE, Job.BATCH))
        self._hide_footer_progress()

    def _stop_recomposite(self):
        """Drop a pending or running background change: a new result replaces it."""
        if self._recomposite_timer:
            GLib.source_remove(self._recomposite_timer)
            self._recomposite_timer = 0
        if self._abandon_jobs({Job.RECOMPOSITE}) is Abandoned.SOME:
            self._recomposite_over()

    def _new_kept(self, bg, blur, animation=None):
        """A new folder for what a Generate keeps; the result owns it once shown."""
        self._kept_made += 1
        folder = os.path.join(self.tmpdir, "kept-%d" % self._kept_made)
        return _Kept(folder, bg, blur, animation)

    def _adopt_kept(self, kept):
        """The shown result was made from `kept`: the one before goes."""
        if kept is None:
            return
        if self._kept is not None and self._kept.folder != kept.folder:
            shutil.rmtree(self._kept.folder, ignore_errors=True)
        self._kept = kept
        self._schedule_recomposite()     # a change made while it generated applies now

    def _forget_kept(self):
        """The result is gone: so is what it was made from."""
        if self._kept is not None:
            shutil.rmtree(self._kept.folder, ignore_errors=True)
        self._kept = None

    def _drop_kept(self, kept):
        """Delete a request's kept folder, unless the shown result owns it."""
        if kept is None or (self._kept is not None and kept.folder == self._kept.folder):
            return
        shutil.rmtree(kept.folder, ignore_errors=True)

    def _seg_bg_changed(self, *_):
        if getattr(self, "seg_panel", None) is None:
            return
        if self.stack.get_visible_child_name() == "segment":
            self._update_seg_preview()
        elif self.seg_result_output is not None:
            # Changed from another page: redrawn on the way back (_on_page_changed).
            self._seg_preview_stale = True

    def get_settings(self):
        model = self._bg_model_key()
        bg = BGS[self.bg_row.get_selected()][1]
        if bg == "custom":
            c = self.color_btn.get_rgba()
            bg = "#%02x%02x%02x" % (round(c.red * 255), round(c.green * 255),
                                    round(c.blue * 255))
        return model, bg, self.alpha_row.get_active(), int(self.blur_row.get_value())

    # ---------- segmentation sidebar + page ----------
    def _build_seg_group(self):
        grp = Adw.PreferencesGroup(title="Segmentation")
        self.seg_grp = grp
        # Mode lives on the page as a segmented control (see _build_segment).

        # Focus speed (visible): how long to hover before drilling to the part.
        self.seg_focus_row = Adw.ComboRow(
            title="Focus speed",
            model=Gtk.StringList.new([f[0] for f in SEG_FOCUS]))
        self.seg_focus_row.set_subtitle("Hover this long to focus a sub-object")
        self.seg_focus_row.set_selected(1)     # Normal
        self.seg_focus_row.connect("notify::selected", self._on_seg_focus_changed)
        grp.add(self.seg_focus_row)

        # Advanced (collapsed by default): model override + detail.
        # (Device is shown once, in the sidebar footer — not duplicated here.)
        self.seg_adv = Adw.ExpanderRow(title="Advanced",
                                       subtitle="Model and detail")
        self.seg_model_row, _ = model_picker("Model", [(None, [SEG_AUTO])],
                                             self._describe_seg_model)
        self.seg_model_row.connect("notify::selected", self._on_seg_model_changed)
        self.seg_adv.add_row(self.seg_model_row)

        self.seg_detail_row = Adw.ComboRow(
            title="Detail", model=Gtk.StringList.new([d[0] for d in SEG_DETAIL]))
        self.seg_detail_row.set_subtitle("Finer finds more, but is slower")
        self.seg_detail_row.connect("notify::selected", lambda *_: None)
        self.seg_adv.add_row(self.seg_detail_row)
        grp.add(self.seg_adv)
        grp.set_visible(False)
        self._refresh_seg_subtitles()
        return grp

    def _apply_seg_models(self):
        rungs = [m["rung"] for m in self._seg_models]
        self._syncing_seg = True
        self.seg_model_row.set_model(Gtk.StringList.new([SEG_AUTO] + rungs))
        self.seg_model_row.set_selected(0)
        self._syncing_seg = False
        self._refresh_seg_subtitles()

    def _seg_model_choice(self):
        """The rung the row asks for, or "auto"."""
        item = self.seg_model_row.get_selected_item()
        return item.get_string() if item is not None else SEG_AUTO

    def _seg_label(self, rung):
        """A rung's name: ours, else the worker's, else the rung itself."""
        info = modelinfo.SEG_MODELS.get(rung)
        if info is not None:
            return info.name
        return next((m["label"] for m in self._seg_models if m["rung"] == rung), rung)

    def _describe_seg_model(self, key):
        if key == SEG_AUTO:
            return "Auto", "Automatic", "the best model your free VRAM fits"
        info = modelinfo.SEG_MODELS.get(key)
        if info is None:
            label = self._seg_label(key)
            return label, label, ""
        return info.short, info.name, modelinfo.entry_subtitle(info, selection.Engine.ORT)

    def _seg_model_name(self):
        """The model Segment uses, for the footer and the Advanced row."""
        loaded = self._seg_in_use
        if loaded is not None:
            fallback = " (fallback)" if loaded.get("mode") == Pick.FALLBACK.value else ""
            return self._seg_label(loaded["rung"]) + fallback
        choice = self._seg_model_choice()
        return "Auto" if choice == SEG_AUTO else self._seg_label(choice)

    def _forget_seg_model(self):
        """Segment's model was released, or its worker went: name no model as loaded."""
        self._seg_in_use = None
        if getattr(self, "seg_model_row", None) is not None:
            self._refresh_seg_subtitles()
        if getattr(self, "footer_context", None) is not None:
            self._update_footer()

    def _refresh_seg_subtitles(self):
        """Name Segment's model, the one loaded or else the one asked for, and its cost.

        On Auto before the first load, the pick isn't known yet: the worker
        chooses it from the free VRAM at load time, and seg_ready names it.
        """
        loaded = self._seg_in_use
        choice = self._seg_model_choice()
        if loaded is not None:
            pick = SEG_PICKS.get(loaded.get("mode"), Pick.MANUAL)
            text = modelinfo.row_subtitle(
                pick, modelinfo.SEG_MODELS.get(loaded["rung"]), selection.Engine.ORT,
                why=_seg_reason(loaded), name=loaded.get("model"),
                device=Device.GPU if loaded.get("gpu") else Device.CPU)
            label = self._seg_model_name() + (" (Auto)" if pick is Pick.AUTO else "")
            adv = "Model: " + modelinfo.model_summary(
                label, modelinfo.SEG_MODELS.get(loaded["rung"]), selection.Engine.ORT,
                device=Device.GPU if loaded.get("gpu") else Device.CPU)
        elif choice == SEG_AUTO:
            text = "Auto · picks by free VRAM when you segment"
            adv = "Model: Auto"
        else:
            text = modelinfo.row_subtitle(Pick.MANUAL, modelinfo.SEG_MODELS.get(choice),
                                          selection.Engine.ORT)
            adv = "Model: " + modelinfo.model_summary(
                self._seg_model_name(), modelinfo.SEG_MODELS.get(choice), selection.Engine.ORT)
        self.seg_model_row.set_subtitle(text)
        self.seg_adv.set_subtitle(adv)

    def _seg_detail(self):
        return SEG_DETAIL[self.seg_detail_row.get_selected()][1] or None

    def _seg_mode(self):
        return self.seg_mode_toggle.get_active_name() or "everything"

    @staticmethod
    def _domain(page):
        # Single and Batch share the background-removal model; Segment is its own.
        return "seg" if page == "segment" else "bg"

    def _on_page_changed(self, *_):
        self._sync_sidebar_page()
        self._update_footer()
        new = self.stack.get_visible_child_name()
        old = getattr(self, "_cur_page", None)
        self._cur_page = new
        if new == "segment" and getattr(self, "_seg_preview_stale", False):
            # Save selection waits for the preview of the settings it'll use.
            self._seg_preview_stale = False
            self.seg_save_btn.set_sensitive(False)
            self._update_seg_preview()
        # Free a domain's model when you leave it (keep it resident while you're
        # on its page, so re-runs stay fast).
        if old is not None and self._domain(old) != self._domain(new):
            self._release_domain(self._domain(old))

    def _send_job(self, job, req, **details):
        """Send a request and track it as `job` until its last reply.

        `details` are _InFlight fields kept with it. Returns the request id,
        or -1 when there is no worker to send it to.
        """
        rid = self.worker.send(req)
        if rid >= 0:
            self._jobs[rid] = _InFlight(job, **details)
        return rid

    def _has_job(self, *jobs):
        return any(req.job in jobs for req in self._jobs.values())

    def _rids(self, *jobs):
        return [rid for rid, req in self._jobs.items() if req.job in jobs]

    def _cancel_jobs(self, *jobs):
        """Ask the worker to stop `jobs` by request id: the running one stops,
        a queued one is skipped. Their replies still arrive and end them."""
        for rid in self._rids(*jobs):
            self.worker.cancel(rid)

    def _abandon_jobs(self, jobs):
        """Stop `jobs` and stop waiting for them: their late replies are dropped.

        Returns Abandoned.SOME when there were any, Abandoned.NONE otherwise.
        """
        rids = self._rids(*jobs)
        for rid in rids:
            self.worker.cancel(rid)
            self._abandoned[rid] = self._jobs.pop(rid)
        if rids and not self._jobs:
            # The last job went without a reply of its own to end it: an
            # Engine change that waited for it gets its turn here.
            GLib.idle_add(self._replace_worker_when_idle)
        return Abandoned.SOME if rids else Abandoned.NONE

    def _release_domain(self, domain):
        if domain == "bg":
            self._unload("bg")
            return
        # Stop any in-flight pass, then unload unconditionally. We can't rely on a
        # seg_canceled reply — only "segment everything" is cancellable; load /
        # point / extract finish normally — so unload authoritatively here.
        self._abandon_jobs(_SEG_JOBS)
        if self._preparing.pop(Prep.SEGMENT, None) is not None:
            self._seg_set_busy(False)    # a load still being prepared: it is never sent
        if self.seg_busy:
            self._seg_set_busy(False)
        self._unload("seg")

    def _unload(self, scope):
        """Ask the worker to release a model. For segmentation, also drop the
        cached encode and invalidate any in-flight load so a later pass
        re-encodes cleanly.

        Segment's found objects and selection stay: they are files on disk and
        cost no VRAM, and Save works from them without the model.
        """
        if scope == "seg":
            self.seg_loaded_for = None
            self.seg_pending = None
            self._abandon_jobs({Job.SEG_LOAD})  # its reply would re-mark it loaded
            self._forget_seg_model()
        if self.worker.ok:
            self.worker.send({"op": "unload", "scope": scope})

    def _sync_sidebar_page(self):
        seg = self.stack.get_visible_child_name() == "segment"
        self.model_grp.set_visible(not seg)
        self.seg_grp.set_visible(seg and self._seg_available)

    def _accel_save(self):
        p = self.stack.get_visible_child_name()
        if p == "single" and self.save_btn.get_sensitive():
            self._on_save()
        elif p == "segment" and self.seg_save_btn.get_sensitive():
            self._on_seg_save()
        return True

    def _accel_primary(self):
        p = self.stack.get_visible_child_name()
        if p == "single":
            self._on_generate()
        elif p == "segment":
            # "Click to select" hides Segment everything, so Ctrl+Return must
            # not start that minutes-long pass behind the user's back.
            if self._seg_mode() == "everything":
                self._on_seg_everything()
        elif p == "batch":
            self._on_run_batch()
        return True

    def _accel_cancel(self):
        """Esc: cancel what the current page is running. False when nothing
        runs, so the key goes on to whatever else wants it."""
        p = self.stack.get_visible_child_name()
        if p == "segment" and self.seg_busy and self.seg_cancel_btn.get_sensitive():
            self._on_seg_cancel()
            return True
        if p in ("single", "batch") and self.busy and self.gen_cancel_btn.get_sensitive():
            self._on_bg_cancel()
            return True
        return False

    def _on_paste(self):
        """Ctrl+V: open the clipboard's image, or the image file copied in a file manager."""
        clip = self.get_clipboard()
        offered = clip.get_formats()
        # The clipboard answers later: an image opened, dropped or pasted
        # meanwhile is newer, and this paste must not replace it.
        gen = self._new_open()
        if offered.contain_gtype(Gdk.FileList.__gtype__):
            def got_files(c, res):
                try:
                    files = c.read_value_finish(res).get_files()
                except GLib.Error as e:
                    self._report_failure("reading copied files", e,
                                         "Could not paste: %s." % e.message)
                    return
                if gen == self._open_gen and files and files[0].get_path():
                    self._load_source(files[0].get_path())
            clip.read_value_async(Gdk.FileList.__gtype__, GLib.PRIORITY_DEFAULT, None,
                                  got_files)
        elif offered.contain_gtype(Gdk.Texture.__gtype__):
            clip.read_texture_async(None, lambda c, res: self._on_pasted_texture(c, res, gen))
        else:
            self._toast("The clipboard has no image to paste.")

    def _new_open(self):
        """Start opening an image; return its generation, which a newer one bumps."""
        self._open_gen += 1
        return self._open_gen

    def _on_pasted_texture(self, clip, res, gen):
        try:
            texture = clip.read_texture_finish(res)
        except GLib.Error as e:
            self._report_failure("reading the clipboard image", e,
                                 "Could not paste: %s." % e.message)
            return
        if texture is None:
            self._toast("The clipboard has no image to paste.")
            return
        # A file in the temp folder, so every step that wants a path works; its
        # name is what Save suggests ("Pasted image 1_nobg.png").
        self._pasted += 1
        path = os.path.join(self.tmpdir, "Pasted image %d.png" % self._pasted)

        def encode():
            if not texture.save_to_png(path):
                raise PasteWriteError(path)
            return path

        def done(result):
            if isinstance(result, Exception):
                self._report_failure("saving the pasted image", result,
                                     "Could not paste: %s." % plain_reason(result))
            elif gen != self._open_gen:
                _remove_quietly(result)      # a newer image won; this file is ours
            else:
                self._load_source(result)
        offmain.run(encode, done)

    def _build_segment(self):
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        for m in ("top", "bottom", "start", "end"):
            getattr(page, "set_margin_" + m)(10)

        self.seg_everything_btn = primary_button(
            "Segment everything", icon="edit-select-all-symbolic")
        self.seg_everything_btn.set_tooltip_text("Find every object in the image")
        self.seg_everything_btn.connect(
            "clicked", lambda *_: self._on_seg_everything())
        self.seg_selall_btn = pill_button("Select all")
        self.seg_selall_btn.connect("clicked", lambda *_: self._on_seg_select_all())
        self.seg_clear_btn = pill_button("Clear")
        self.seg_clear_btn.connect("clicked", lambda *_: self._seg_clear_overlays())
        self.seg_cancel_btn = pill_button("Cancel")
        self.seg_cancel_btn.set_sensitive(False)
        self.seg_cancel_btn.connect("clicked", lambda *_: self._on_seg_cancel())
        self.seg_save_btn = save_button("Save selection…")
        self.seg_save_btn.set_tooltip_text("Export the picked objects (Ctrl+S)")
        self.seg_save_btn.connect("clicked", lambda *_: self._on_seg_save())
        bar, self.seg_status, self.seg_spinner = action_bar(
            primary=self.seg_everything_btn,
            secondary=(self.seg_selall_btn, self.seg_clear_btn),
            cancel=self.seg_cancel_btn, save=self.seg_save_btn,
            status_text="Open an image, then Segment.")
        self.seg_banner = Adw.Banner(
            title="Tip: hold Space to pan · click to keep objects · "
                  "in Click to select, Ctrl-click removes")
        self.seg_banner.set_button_label("Got it")
        self.seg_banner.connect(
            "button-clicked", lambda *_: self.seg_banner.set_revealed(False))
        self.seg_banner.set_revealed(True)
        page.append(self.seg_banner)
        page.append(bar)

        # Segmented Mode control (Apple-style), centered under the action bar.
        self.seg_mode_toggle = Adw.ToggleGroup()
        self.seg_mode_toggle.add(Adw.Toggle(name="everything", label="Everything"))
        self.seg_mode_toggle.add(Adw.Toggle(name="point", label="Click to select"))
        self.seg_mode_toggle.set_active_name("everything")
        self.seg_mode_toggle.set_halign(Gtk.Align.CENTER)
        self.seg_mode_toggle.connect("notify::active-name", self._on_seg_mode_changed)
        page.append(self.seg_mode_toggle)

        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL, wide_handle=True)
        paned.add_css_class("card-split")
        paned.set_vexpand(True)
        self.seg_panel = Panel(
            "Source — click objects", transforms=False,
            empty_title="Drop an image here",
            empty_hint="then Segment everything, or click to select")
        self.seg_panel.view.on_seg_click = self._on_seg_click
        self.seg_panel.view.on_seg_hover = self._on_seg_hover
        self.seg_panel.view.set_seg_mode("everything")
        self.seg_res_panel = Panel(
            "Result", empty_icon="emblem-photos-symbolic",
            empty_title="Your selection appears here",
            empty_hint="Pick objects, then Save")
        self.seg_res_panel.add_tool(self._output_button())
        paned.set_start_child(self.seg_panel.box)
        paned.set_end_child(self.seg_res_panel.box)
        paned.set_resize_start_child(True)
        paned.set_resize_end_child(True)
        paned.connect("notify::max-position", self._center_seg_paned)
        page.append(paned)
        return page

    def _center_seg_paned(self, paned, *_):
        if not getattr(self, "_seg_paned_centered", False):
            mx = paned.get_property("max-position")
            if mx > 0:
                paned.set_position(mx // 2)
                self._seg_paned_centered = True

    # ---------- segmentation actions ----------
    def _seg_set_busy(self, busy, cancel=False):
        self.seg_busy = busy
        self.seg_everything_btn.set_sensitive(
            not busy and self._worker_state is WorkerState.UP)
        self.seg_selall_btn.set_sensitive(not busy)
        self.seg_cancel_btn.set_sensitive(busy and cancel)
        (self.seg_spinner.start if busy else self.seg_spinner.stop)()
        if not busy:                        # any terminal state stops the shimmer
            self.seg_panel.view.set_scanning(False)
            self._hide_footer_progress()
            # An Engine change waits for this job; after this reply is handled.
            GLib.idle_add(self._replace_worker_when_idle)

    def _extract_bg(self):
        bg = BGS[self.bg_row.get_selected()][1]
        if bg == "custom":
            c = self.color_btn.get_rgba()
            bg = "#%02x%02x%02x" % (round(c.red * 255), round(c.green * 255),
                                    round(c.blue * 255))
        return bg, int(self.blur_row.get_value())

    def _ensure_seg_loaded(self, cb):
        if not self.source_path:
            self._toast("Open an image first.")
            return
        if self._worker_state is WorkerState.DOWN:
            return
        # The model loaded for this image, and the model asked for: a load
        # still running for another model doesn't count when it lands.
        wanted = (self.source_path, self._seg_model_choice())
        if self.seg_loaded_for == wanted:
            cb()
            return
        if self.seg_busy:
            return
        self.seg_pending = cb
        self._seg_set_busy(True, cancel=True)
        self.seg_status.set_text("Preparing segmentation model…")

        def send(inp):
            # Segment's extracts read this input again later, so it stays until
            # the next one replaces it.
            if self._seg_input != inp:
                self._remove_temp_input(self._seg_input)
            self._seg_input = inp
            self._footer_loading()
            # Tracked as a job, so a reply that lands after we've left or
            # unloaded is dropped instead of re-marking the model as resident.
            # The masks go in our temp folder: the input can be the user's photo.
            self._send_job(Job.SEG_LOAD, {
                "op": "seg_load", "input": inp, "model": wanted[1],
                "maskdir": os.path.join(self.tmpdir, "segmasks")}, loads=wanted)

        def failed(e):
            self._seg_set_busy(False)
            self._report_failure("preparing the Segment input", e,
                                 "Could not prepare the image: %s." % plain_reason(e))

        self._prepare_input(Prep.SEGMENT, self.seg_panel.view, send, failed)

    def _prepare_input(self, prep, view, send, failed):
        """Hand the worker a file of what `view` shows, then call send(path).

        The source file itself when the worker can read it, it is a still, and
        it isn't rotated or flipped; otherwise a PNG encoded on a background
        thread. Encoding a 4K photo on the UI thread froze the window for ~3 s
        before each run.

        An encode is tracked under `prep` from its start: a new image or a
        cancel drops it, and its file, and send never runs. Each encode writes
        a file of its own, so a late one can't overwrite a newer job's input.
        """
        src = self.source_path
        if (src and not view.is_transformed() and formats.is_readable(src)
                and self._source_animation is formats.Animation.STILL):
            send(src)
            return
        pixbuf = view.export_pixbuf()
        self._inputs_made += 1
        dst = os.path.join(self.tmpdir, "input-%d.png" % self._inputs_made)
        self._encoded_inputs.add(dst)
        token = object()
        self._preparing[prep] = token

        def encode():
            # Fast compression: the file is a temporary hand-off, read once.
            pixbuf.savev(dst, "png", ["compression"], ["1"])
            return dst

        def encoded(result):
            if self._preparing.get(prep) is not token:   # replaced or cancelled
                self._remove_temp_input(dst)
                return
            del self._preparing[prep]
            if isinstance(result, Exception):
                self._remove_temp_input(dst)
                failed(result)
            else:
                send(result)
        offmain.run(encode, encoded)

    def _remove_temp_input(self, path):
        """Delete an input file this window encoded; leave any other file alone.

        Only _prepare_input's own encodings: a pasted image lives in the same
        temp folder, but it is the source, and later runs read it again.
        """
        if path in self._encoded_inputs:
            self._encoded_inputs.discard(path)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as e:
                _log.warning("could not remove the temporary input %s: %s", path, e)

    def _in_tmpdir(self, path):
        """Whether `path` is one of this window's own temporary files."""
        tmp = os.path.normcase(os.path.abspath(self.tmpdir))
        return os.path.normcase(os.path.abspath(path)).startswith(tmp + os.sep)

    def _on_seg_everything(self):
        if self.seg_busy or self._has_job(Job.OPEN):
            return
        self._ensure_seg_loaded(self._do_seg_everything)

    def _do_seg_everything(self):
        # The objects already found stay up until the new ones replace them,
        # so a cancelled re-run costs nothing.
        self._seg_set_busy(True, cancel=True)
        self.seg_panel.view.set_scanning(True)
        self.seg_status.set_text("Finding objects…")
        req = {"op": "seg_everything"}
        detail = self._seg_detail()
        if detail:
            req["points_per_side"] = detail
        self._send_job(Job.SEG_RUN, req)

    def _on_seg_click(self, ix, iy, value, kind):
        if kind == "toggle":
            self._update_seg_selection_ui()
        else:  # point mode
            if self.seg_busy or self._has_job(Job.OPEN):
                return
            self.seg_points.append([ix, iy, value])
            self._ensure_seg_loaded(self._do_seg_point)

    def _do_seg_point(self):
        self._seg_set_busy(True)
        self.seg_status.set_text("Refining selection…")
        self._send_job(Job.SEG_RUN, {"op": "seg_point", "points": self.seg_points,
                                     "use_prev": len(self.seg_points) > 1})

    def _on_seg_select_all(self):
        self.seg_panel.view.set_seg_selection([o["id"] for o in self.seg_objects])
        self._update_seg_selection_ui()

    def _seg_forget_overlays(self):
        """Take the current mode's objects or click selection off the page."""
        if self._seg_layers_loading is not None:   # objects still decoding: dropped
            self._seg_layers_loading = None
            self._seg_set_busy(False)
        self.seg_panel.view.clear_seg(keep_mode=True)
        self.seg_objects = []
        self._seg_maps = None
        self.seg_points = []
        self.seg_point_mask = None
        self._seg_point_contour = None
        self.seg_result_output = None
        self.seg_res_panel.view.clear()
        self.seg_save_btn.set_sensitive(False)

    def _seg_clear_overlays(self):
        """The Clear button: the one deliberate way to drop what Segment found."""
        self._seg_forget_overlays()
        if not self.seg_busy:
            self.seg_status.set_text("Cleared.")

    def _seg_snapshot(self, mode):
        """What `mode` has on the page, to put back when the user returns to it."""
        if mode == "point":
            if self.seg_point_mask is None:
                return None
            return {"points": list(self.seg_points), "mask": self.seg_point_mask,
                    "contour": self._seg_point_contour}
        if not self.seg_objects:
            return None
        return {"objects": self.seg_objects, "maps": self._seg_maps,
                "selection": self.seg_panel.view.get_seg_selection()}

    def _seg_restore(self, mode, state):
        """Put back a _seg_snapshot of `mode`, if there is one."""
        if state is None:
            return
        view = self.seg_panel.view
        if mode == "point":
            self.seg_points = state["points"]
            self.seg_point_mask = state["mask"]
            self._seg_point_contour = state["contour"]
            view.set_point_mask(state["mask"], state["contour"])
            self._update_seg_preview()
            self.seg_status.set_text("Your click selection is back — Save, or click to refine.")
            return
        self.seg_objects = state["objects"]
        self._seg_maps = state["maps"]

        def shown(loaded):
            if loaded is Loaded.SHOWN:     # else the user moved on before it showed
                view.set_seg_selection(state["selection"])
                self._update_seg_selection_ui()
        view.set_seg_layers(state["objects"], *state["maps"], done=shown)

    def _update_seg_selection_ui(self):
        self._update_seg_preview()
        n = len(self.seg_panel.view.get_seg_selection())
        if not self.seg_busy:
            if not self.seg_objects:       # a click before there is anything to pick
                self._seg_base_status = ("Press Segment everything first: it finds the "
                                         "objects to pick from.")
            elif n:
                self._seg_base_status = "%d selected — Save selection." % n
            else:
                self._seg_base_status = "Click objects to keep them (they light up on hover)."
            self.seg_status.set_text(self._seg_base_status)

    def _on_seg_hover(self, depth, drilled, wx=0.0, wy=0.0):
        # Hint that overlapping objects are stacked, and how to reach the part.
        if self.seg_busy:
            self.seg_panel.hide_cursor_badge()
            return
        # A floating badge trails the cursor only when objects are stacked here.
        if depth >= 2:
            self.seg_panel.show_cursor_badge(
                "%d objects · focused the part" % depth if drilled
                else "%d objects here · hold to focus the part" % depth, wx, wy)
        else:
            self.seg_panel.hide_cursor_badge()
        # Only rewrite the status (which mirrors to the footer) when the hint
        # actually changes — not on every pointer-motion frame.
        key = (depth, bool(drilled))
        if key == getattr(self, "_last_hover_key", None):
            return
        self._last_hover_key = key
        if depth <= 0:
            self.seg_status.set_text(
                getattr(self, "_seg_base_status", "")
                or "Hover to highlight, click to keep.")
        elif depth >= 2 and drilled:
            self.seg_status.set_text(
                "%d objects here · focused the part — click to keep." % depth)
        elif depth >= 2:
            self.seg_status.set_text(
                "%d objects here · showing the whole — hold still to focus a part."
                % depth)
        else:
            self.seg_status.set_text("Click to keep this object.")

    def _seg_has_selection(self):
        if self._seg_mode() == "point":
            return self.seg_point_mask is not None
        return bool(self.seg_panel.view.get_seg_selection())

    def _seg_extract_req(self, outp):
        bg, blur = self._extract_bg()
        # Mask files and the image by path, not object ids: the worker may have
        # dropped its objects (a reload, a restart) while the page kept them.
        req = {"op": "seg_extract", "bg": bg, "blur": blur, "output": outp,
               "input": self._seg_input}
        if self._seg_mode() == "point":
            req["mask"] = self.seg_point_mask
        else:
            sel = set(self.seg_panel.view.get_seg_selection())
            req["masks"] = [o["mask"] for o in self.seg_objects if o["id"] in sel]
        return req

    def _update_seg_preview(self):
        """Debounced worker prerender: the right panel shows the real output —
        all selected objects composited over the chosen background — exactly like
        the Single result, and as a normal image you can flip/rotate/zoom."""
        # A preview still out is for the selection before this one: drop it
        # now, not when the debounce fires, or it could still land.
        self._abandon_jobs({Job.SEG_PREVIEW})
        if not self._seg_has_selection():
            if getattr(self, "_seg_preview_timer", None):
                GLib.source_remove(self._seg_preview_timer)
                self._seg_preview_timer = None
            self.seg_res_panel.view.clear()
            self.seg_result_output = None
            self.seg_save_btn.set_sensitive(False)
            return
        if getattr(self, "_seg_preview_timer", None):
            GLib.source_remove(self._seg_preview_timer)
        self._seg_preview_timer = GLib.timeout_add(150, self._do_seg_prerender)

    def _do_seg_prerender(self):
        self._seg_preview_timer = None
        if self._seg_has_selection():
            self._send_job(Job.SEG_PREVIEW, self._seg_extract_req(
                os.path.join(self.tmpdir, "seg_preview.png")))
        return False

    def _on_seg_save(self):
        """Render the final PNG in the worker, then offer a save dialog."""
        if self.seg_busy or self._has_job(Job.OPEN):
            return
        if not self._seg_has_selection():
            self._toast("Pick at least one object first.")
            return
        if self._worker_state is not WorkerState.UP:
            return
        self._seg_set_busy(True)
        self.seg_status.set_text("Preparing export…")
        req = self._seg_extract_req(os.path.join(self.tmpdir, "seg_result.png"))
        if self._send_job(Job.SEG_SAVE, req, alpha=_alpha_of(req["bg"])) == -1:
            # No reply will come to end the busy state: end it here.
            self._seg_set_busy(False)
            self.seg_status.set_text("Couldn't save: the background process isn't running.")
            self._toast_error("Could not save the selection: the background process "
                              "isn't running.")

    def _seg_save_file(self, src, alpha):
        """Offer the export at `src`, whose background `alpha` describes."""
        def save(target, choices):
            # The result panel's rotate/flip goes into the file, as on Single.
            view = self.seg_res_panel.view
            transform, transformed = view.transform(), view.is_transformed()

            def render(path, png_options):
                if not transformed:
                    shutil.copyfile(src, path)
                else:
                    pb = transform_pixbuf(GdkPixbuf.Pixbuf.new_from_file(src), transform)
                    pb.savev(path, "png", *png_options)
            self._save_result(target, choices, render, formats.PNG)
        self._ask_save_path("_object", formats.save_formats(alpha), save)

    def _on_seg_cancel(self):
        if self._preparing.pop(Prep.SEGMENT, None) is not None:
            # Still preparing the image: nothing has reached the worker yet.
            self.seg_pending = None
            self._seg_set_busy(False)
            self.seg_status.set_text("Cancelled.")
            return
        # A load past its download can't be stopped. It still finishes, and
        # keeps its model for the next run, but runs nothing.
        self.seg_pending = None
        # Its reply (stopped, or skipped if still queued) ends the busy state.
        self._cancel_jobs(Job.SEG_LOAD, Job.SEG_RUN)
        self.seg_cancel_btn.set_sensitive(False)
        self.seg_status.set_text("Cancelling…")

    def _on_seg_mode_changed(self, *_):
        mode = self._seg_mode()
        old = self._seg_shown_mode
        self._seg_shown_mode = mode
        # A run started in the other mode would land in this one: stop it. A
        # load keeps going, for the next run, but runs nothing.
        if self._has_job(Job.SEG_RUN, Job.SEG_PREVIEW):
            self._abandon_jobs({Job.SEG_RUN, Job.SEG_PREVIEW})
            self._seg_set_busy(False)
        self.seg_pending = None
        # Each mode keeps what it found, for when the user switches back.
        self._seg_stash[old] = self._seg_snapshot(old)
        self._seg_forget_overlays()
        self.seg_panel.view.set_seg_mode(mode)
        self.seg_everything_btn.set_visible(mode == "everything")
        self.seg_selall_btn.set_visible(mode == "everything")
        if mode == "point":
            self.seg_status.set_text(
                "Click an object (Ctrl-click or right-click removes).")
        else:
            self.seg_status.set_text(
                "Click Segment everything, then pick the objects to keep.")
        # After the hint, so a restored selection's own status wins.
        self._seg_restore(mode, self._seg_stash.pop(mode, None))
        self._update_footer()

    def _on_seg_model_changed(self, *_):
        if getattr(self, "_syncing_seg", False):
            return
        self.seg_loaded_for = None  # force re-encode with the new model
        self._seg_in_use = None
        self._refresh_seg_subtitles()
        self.seg_adv.set_subtitle(self.seg_adv.get_subtitle() + " · loads on next segment")
        self._update_footer()

    def _on_seg_focus_changed(self, *_):
        if getattr(self, "seg_panel", None) is None:
            return
        self.seg_panel.view.set_dwell_ms(SEG_FOCUS[self.seg_focus_row.get_selected()][1])

    # ---------- single page ----------
    def _build_single(self):
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        page.set_margin_top(10)
        page.set_margin_bottom(10)
        page.set_margin_start(10)
        page.set_margin_end(10)

        self.gen_btn = primary_button("Generate", icon="image-x-generic-symbolic")
        self.gen_btn.set_sensitive(False)
        self.gen_btn.set_tooltip_text("Remove the background (Ctrl+Return)")
        self.gen_btn.connect("clicked", lambda *_: self._on_generate())
        self.save_btn = save_button("Save result…")
        self.save_btn.set_tooltip_text("Save the cut-out (Ctrl+S)")
        self.save_btn.connect("clicked", lambda *_: self._on_save())
        self.gen_cancel_btn = pill_button("Cancel")
        self.gen_cancel_btn.set_sensitive(False)      # enabled while a pass runs
        self.gen_cancel_btn.connect("clicked", lambda *_: self._on_bg_cancel())
        bar, self.status, self.spinner = action_bar(
            primary=self.gen_btn, cancel=self.gen_cancel_btn, save=self.save_btn,
            status_text="Open or drop an image to begin.")
        page.append(bar)

        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL,
                          wide_handle=True)
        paned.add_css_class("card-split")
        paned.set_vexpand(True)
        self.src_panel = Panel("Source", on_change=self._on_source_change)
        self.res_panel = Panel(
            "Result", empty_icon="emblem-photos-symbolic",
            empty_title="Your cut-out appears here", empty_hint="Press Generate")
        self.res_panel.add_tool(self._output_button())
        paned.set_start_child(self.src_panel.box)
        paned.set_end_child(self.res_panel.box)
        paned.set_resize_start_child(True)
        paned.set_resize_end_child(True)
        # Balance the two panes once the widget knows its width.
        paned.connect("notify::max-position", self._center_paned)
        self.paned = paned
        page.append(paned)
        return page

    def _center_paned(self, paned, *_):
        if not getattr(self, "_paned_centered", False):
            mx = paned.get_property("max-position")
            if mx > 0:
                paned.set_position(mx // 2)
                self._paned_centered = True

    # ---------- batch page ----------
    def _build_batch(self):
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        clamp = Adw.Clamp(maximum_size=640)
        clamp.set_margin_top(24)
        clamp.set_margin_bottom(24)
        clamp.set_margin_start(12)
        clamp.set_margin_end(12)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        clamp.set_child(box)
        outer.append(clamp)

        grp = Adw.PreferencesGroup(
            title="Batch", description="Process every image in a folder.")
        self.in_row = Adw.ActionRow(title="Input folder", subtitle="Not set")
        in_btn = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        in_btn.connect("clicked", self._on_pick_input)
        self.in_row.add_suffix(in_btn)
        grp.add(self.in_row)

        self.out_row = Adw.ActionRow(title="Output folder", subtitle="Not set")
        out_btn = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        out_btn.connect("clicked", self._on_pick_output)
        self.out_row.add_suffix(out_btn)
        grp.add(self.out_row)

        self.pattern_row = Adw.EntryRow(title="Rename pattern")
        self.pattern_row.set_text("{name}_nobg")
        self.pattern_row.connect("changed", lambda *_: self._check_pattern())
        grp.add(self.pattern_row)
        box.append(grp)

        hint = Gtk.Label(xalign=0, wrap=True)
        hint.add_css_class("dim-label")
        hint.set_markup(_PATTERN_HINT)
        self.pattern_hint = hint
        box.append(hint)

        run_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        run_row.set_halign(Gtk.Align.START)
        self.run_btn = primary_button("Run batch", icon="view-grid-symbolic")
        self.run_btn.set_sensitive(False)
        self.run_btn.connect("clicked", lambda *_: self._on_run_batch())
        run_row.append(self.run_btn)
        self.batch_cancel_btn = pill_button("Cancel")
        self.batch_cancel_btn.set_sensitive(False)
        self.batch_cancel_btn.connect("clicked", lambda *_: self._on_bg_cancel())
        run_row.append(self.batch_cancel_btn)
        box.append(run_row)

        # Progress + status are shown in the window footer (statusbar), not on the
        # page — kept as detached objects so existing set_fraction/set_text calls
        # (and the footer mirror) keep working.
        self.progress = Gtk.ProgressBar(show_text=True)
        self.batch_status = Gtk.Label(xalign=0, label="")
        return outer

    # ---------- file dialogs ----------
    def _image_filters(self):
        store = Gio.ListStore.new(Gtk.FileFilter)
        f = Gtk.FileFilter(name="Images")
        # Suffixes, unlike patterns, match in any case: cameras write .JPG.
        for suffix in formats.READ_SUFFIXES:
            f.add_suffix(suffix[1:])
        store.append(f)
        return store

    def _on_open_image(self, *_):
        dlg = Gtk.FileDialog(title="Open image")
        dlg.set_filters(self._image_filters())
        start = self._remembered_folder("open_folder")
        if start is not None:
            dlg.set_initial_folder(start)

        def done(d, res):
            try:
                f = d.open_finish(res)
            except GLib.Error:
                return
            self._load_source(f.get_path())
            self._remember(open_folder=os.path.dirname(f.get_path()))
        dlg.open(self, None, done)

    def _on_drop(self, target, value, x, y):
        try:
            files = value.get_files()
        except Exception:
            return False
        paths = [f.get_path() for f in files if f.get_path()]
        if not paths:
            return False
        page = self.stack.get_visible_child_name()
        first = paths[0]
        if os.path.isdir(first):
            if page == "batch":
                self.batch_input = first
                self.in_row.set_subtitle(first)
                self._remember(batch_input=first)
                self._sync_batch()
                self.batch_status.set_text("Input folder: " + first)
            else:
                self._toast("That's a folder. Drop it on the Batch page to process "
                            "every image in it.")
            return True
        self._load_source(first)
        # Stay on the page the image was dropped on; only Batch has no image
        # view, so an image dropped there opens on Single.
        if page == "batch":
            self.stack.set_visible_child_name("single")
        if len(paths) > 1:
            rest = ("the other file was" if len(paths) == 2
                    else "the other %d files were" % (len(paths) - 1))
            self._toast("Opened %s; %s left out. Use Batch for several images."
                        % (os.path.basename(first), rest))
        return True

    def _animation_of(self, path):
        if not formats.can_animate(path):
            return formats.Animation.STILL
        try:
            anim = GdkPixbuf.PixbufAnimation.new_from_file(path)
        except GLib.Error:
            return formats.Animation.STILL
        if anim.is_static_image():
            return formats.Animation.STILL
        return formats.Animation.ANIMATED

    def _load_source(self, path):
        """Open a source image.

        GdkPixbuf previews it when it has a loader for the format; otherwise
        the worker decodes it into a PNG to show. A WebP is adopted only once
        the worker has counted its frames: a WebP loader may decode stills only.
        """
        if not path:
            return
        self._new_open()                    # a paste still on its way is now stale
        if self._has_job(Job.OPEN):         # this open wins over one in flight
            self._abandon_jobs({Job.OPEN})
            self._release_source_actions()
        name = os.path.basename(path)
        if self.src_panel.view.load_file(path):
            if formats.format_of(path) is formats.WEBP:
                self._request_open(path, {"op": "frames", "input": path}, preview=path)
                return
            self._show_source(path, path, self._animation_of(path))
            return
        if not formats.is_readable(path):
            _log.warning("can't open %s: not a format bg-be-gone reads", path)
            self._toast("Can't open %s: not a format bg-be-gone reads." % name)
            return
        # Typically HEIC or AVIF, which few systems have GdkPixbuf loaders for.
        _log.info("no GdkPixbuf loader decoded %s; asking the worker", path)
        self._request_open(path, {"op": "convert", "input": path,
                                  "output": os.path.join(self.tmpdir, "source.png")},
                           preview=None)

    def _request_open(self, path, req, preview):
        """Send the worker the convert or frames request that opening `path` waits on."""
        rid = self._send_job(Job.OPEN, req, source=_PendingOpen(path, preview))
        if rid == -1:
            _log.warning("can't open %s: the worker isn't running", path)
            if preview is not None:        # shown already; it just can't animate
                self._show_source(path, preview, formats.Animation.STILL)
                return
            self._toast("Can't open %s: the worker isn't running." % os.path.basename(path))
            return
        self._drop_old_source_jobs()
        self._hold_source_actions()
        self.status.set_text("Opening %s…" % os.path.basename(path))

    def _hold_source_actions(self):
        """While an open is pending, nothing may run on the old source."""
        for button in (self.gen_btn, self.save_btn, self.seg_everything_btn,
                       self.seg_save_btn):
            button.set_sensitive(False)

    def _release_source_actions(self):
        up = self._worker_state is WorkerState.UP
        self.gen_btn.set_sensitive(not self.busy and self.src_panel.view.has_image() and up)
        self.save_btn.set_sensitive(self.result_output is not None)
        self.seg_everything_btn.set_sensitive(not self.seg_busy and up)
        # Save selection renders in the worker: without one it can't finish.
        self.seg_save_btn.set_sensitive(self.seg_result_output is not None and up)

    def _drop_old_source_jobs(self):
        """A new image replaces the source: drop what is still on its way for
        the old one, so the old cut-out or objects never land next to the new
        image."""
        # An input still being encoded for the old image: it is never sent.
        if self._preparing.pop(Prep.SINGLE, None) is not None:
            self._set_busy(False)
        if self._preparing.pop(Prep.SEGMENT, None) is not None:
            self.seg_pending = None
            self._seg_set_busy(False)
        # A Generate or Segment pass still running for it: whatever it still
        # sends is dropped.
        if self._abandon_jobs({Job.SINGLE}) is Abandoned.SOME:
            self._set_busy(False)
        if self._abandon_jobs(_SEG_JOBS) is Abandoned.SOME:
            self.seg_pending = None
            self._seg_set_busy(False)

    def _on_open_reply(self, pending, msg):
        """The worker's answer to the open in flight: frames, converted or error."""
        self._release_source_actions()
        name = os.path.basename(pending.path)
        if msg["type"] == "error":
            _log.warning("can't open %s: %s", pending.path, msg["message"])
            if pending.preview is not None:   # GdkPixbuf shows it, so it works as a still
                self._show_source(pending.path, pending.preview, formats.Animation.STILL)
                return
            self.status.set_text(msg["message"])
            self._toast_error(msg["message"])
            return
        preview = pending.preview or msg["output"]
        if pending.preview is None and not self.src_panel.view.load_file(preview):
            _log.warning("can't show %s, decoded by the worker to %s", pending.path, preview)
            self._toast("Can't open %s." % name)
            return
        # A multi-page TIFF or multi-image HEIC has frames too, but isn't an animation.
        if formats.can_animate(pending.path) and msg.get("frames", 1) > 1:
            animation = formats.Animation.ANIMATED
        else:
            animation = formats.Animation.STILL
        self._show_source(pending.path, preview, animation)

    def _show_source(self, path, preview, animation):
        """Adopt `path` as the source; `preview` is the file the views show."""
        self._drop_old_source_jobs()
        self.source_path = path
        self._source_animation = animation
        self.res_panel.view.clear()
        self._stop_recomposite()
        self._replace_result(None)
        self._forget_kept()
        self.save_btn.set_sensitive(False)
        self.gen_btn.set_sensitive(not self.busy)   # a batch may still be running
        if self._worker_state is not WorkerState.UP:
            self.gen_btn.set_sensitive(False)       # no worker to run it
        if animation is formats.Animation.ANIMATED:
            tail = " — animated %s, press Generate." % formats.format_of(path).name
        else:
            tail = " — press Generate."
        self.status.set_text(os.path.basename(path) + tail)
        # Mirror into the Segment page and reset its state for the new image.
        if getattr(self, "seg_panel", None) is not None:
            self.seg_panel.view.load_file(preview)
            self.seg_res_panel.view.clear()
            self._seg_forget_overlays()
            self._seg_stash = {}            # the other mode's finds were for the old image
            self._remove_temp_input(self._seg_input)
            self._seg_input = None
            self.seg_loaded_for = None
            self.seg_result_output = None
            self.seg_save_btn.set_sensitive(False)
            self.seg_status.set_text(os.path.basename(path) + " — Segment.")

    def _on_save(self):
        if self._has_job(Job.OPEN):
            return
        # An animated result is the worker's file itself (all frames), saved in
        # its own format; a still result is the current pixbuf.
        result = self.result_output
        if result and formats.can_animate(result):
            fmt = formats.format_of(result)
            release = self._hold_animation(result)
            self._ask_save_path(
                "_nobg", (fmt,), lambda target, choices: self._save_result(
                    target, choices, lambda path, _: shutil.copyfile(result, path), fmt,
                    finished=release),
                dismissed=release)
            return
        # The shown result and its rotate/flip, taken at the press: the encode
        # runs later, off the UI thread, where a 4K PNG can't freeze the window.
        view = self.res_panel.view
        shown, transform = view.pixbuf, view.transform()
        if shown is None:
            return

        def render(path, png_options):
            transform_pixbuf(shown, transform).savev(path, "png", *png_options)
        self._ask_save_path("_nobg", formats.save_formats(self._result_alpha),
                            lambda target, choices: self._save_result(
                                target, choices, render, formats.PNG))

    def _replace_result(self, path):
        """Make `path` the result Save offers; an animation it replaces goes
        once no save reads it."""
        old, self.result_output = self.result_output, path
        if old is not None and old != path and old not in self._animation_saves:
            self._remove_animation(old)
        if old in self._live_results and old != path:
            self._live_results.discard(old)
            _remove_quietly(old)
            _remove_quietly(old + ".preview.png")

    def _hold_animation(self, path):
        """Keep animation result `path` for a save until the returned release()."""
        self._animation_saves[path] = self._animation_saves.get(path, 0) + 1

        def release():
            left = self._animation_saves.pop(path) - 1
            if left:
                self._animation_saves[path] = left
            elif path != self.result_output:
                self._remove_animation(path)
        return release

    def _remove_animation(self, path):
        """Delete an animation result, and its first-frame preview, if it is one."""
        if path in self._animations:
            self._animations.discard(path)
            _remove_quietly(path)
            _remove_quietly(path + ".preview.png")

    def _save_result(self, target, choices, render, rendered, finished=None):
        """Write a result to `target` off the UI thread, whole or not at all.

        render(path, png_options) writes the result, in format `rendered`, to
        `path`, on a background thread. A failed or interrupted save leaves the
        old file whole:

        - in the target's format: render writes a temp file beside the target,
          which replaces it once complete.
        - in another format: render stages a copy, and the worker re-encodes it
          into a temp file beside the target (see _save_via_worker). Pillow
          encodes WebP and JPEG on every platform; GdkPixbuf needs a loader for WebP.
        """
        direct = rendered is target.format

        def work():
            if direct:
                part = _part_beside(target)
                try:
                    render(part, _PNG_FINAL)
                    os.replace(part, target.path)
                finally:
                    _remove_quietly(part)
                return None
            fd, staging = tempfile.mkstemp(suffix=rendered.suffix, dir=self.tmpdir)
            os.close(fd)
            try:
                render(staging, _PNG_HANDOFF)
            except BaseException:
                _remove_quietly(staging)
                raise
            return staging

        def written(result):
            self._saves_writing -= 1
            if finished is not None:
                finished()                   # render is done with what it read
            if isinstance(result, (OSError, GLib.Error)):
                self._report_save_failure(target, result)
            elif isinstance(result, Exception):
                self._after_save(SaveOutcome.FAILED)
                raise result                 # a bug, not a save failure
            elif direct:
                self._report_saved(target, choices)
            else:
                self._save_via_worker(result, target, choices, staging=result)
        self._saves_writing += 1
        offmain.run(work, written)

    def _ask_save_path(self, stem_suffix, choices, save, dismissed=None):
        """Ask where to save, offering `choices`, then call save(target, choices),
        or dismissed() when no file was chosen.

        The name's suffix picks the format. The default is the source's own
        format when it is one of the choices, so a WebP comes back as a WebP.
        """
        default = formats.default_save_format(self.source_path, choices)
        dlg = Gtk.FileDialog(title="Save result")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        for fmt in choices:
            f = Gtk.FileFilter(name=fmt.name)
            for suffix in fmt.suffixes:
                f.add_suffix(suffix[1:])
            filters.append(f)
            if fmt is default:
                dlg.set_default_filter(f)
        dlg.set_filters(filters)
        base = os.path.splitext(os.path.basename(self.source_path or "image"))[0]
        dlg.set_initial_name(base + stem_suffix + default.suffix)
        # Next to the source image, else where the last save went.
        start = self._source_folder() or self._remembered_folder("save_folder")
        if start is not None:
            dlg.set_initial_folder(start)

        def done(d, res):
            try:
                f = d.save_finish(res)
            except GLib.Error:
                if dismissed is not None:
                    dismissed()
                return
            target = formats.save_target(f.get_path(), choices, default)
            if formats.overwrites_unasked(target):
                self._confirm_overwrite(target, lambda: save(target, choices), dismissed)
            else:
                save(target, choices)
        dlg.save(self, None, done)

    def _confirm_overwrite(self, target, save, dismissed=None):
        """Ask before replacing a file the dialog never asked about.

        The dialog confirmed the typed name; save_target then changed its
        suffix, and the file under the new name already exists.
        """
        name = os.path.basename(target.path)
        dlg = Adw.AlertDialog(
            heading="Replace %s?" % name,
            body="This result saves as %s, and %s already exists." % (target.format.name, name))
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("replace", "Replace")
        dlg.set_response_appearance("replace", Adw.ResponseAppearance.DESTRUCTIVE)
        dlg.set_default_response("cancel")
        dlg.set_close_response("cancel")

        def answered(_dlg, response):
            if response == "replace":
                save()
            elif dismissed is not None:
                dismissed()
        dlg.connect("response", answered)
        dlg.present(self)

    def _report_saved(self, target, choices):
        self._toast(_saved_notice(target, choices))
        self._remember(save_folder=os.path.dirname(target.path))
        self._after_save(SaveOutcome.WRITTEN)

    def _report_save_failure(self, target, exc):
        self._report_failure("saving " + target.path, exc, "Could not save %s: %s." % (
            os.path.basename(target.path), plain_reason(exc)))
        self._after_save(SaveOutcome.FAILED)

    def _saves_running(self):
        """Saves still being written: rendered off the UI thread, or re-encoded
        by the worker."""
        return self._saves_writing + len(self._rids(Job.SAVE))

    def _after_save(self, outcome):
        """A save ended: close now if a quit was waiting for the last one."""
        if self._quit is not Quit.AFTER_SAVES:
            return
        if outcome is SaveOutcome.FAILED:
            # Stay open, so the error is seen, whatever the other saves do.
            self._quit = Quit.NOT_ASKED
            return
        if not self._saves_running():
            self._quit = Quit.NOT_ASKED
            self.close()

    def _save_via_worker(self, src, target, choices, staging):
        """Have the worker re-encode `src` as `target`; `staging` is ours to delete.

        The worker writes a temp file beside the target, which replaces the
        target once the worker says it's done, so a failed encode leaves the
        old file whole.
        """
        try:
            part = _part_beside(target)
        except OSError as e:
            _remove_quietly(staging)
            self._report_save_failure(target, e)
            return
        rid = self._send_job(Job.SAVE, {"op": "convert", "input": src, "output": part},
                             save=_PendingSave(target, choices, staging, part))
        if rid == -1:
            _remove_quietly(staging)
            _remove_quietly(part)
            self._toast_error("Could not save %s: the background process isn't running."
                              % os.path.basename(target.path))
            self._after_save(SaveOutcome.FAILED)

    def _on_save_reply(self, pending, msg):
        _remove_quietly(pending.staging)
        if msg["type"] == "error":
            _remove_quietly(pending.part)
            _log.warning("saving %s failed: %s", pending.target.path, msg["message"])
            self._toast_error("Could not save %s: %s" % (
                os.path.basename(pending.target.path), msg["message"]))
            self._after_save(SaveOutcome.FAILED)
            return
        try:
            os.replace(pending.part, pending.target.path)
        except OSError as e:
            _remove_quietly(pending.part)
            self._report_save_failure(pending.target, e)
            return
        self._report_saved(pending.target, pending.choices)

    def _on_pick_input(self, *_):
        dlg = Gtk.FileDialog(title="Input folder")
        start = self._remembered_folder("batch_input")
        if start is not None:
            dlg.set_initial_folder(start)

        def done(d, res):
            try:
                f = d.select_folder_finish(res)
            except GLib.Error:
                return
            self.batch_input = f.get_path()
            self.in_row.set_subtitle(self.batch_input)
            self._remember(batch_input=self.batch_input)
            self._sync_batch()
        dlg.select_folder(self, None, done)

    def _on_pick_output(self, *_):
        dlg = Gtk.FileDialog(title="Output folder")
        start = (self._remembered_folder("batch_output")
                 or self._remembered_folder("batch_input"))
        if start is not None:
            dlg.set_initial_folder(start)

        def done(d, res):
            try:
                f = d.select_folder_finish(res)
            except GLib.Error:
                return
            self.batch_output = f.get_path()
            self.out_row.set_subtitle(self.batch_output)
            self._remember(batch_output=self.batch_output)
            self._sync_batch()
        dlg.select_folder(self, None, done)

    def _pattern_problem(self):
        """Why the rename pattern can't name files, or None."""
        try:
            batchnames.check_pattern(self.pattern_row.get_text().strip() or "{name}_nobg")
        except batchnames.BatchPatternError as e:
            return e.user_message
        return None

    def _check_pattern(self):
        problem = self._pattern_problem()
        if problem:
            self.pattern_row.add_css_class("error")
            self.pattern_hint.set_text(problem)
        else:
            self.pattern_row.remove_css_class("error")
            self.pattern_hint.set_markup(_PATTERN_HINT)
        self._sync_batch()

    def _sync_batch(self):
        self.run_btn.set_sensitive(
            bool(self.batch_input and self.batch_output) and not self.busy
            and self._worker_state is WorkerState.UP)
        if self._pattern_problem() is not None:
            self.run_btn.set_sensitive(False)   # Run waits for a pattern that works

    # ---------- actions ----------
    def _set_busy(self, busy, spin=True, cancel=False):
        self.busy = busy
        self.gen_btn.set_sensitive(not busy and self.src_panel.view.has_image()
                                   and self._worker_state is WorkerState.UP)
        self.gen_cancel_btn.set_sensitive(busy and cancel)
        self.batch_cancel_btn.set_sensitive(busy and cancel)
        self._sync_batch()
        if spin:
            (self.spinner.start if busy else self.spinner.stop)()
        if not busy:
            self.src_panel.view.set_scanning(False)   # end the scan shimmer
            self._hide_footer_progress()
            self._schedule_recomposite()      # a background change waits for the job
            GLib.idle_add(self._replace_worker_when_idle)   # an Engine change waits for this job

    def _on_source_change(self):
        if self.src_panel.view.has_image() and self.result_output:
            self.status.set_text("Source changed — press Generate to update.")

    def _mark_stale(self, *_):
        if getattr(self, "result_output", None):
            self.status.set_text("Settings changed — press Generate to update.")

    def _on_generate(self):
        if self.busy or self._has_job(Job.OPEN) or not self.src_panel.view.has_image():
            return
        if self._worker_state is WorkerState.DOWN:
            return
        self._stop_recomposite()             # Generate makes a new result anyway
        model, bg, alpha, blur = self.get_settings()
        if self._source_animation is formats.Animation.ANIMATED:
            self._generate_animation(model, bg, alpha, blur)
            return
        out = os.path.join(self.tmpdir, "result.png")
        self._set_busy(True, cancel=True)
        self.src_panel.view.set_scanning(True)      # scan shimmer over the source
        self.status.set_text("Generating…")

        kept = self._new_kept(bg, blur)

        def send(inp):
            # The input is deleted once the job ends, if we encoded it.
            self._send_job(Job.SINGLE, {
                "op": "single", "input": inp, "output": out,
                "model": model, "alpha": alpha, "bg": bg, "blur": blur,
                "keep_dir": kept.folder},
                alpha=_alpha_of(bg), input=inp, kept=kept)

        def failed(e):
            self.src_panel.view.set_scanning(False)
            self._set_busy(False)
            self._report_failure("preparing the Generate input", e,
                                 "Could not prepare the image: %s." % plain_reason(e))

        self._prepare_input(Prep.SINGLE, self.src_panel.view, send, failed)

    def _generate_animation(self, model, bg, alpha, blur):
        # Same options as a still; the worker processes each frame and reports
        # per-frame progress. Pass the current rotate/flip so it bakes in like a
        # still would; process the original file (all frames) not the preview.
        # The result keeps the source's format: a WebP keeps soft alpha edges.
        fmt = formats.format_of(self.source_path)
        # A file of its own: a save of the previous animation may still read
        # that one (see _hold_animation).
        self._animations_made += 1
        out = os.path.join(self.tmpdir, "result-%d%s" % (self._animations_made, fmt.suffix))
        v = self.src_panel.view
        self._set_busy(True, cancel=True)
        self.src_panel.view.set_scanning(True)
        self.status.set_text("Removing background from %s…" % fmt.name)
        source = {"input": self.source_path, "rot": v.rot, "fh": v.fh, "fv": v.fv,
                  "suffix": fmt.suffix}
        kept = self._new_kept(bg, blur, animation=source)
        self._send_job(Job.SINGLE, {
            "op": "gif", "input": self.source_path, "output": out,
            "model": model, "alpha": alpha, "bg": bg, "blur": blur,
            "rot": v.rot, "fh": v.fh, "fv": v.fv, "keep_dir": kept.folder},
            alpha=_alpha_of(bg), kept=kept)

    def _on_bg_cancel(self):
        if self._has_job(Job.RECOMPOSITE):
            self._cancel_jobs(Job.RECOMPOSITE)
            self.gen_cancel_btn.set_sensitive(False)
        if not self.busy:
            return
        if self._preparing.pop(Prep.SINGLE, None) is not None:
            # Still preparing the image: nothing has reached the worker yet.
            self._set_busy(False)
            self.status.set_text("Cancelled.")
            return
        # Its reply (stopped, or skipped if still queued) ends the busy state.
        self._cancel_jobs(Job.SINGLE, Job.BATCH)
        self.gen_cancel_btn.set_sensitive(False)
        self.batch_cancel_btn.set_sensitive(False)
        self.status.set_text("Cancelling…")
        if self.stack.get_visible_child_name() == "batch":
            self.batch_status.set_text("Cancelling…")

    def _on_run_batch(self):
        if (self.busy or not (self.batch_input and self.batch_output)
                or self._worker_state is WorkerState.DOWN):
            return
        model, bg, alpha, blur = self.get_settings()
        pattern = self.pattern_row.get_text().strip() or "{name}_nobg"
        self._set_busy(True, spin=False, cancel=True)
        self.progress.set_fraction(0)
        self.progress.set_text("starting…")
        self.batch_status.set_text("Preparing…")
        self._send_job(Job.BATCH, {
            "op": "batch", "input_dir": self.batch_input,
            "output_dir": self.batch_output, "model": model,
            "alpha": alpha, "bg": bg, "blur": blur, "pattern": pattern})

    # ---------- worker replies ----------
    def _on_worker_message(self, msg):
        t = msg.get("type")
        rid = msg.get("id")
        if t in _JOB_REPLIES and rid in self._abandoned:
            # Given up on: its replies, success or error, touch nothing. Once
            # it ends, the worker is done with any input of ours it read.
            if t in _JOB_ENDS:
                gone = self._abandoned.pop(rid)
                self._remove_temp_input(gone.input)
                self._drop_kept(gone.kept)
            if t == "recomposited":
                # Written for nothing: a newer change replaced it.
                for written in {msg["output"], msg.get("preview") or msg["output"]}:
                    _remove_quietly(written)
            return False
        if t in _JOB_ENDS:
            req = self._jobs.pop(rid, None)
            if req is not None:
                self._remove_temp_input(req.input)
                if t not in _KEPT_ADOPTED:
                    self._drop_kept(req.kept)
                # An Engine change waits for every job; after this reply is handled.
                GLib.idle_add(self._replace_worker_when_idle)
        else:
            req = self._jobs.get(rid)
        job = req.job if req is not None else None
        if job is Job.BATCH and t in _JOB_ENDS and self._quit is Quit.AFTER_BATCH:
            GLib.idle_add(self._finish_quit)   # once this last reply is handled
        if msg.get("device"):           # a finished job: where it actually ran
            self._set_device(msg["device"]["provider"], msg["device"]["label"])
        if job is Job.OPEN:
            self._on_open_reply(req.source, msg)
            return False
        if job is Job.SAVE:
            self._on_save_reply(req.save, msg)
            return False
        if t == "worker_exited":
            self._show_worker_down("The background process stopped (exit code %s)."
                                   % _exit_code_text(msg["code"]))
        elif t == "ready":
            self.worker.started()
            if self._worker_state is WorkerState.DOWN:
                self._show_worker_up()      # whatever started it, a worker is up
            self._offer_engines(msg.get("engines") or ["ort"], msg.get("settings"))
            self._set_bg_engine(selection.Engine(msg.get("engine", "ort")))
            provs = msg.get("providers") or []
            if provs:
                # Installed is not proven: the first run says where it really ran.
                self._set_device(provs[0],
                                 PROVIDER_LABELS.get(provs[0], provs[0]) + " · not used yet")
            self._seg_available = bool(msg.get("seg"))
            self._seg_models = msg.get("seg_models") or []
            self._apply_seg_models()
            self.seg_page.set_property("visible", self._seg_available)
            if not msg.get("bgremove", True):
                # Segmentation-only build: focus the Segment page.
                self.single_page.set_property("visible", False)
                self.batch_page.set_property("visible", False)
                if self._seg_available:
                    self.stack.set_visible_child_name("segment")
            self._sync_sidebar_page()
            if SELFTEST:
                _finish_selftest(self, {"ok": True, "ready": msg})
        elif t == "loading":
            self._footer_loading()
            if job is Job.BATCH:
                self.batch_status.set_text("Loading model %s…" % msg["model"])
            else:
                self.status.set_text(
                    "Loading %s… (the first run is slower)" % msg["model"])
        elif t == "notice":
            # Stays: it explains a slow run, or why files were left out.
            self.toasts.add_toast(Adw.Toast(title=msg["message"], timeout=0))
        elif t == "device":
            if msg.get("model") in modelinfo.BG_MODELS and msg.get("engine"):
                self._note_engine_used(msg["model"], selection.Engine(msg["engine"]))
            self._set_device(msg["provider"], msg["label"])
            if msg.get("gpu") and not self._gpu_notified:
                self._gpu_notified = True
                self._toast("%s ready — the first run optimises GPU kernels and "
                            "is slower; later runs are fast." % msg["label"])
        elif t == "done_single":
            self._replace_result(msg["output"])
            self._result_alpha = req.alpha
            self._adopt_kept(req.kept)
            # Save waits for the decode: until then the panel shows the old result.
            self.save_btn.set_sensitive(False)
            output = msg["output"]

            def decoded(loaded):
                if loaded is not Loaded.SHOWN or self.result_output != output:
                    return            # failed, or a newer result replaced it
                if self._has_job(Job.OPEN):
                    return            # a new image is on its way
                self.save_btn.set_sensitive(True)

            self.res_panel.view.load_file_async(output, done=decoded)
            self.status.set_text("Done in %.2fs." % msg["seconds"])
            self._set_busy(False)
        elif t == "gif_progress" and job is Job.RECOMPOSITE:
            self._footer_progress(msg["done"], msg["total"])
            self.status.set_text("Changing the background — frame %d / %d"
                                 % (msg["done"], msg["total"]))
        elif t == "gif_progress":
            self._footer_progress(msg["done"], msg["total"])
            if msg["done"] == 0:
                self.status.set_text("Removing background — %d frames…"
                                     % msg["total"])
            else:
                self.status.set_text("Removing background — frame %d / %d"
                                     % (msg["done"], msg["total"]))
        elif t == "gif_done":
            # The animation plays where GdkPixbuf has a loader for it (always
            # for GIF); otherwise the worker's first-frame PNG stands in.
            if not self.res_panel.view.load_file(msg["output"]):
                self.res_panel.view.load_file(msg["preview"])
            self._animations.add(msg["output"])
            self._replace_result(msg["output"])
            self._result_alpha = req.alpha
            self.save_btn.set_sensitive(True)
            self.status.set_text("Done: %d frames in %.1fs — Save the %s."
                                 % (msg["frames"], msg["seconds"],
                                    formats.format_of(msg["output"]).name))
            self._set_busy(False)
            self._adopt_kept(req.kept)
        elif t == "recomposited":
            self._on_recomposited(req, msg)
        elif t == "canceled" and job is Job.RECOMPOSITE:
            self._recomposite_over()
            self.status.set_text("Cancelled: the result keeps its last background.")
        elif t == "canceled":
            self._set_busy(False, spin=(msg.get("scope") != "batch"))
            if msg.get("scope") == "batch":
                self.progress.set_text("cancelled")
                self.batch_status.set_text(
                    # No count when it stopped before the first file, say in a download.
                    "Cancelled after %d / %d." % (msg["done"], msg["total"])
                    if "total" in msg else "Cancelled.")
            else:
                self.status.set_text("Cancelled.")
        elif t == "progress":
            total = max(msg["total"], 1)
            self.progress.set_fraction(msg["done"] / total)
            self.progress.set_text("%d / %d" % (msg["done"], total))
            self._footer_progress(msg["done"], total)
            if msg.get("name"):
                self.batch_status.set_text("Processing " + msg["name"])
        elif t == "done_batch":
            self.progress.set_fraction(1.0)
            self.progress.set_text("done")
            self._set_busy(False, spin=False)
            self._on_batch_done(msg)
        elif t == "seg_ready":
            self._seg_set_busy(False)
            self.seg_loaded_for = req.loads
            # The row names what loaded, so Auto's pick is on screen.
            self._seg_in_use = msg
            self._refresh_seg_subtitles()
            self._update_footer()
            self._set_device(msg["provider"], msg["label"])
            cb, self.seg_pending = self.seg_pending, None
            if cb:
                cb()
            else:
                self.seg_status.set_text("Ready: %s on %s." %
                                         (msg["model"], msg["label"]))
        elif t == "seg_download":
            self.seg_status.set_text("Downloading the Segment model: %s. It's needed once."
                                     % self._download_text(msg))
            self._footer_download(msg)
        elif t == "download":
            info = modelinfo.BG_MODELS.get(msg["model"])
            name = info.name if info else msg["model"]
            text = "Downloading %s: %s. It's needed once." % (name, self._download_text(msg))
            (self.batch_status if job is Job.BATCH else self.status).set_text(text)
            self._footer_download(msg)
        elif t == "seg_step":
            label = {m["rung"]: m["label"] for m in self._seg_models}.get(msg["rung"], msg["rung"])
            self.seg_status.set_text("%s couldn't load — trying a lighter model…" % label)
        elif t == "seg_progress":
            # The scan shimmer conveys activity; keep a calm static label, but
            # also drive the shared footer progress bar.
            if msg.get("phase") == "layers":
                self.seg_status.set_text("Building layers…")
            else:
                self.seg_status.set_text("Finding objects…")
            self._footer_progress(msg["done"], msg["total"])
        elif t == "seg_objects":
            self.seg_objects = msg["objects"]
            self._seg_maps = (msg["label_map"], msg.get("general_map"), msg.get("depth_map"))
            status = "%d objects — hover to highlight, click to keep." % msg["count"]

            # A new image, Clear or a mode switch drops this load and ends the
            # wait itself (_seg_forget_overlays); the token says if it still counts.
            token = self._seg_layers_loading = object()

            def shown(loaded):
                if self._seg_layers_loading is not token:
                    return
                self._seg_layers_loading = None
                # Busy until the layers are on screen, so nothing is clicked
                # before there is anything to click.
                self._seg_set_busy(False)
                if loaded is Loaded.SHOWN:
                    self._update_seg_preview()
                    self._seg_base_status = status
                    self.seg_status.set_text(status)

            self.seg_panel.view.set_seg_layers(
                msg["objects"], msg["label_map"],
                msg.get("general_map"), msg.get("depth_map"), done=shown)
        elif t == "seg_mask":
            self._seg_set_busy(False)
            self.seg_point_mask = msg["mask"]
            self._seg_point_contour = msg.get("contour")
            self.seg_panel.view.set_point_mask(msg["mask"], msg.get("contour"))
            self._update_seg_preview()
            self.seg_status.set_text(
                "Object selected (the model is %d%% sure) — Save, or click to refine."
                % round(100 * msg["score"]))
        elif t == "seg_extracted":
            out = msg["output"]
            if job is Job.SEG_PREVIEW:              # live prerender → right panel
                # The selection can be cleared while the prerender was out.
                if not self._seg_has_selection():
                    return False
                self.seg_res_panel.view.load_file_async(out, transform=Transform.KEEP)
                self.seg_result_output = out
                self.seg_save_btn.set_sensitive(True)
            else:                                    # explicit Save → file dialog
                self._seg_set_busy(False)
                self.seg_status.set_text("Ready — choose where to save.")
                self._seg_save_file(out, req.alpha)
        elif t == "seg_canceled":
            # The model stays loaded, so the next run starts at once; leaving
            # the page is what frees it.
            self._seg_set_busy(False)
            self.seg_status.set_text("Cancelled.")
        elif t == "error":
            self._on_job_error(job, msg["message"])
        return False

    def _on_batch_done(self, msg):
        """Summarise a finished batch; list what failed or was skipped."""
        failed, skipped = msg["failed"], msg["skipped"]
        parts = ["%d done" % msg["count"]]
        if failed:
            parts.append("%d failed" % len(failed))
        if skipped:
            parts.append("%d skipped" % len(skipped))
        summary = ", ".join(parts)
        renamed = msg["renamed"]
        if renamed:
            summary += " (%d numbered, so nothing was overwritten)" % renamed
        self.batch_status.set_text("%s in %.1fs → %s" % (summary, msg["seconds"], msg["outdir"]))
        if not (failed or skipped):
            toast = Adw.Toast(title="Batch finished: " + summary, timeout=5)
            toast.set_button_label("Open folder")
            toast.connect("button-clicked", lambda *_: self._open_folder(msg["outdir"]))
            self.toasts.add_toast(toast)
            return
        lines = []
        for title, entries in (("Failed", failed), ("Skipped", skipped)):
            if not entries:
                continue
            lines.append(title + ":")
            lines += ["%s: %s" % (e["name"], e["message"]) for e in entries[:_MAX_LISTED]]
            if len(entries) > _MAX_LISTED:
                lines.append("…and %d more; the log has them all."
                             % (len(entries) - _MAX_LISTED))
        dlg = Adw.AlertDialog(heading="Batch finished: " + summary, body="\n".join(lines))
        dlg.add_response("close", "Close")
        dlg.add_response("open", "Open folder")
        dlg.set_default_response("close")
        dlg.connect("response", lambda d, r: self._open_folder(msg["outdir"])
                    if r == "open" else None)
        dlg.present(self)

    def _open_folder(self, path):
        launcher = Gtk.FileLauncher.new(Gio.File.new_for_path(path))

        def done(launcher, res):
            try:
                launcher.launch_finish(res)
            except GLib.Error as e:
                self._report_failure("opening " + path, e, "Could not open %s: %s."
                                     % (path, e.message))
        launcher.launch(self, None, done)

    def _on_job_error(self, job, text):
        """Show a failed job's message on its own page, and keep it on screen."""
        if job is Job.SEG_PREVIEW:
            # The live preview only: a Segment run may still be going.
            self.seg_status.set_text(text)
        elif job in _SEG_JOBS:
            self.seg_pending = None
            self._seg_set_busy(False)
            self.seg_status.set_text(text)
        elif job is Job.BATCH:
            self.batch_status.set_text(text)
            self._set_busy(False, spin=False)
        elif job is Job.SINGLE:
            self.status.set_text(text)
            self._set_busy(False)
        elif job is Job.RECOMPOSITE:
            # The shown result keeps its last background, e.g. when an
            # animation's source changed on disk since it was generated.
            self.status.set_text(text)
            self._recomposite_over()
        else:                       # a reply to no request we track
            for label in (self.status, self.batch_status, self.seg_status):
                label.set_text(text)
        self._toast_error(text)

    def _on_help(self, *_):
        dlg = Adw.Dialog()
        dlg.set_title("How to use bg-be-gone")
        dlg.set_content_width(560)
        dlg.set_content_height(640)
        tv = Adw.ToolbarView()
        tv.add_top_bar(Adw.HeaderBar())
        page = Adw.PreferencesPage()

        def group(title, description, rows):
            g = Adw.PreferencesGroup(title=title, description=description)
            for icon, t, s in rows:
                r = Adw.ActionRow(title=t, subtitle=s)
                r.add_prefix(Gtk.Image.new_from_icon_name(icon))
                g.add(r)
            page.add(g)

        group("Remove a background", "Single &amp; Batch tabs", [
            ("document-open-symbolic", "Open, drop or paste an image",
             "Drag a file in, click Open in the header, or press Ctrl+V."),
            ("applications-graphics-symbolic", "Pick a subject or model",
             "General, portrait, anime — or an exact model with Subject → Custom model…"),
            ("image-x-generic-symbolic", "Generate, then Save",
             "Output transparent, blurred, or a solid colour."),
            ("view-grid-symbolic", "Batch a whole folder",
             "Choose input/output folders and Run batch."),
        ])
        group("Segment objects", "The Segment tab", [
            ("edit-select-all-symbolic", "Everything → pick layers",
             "Segment everything, then click the objects to keep — the rest dim."),
            ("edit-find-symbolic", "Click to select",
             "Click one object; Ctrl-click or right-click removes; click to refine."),
            ("input-mouse-symbolic", "Hold Space to pan",
             "Space + drag moves the image without selecting. Scroll to zoom."),
            ("document-save-symbolic", "Save selection",
             "Your picks composite live on the right; Save exports a PNG or WebP."),
        ])
        group("Keyboard &amp; mouse", None, [
            ("document-open-symbolic", "Ctrl+O", "Open an image."),
            ("edit-paste-symbolic", "Ctrl+V",
             "Paste an image, or an image file copied in a file manager."),
            ("media-playback-start-symbolic", "Ctrl+Return",
             "Generate · Segment everything · Run batch, on the page you're on."),
            ("process-stop-symbolic", "Esc", "Cancel what the page is running."),
            ("document-save-symbolic", "Ctrl+S", "Save the result or the selection."),
            ("window-close-symbolic", "Ctrl+Q · Ctrl+W", "Quit."),
            ("input-keyboard-symbolic", "Space", "Hold to pan the canvas."),
            ("input-mouse-symbolic", "Scroll · drag · double-click",
             "Zoom · pan · fit/100%."),
            ("help-about-symbolic", "F1", "Open this help."),
        ])
        tv.set_content(page)
        dlg.set_child(tv)
        dlg.present(self)

    def _toast(self, text):
        self.toasts.add_toast(Adw.Toast(title=text, timeout=3))

    def _show_worker_down(self, what):
        """No worker: say so until it's restarted, and stop offering actions."""
        self._worker_state = WorkerState.DOWN
        self._end_requests_in_flight()
        self.seg_pending = None
        self.seg_loaded_for = None
        self._forget_seg_model()
        self._set_busy(False)
        self._seg_set_busy(False)
        self._release_source_actions()
        self.worker_banner.set_title("%s Its log: %s" % (what, LOG))
        self.worker_banner.set_revealed(True)
        self.device_lbl.set_text("No background process")
        for label in (self.status, self.batch_status, self.seg_status):
            label.set_text("The background process isn't running. Press Restart.")

    def _end_requests_in_flight(self):
        """Nothing will answer the requests in flight: end each one here."""
        lost = list(self._jobs.values()) + list(self._abandoned.values())
        self._jobs.clear()
        self._abandoned.clear()
        self._preparing.clear()
        for req in lost:
            self._remove_temp_input(req.input)
            self._drop_kept(req.kept)
            if req.job is Job.SAVE:
                self._on_save_reply(req.save, {
                    "type": "error", "message": "the background process stopped."})

    def _restart_worker(self):
        self.worker.shutdown()
        # Before the new worker exists, so none of the old requests waits on it.
        self._end_requests_in_flight()
        # The new worker reads the saved Engine setting at start-up: a change
        # still waiting for a worker is applied by this start.
        self._engine_switch = None
        self._start_worker(first_id=self.worker.next_id)
        if not self.worker.ok:
            self._show_worker_down("Could not start the background process (%s)."
                                   % self.worker.error)
            return
        self._show_worker_up()

    def _show_worker_up(self):
        """A worker is back: take the banner down and offer the actions again."""
        self._worker_state = WorkerState.UP
        self.worker_banner.set_revealed(False)
        self.device_lbl.set_text("Detecting device…")
        for label in (self.status, self.batch_status, self.seg_status):
            label.set_text("Restarted.")
        self._set_busy(False)
        self._seg_set_busy(False)
        self._release_source_actions()

    def _report_failure(self, what, exc, text):
        """A frontend failure: traceback to the log, `text` on screen."""
        _log.error("%s failed", what, exc_info=exc)
        self._toast_error(text)

    def _toast_error(self, text):
        """A toast that stays until dismissed, with a way to the log."""
        toast = Adw.Toast(title=text, timeout=0)
        toast.set_button_label("Open log")
        toast.set_action_name("win.open-log")
        self.toasts.add_toast(toast)

    def _on_open_log(self, *_):
        launcher = Gtk.FileLauncher.new(Gio.File.new_for_path(LOG))

        def done(launcher, res):
            try:
                launcher.launch_finish(res)
            except GLib.Error as e:
                self._toast("Could not open the log (%s): %s" % (e.message, LOG))
        launcher.launch(self, None, done)

    def _on_close(self, *_):
        if self._quit is Quit.AFTER_BATCH:
            return True                 # already stopping the batch to quit
        if self._saves_running():
            # Quitting now could cut a save short: finish it, then close.
            self._quit = Quit.AFTER_SAVES
            self._toast("Finishing the save, then quitting…")
            return True
        if self._has_job(Job.BATCH):
            # Closing stops the batch; a minutes-long run deserves a question.
            dlg = Adw.AlertDialog(
                heading="A batch is running",
                body="Quitting stops it. The images done so far stay in the output folder.")
            dlg.add_response("keep", "Keep running")
            dlg.add_response("quit", "Stop and quit")
            dlg.set_response_appearance("quit", Adw.ResponseAppearance.DESTRUCTIVE)
            dlg.set_default_response("keep")
            dlg.set_close_response("keep")

            def answered(d, response):
                if response == "quit":
                    self._stop_batch_then_quit()
            dlg.connect("response", answered)
            dlg.present(self)
            return True
        self._shut_down()
        return False

    def _stop_batch_then_quit(self):
        """Cancel the batch, and quit once it has stopped, or after a short wait.

        Stopping it lets the worker delete its own unfinished file; shutting
        the worker down mid-file left that hidden file behind. The wait runs on
        the main loop, so the window stays responsive.
        """
        self._quit = Quit.AFTER_BATCH
        self._cancel_jobs(Job.BATCH)
        self.batch_status.set_text("Stopping the batch, then quitting…")
        self._quit_timer = GLib.timeout_add(_BATCH_STOP_WAIT_MS, self._finish_quit)

    def _finish_quit(self):
        """Quit after a batch stop: its last reply came, or the wait ran out."""
        if self._quit is not Quit.AFTER_BATCH:
            return False
        self._quit = Quit.NOT_ASKED
        if self._quit_timer:
            GLib.source_remove(self._quit_timer)
            self._quit_timer = 0
        self._shut_down()
        self.destroy()                  # no second close-request, no second question
        return False

    def _shut_down(self):
        if self.is_maximized():
            self._remember(window_maximized=True)
        else:
            self._remember(window_maximized=False, window_width=self.get_width(),
                           window_height=self.get_height())
        self.worker.shutdown()
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class App(Adw.Application):
    def __init__(self):
        flags = Gio.ApplicationFlags.HANDLES_OPEN
        if SELFTEST:
            # A self-test never registers with, or hands off to, another instance.
            flags |= Gio.ApplicationFlags.NON_UNIQUE
        super().__init__(application_id=INSTANCE_ID, flags=flags)
        self._pending_open = None
        self.opened = None             # the last file handed in to open; the self-test reports it

    def do_startup(self):
        Adw.Application.do_startup(self)
        provider = Gtk.CssProvider()
        provider.load_from_string(CSS)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        plat.frontend_started(APP_ID)
        about = Gio.SimpleAction.new("about", None)
        about.connect("activate", self._on_about)
        self.add_action(about)
        help_act = Gio.SimpleAction.new("help", None)
        help_act.connect("activate", self._on_help)
        self.add_action(help_act)
        self.set_accels_for_action("app.help", ["F1"])

    def _on_help(self, *_):
        win = self.get_active_window()
        if win is not None and hasattr(win, "_on_help"):
            win._on_help()

    def do_activate(self):
        win = Window(self)
        if self._pending_open:
            win._load_source(self._pending_open)
            self._pending_open = None
        win.present()
        plat.frontend_window(win)
        if SELFTEST:
            if not win.worker.ok:
                _finish_selftest(win, {"ok": False, "error": win.worker.error})
            else:
                GLib.timeout_add_seconds(_SELFTEST_TIMEOUT_S, _finish_selftest, win,
                                         {"ok": False, "error": "no ready from the worker"})
        # Env-gated demo helper: auto-run Generate once, for screenshots/tests.
        if os.environ.get("BGBG_AUTOGEN") and win.src_panel.view.has_image():
            GLib.timeout_add(1400, lambda: (win._on_generate(), False)[1])

    def do_open(self, files, n, hint):
        if files:
            self._pending_open = self.opened = files[0].get_path()
        self.do_activate()

    def _on_about(self, *_):
        about = Adw.AboutWindow(
            application_name="bg-be-gone",
            application_icon="io.github.cristeigabriela.BgBeGone",
            developer_name="bg-be-gone",
            version=APP_VERSION,
            comments="Local image background remover (BiRefNet / rembg) and "
                     "object segmentation (Segment Anything / SAM 2.1).",
            license_type=Gtk.License.MIT_X11,
            website="https://github.com/cristeigabriela/bg-be-gone")
        about.set_transient_for(self.get_active_window())
        about.present()


def main():
    # On Windows the log sits in the app's own data folder, which the first
    # start has to make; GLib's cache folder on Linux already exists.
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    # Shares the worker's log file, so one file tells the whole story.
    logging.basicConfig(filename=LOG, level=logging.INFO,
                        format="%(asctime)s frontend %(levelname)s: %(message)s")
    app = App()
    status = app.run(sys.argv)
    if app.get_is_remote():
        plat.frontend_handed_off()
    return status


if __name__ == "__main__":
    sys.exit(main())
