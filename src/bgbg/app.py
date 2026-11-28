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
import formats  # noqa: E402
from engines import selection  # noqa: E402  (standard library only)
import offmain  # noqa: E402

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


# Reply types that belong to one request; any other reply is window-wide state.
_JOB_REPLIES = {
    "loading", "done_single", "gif_progress", "gif_done", "canceled", "progress",
    "done_batch", "converted", "frames", "seg_ready", "seg_download", "seg_step",
    "seg_progress", "seg_objects", "seg_mask", "seg_extracted", "seg_canceled", "error",
}
# Reply types that end their request.
_JOB_ENDS = {
    "done_single", "gif_done", "canceled", "done_batch", "converted", "frames",
    "seg_ready", "seg_objects", "seg_mask", "seg_extracted", "seg_canceled", "error",
}
_SEG_JOBS = {Job.SEG_LOAD, Job.SEG_RUN, Job.SEG_PREVIEW, Job.SEG_SAVE}


class Abandoned(enum.Enum):
    """Whether Window._abandon_jobs found any of the jobs it was given."""

    SOME = "some"
    NONE = "none"


class Prep(enum.Enum):
    """Whose input file is being encoded, before its request can be sent."""

    SINGLE = "single"            # Generate's input
    SEGMENT = "segment"          # Segment's model load


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
    loads: Optional[tuple] = None            # SEG_LOAD: the (source, model) it loads


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
MODELS = [
    ("BiRefNet — General", "birefnet-general"),
    ("BiRefNet — General Lite", "birefnet-general-lite"),
    ("BiRefNet — Massive", "birefnet-massive"),
    ("BiRefNet — Portrait", "birefnet-portrait"),
    ("BiRefNet — HRSOD", "birefnet-hrsod"),
    ("BiRefNet — DIS", "birefnet-dis"),
    ("ISNet — General", "isnet-general-use"),
    ("ISNet — Anime", "isnet-anime"),
    ("U2Net", "u2net"),
    ("U2Net — Human Seg", "u2net_human_seg"),
    ("Silueta", "silueta"),
]
# Model > Advanced > Engine: what runs BiRefNet. Automatic leaves it to the
# install's profile, then ONNX Runtime (engines/selection.py).
ENGINES = [
    ("Automatic", selection.Engine.AUTO),
    ("ONNX Runtime", selection.Engine.ORT),
    ("PyTorch", selection.Engine.TORCH),
]
BGS = [
    ("Transparent", "transparent"),
    ("Blur background", "blur"),
    ("White", "#ffffff"),
    ("Black", "#000000"),
    ("Green screen", "#00b140"),
    ("Custom…", "custom"),
]
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
"""


def primary_button(label, icon=None):
    """The one bold, rounded call-to-action button (Generate / Segment / Run)."""
    b = Gtk.Button()
    if icon:
        b.set_child(Adw.ButtonContent(icon_name=icon, label=label))
    else:
        b.set_label(label)
    b.add_css_class("suggested-action")
    b.add_css_class("pill")
    return b


def pill_button(label):
    b = Gtk.Button(label=label)
    b.add_css_class("pill")
    return b


def save_button(label="Save…"):
    """Consistent Save button (icon + label), disabled until there's something."""
    b = Gtk.Button()
    b.set_child(Adw.ButtonContent(icon_name="document-save-symbolic", label=label))
    b.add_css_class("pill")
    b.set_sensitive(False)
    return b


def action_bar(primary=None, secondary=(), cancel=None, save=None, status_text=""):
    """Standard action row with fixed slots so Save is always far-right and the
    layout is identical on every page. Returns (bar, status_label, spinner).

    The status label is intentionally NOT shown here — status text lives only in
    the window footer (statusbar) to avoid showing the same message twice. The
    label object is still returned so callers `set_text` on it and the footer
    mirror picks it up."""
    bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    bar.add_css_class("action-bar")
    if primary is not None:
        bar.append(primary)
    for b in secondary:
        bar.append(b)
    spinner = Gtk.Spinner()
    spinner.set_size_request(18, 18)   # reserve space so start/stop never reflows
    spinner.set_valign(Gtk.Align.CENTER)
    bar.append(spinner)
    status = Gtk.Label(xalign=0, label=status_text)   # detached: footer shows it
    spacer = Gtk.Box()
    spacer.set_hexpand(True)           # keep Save pinned far-right
    bar.append(spacer)
    if cancel is not None:
        bar.append(cancel)
    if save is not None:
        bar.append(save)
    return bar, status, spinner


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
        self._encoded_inputs = set()   # paths _prepare_input wrote, the only ones it deletes
        self.busy = False
        self.batch_input = None
        self.batch_output = None
        self._gpu_notified = False
        # segmentation state
        self._seg_available = False
        self._seg_models = []          # [{"rung","label"}] from the worker
        self.seg_loaded_for = None     # (source path, model asked for) the worker holds
        self.seg_pending = None        # callback to run once seg_ready arrives
        self.seg_busy = False
        self.seg_objects = []
        self.seg_points = []           # accumulated [x, y, label] in point mode
        self.seg_point_mask = None     # last point-mode mask path
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
        body.append(self._build_sidebar())
        body.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))
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

        # keyboard shortcuts (Ctrl+S save, Ctrl+Return primary action)
        sc = Gtk.ShortcutController()
        sc.set_scope(Gtk.ShortcutScope.MANAGED)
        sc.add_shortcut(Gtk.Shortcut.new(
            Gtk.ShortcutTrigger.parse_string("<Control>s"),
            Gtk.CallbackAction.new(lambda *_: self._accel_save())))
        sc.add_shortcut(Gtk.Shortcut.new(
            Gtk.ShortcutTrigger.parse_string("<Control>Return"),
            Gtk.CallbackAction.new(lambda *_: self._accel_primary())))
        self.add_controller(sc)

        self.connect("close-request", self._on_close)
        if not self.worker.ok:
            self._show_worker_down("Could not start the background process (%s)."
                                   % self.worker.error)

    # ---------- sidebar ----------
    def _build_sidebar(self):
        scroller = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER)
        scroller.set_size_request(280, -1)
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
        self.subject_row = Adw.ComboRow(
            title="Subject",
            model=Gtk.StringList.new([s[0] for s in SUBJECTS] + ["Custom model…"]))
        self.subject_row.set_subtitle("What the image mainly contains")
        self.subject_row.connect("notify::selected", self._on_subject_changed)
        model_grp.add(self.subject_row)

        self.model_row = Adw.ComboRow(
            title="Model", model=Gtk.StringList.new([m[0] for m in MODELS]))
        self.model_row.set_subtitle("Exact rembg model")
        self.model_row.set_visible(False)
        self.model_row.connect("notify::selected", lambda *_: self._mark_stale())
        model_grp.add(self.model_row)

        # Shown only when the worker says it can run PyTorch, so an install
        # without it looks as it always did.
        self.model_adv = Adw.ExpanderRow(title="Advanced")
        self.engine_row = Adw.ComboRow(
            title="Engine", model=Gtk.StringList.new([e[0] for e in ENGINES]))
        self.engine_row.set_subtitle("What runs BiRefNet")
        self._engine_handler = self.engine_row.connect("notify::selected", self._on_engine_changed)
        self.model_adv.add_row(self.engine_row)
        self.model_adv.set_visible(False)
        model_grp.add(self.model_adv)
        self.model_grp = model_grp
        box.append(model_grp)
        box.append(self._build_seg_group())

        out_grp = Adw.PreferencesGroup(title="Output")
        self.bg_row = Adw.ComboRow(
            title="Background",
            model=Gtk.StringList.new([b[0] for b in BGS]))
        self.bg_row.connect("notify::selected", lambda *_: self._sync_color())
        out_grp.add(self.bg_row)

        self.color_row = Adw.ActionRow(title="Custom colour")
        self.color_btn = Gtk.ColorDialogButton.new(Gtk.ColorDialog())
        rgba = Gdk.RGBA()
        rgba.parse("#00b140")
        self.color_btn.set_rgba(rgba)
        self.color_btn.connect("notify::rgba", self._seg_bg_changed)
        self.color_btn.set_valign(Gtk.Align.CENTER)
        self.color_row.add_suffix(self.color_btn)
        self.color_row.set_sensitive(False)
        out_grp.add(self.color_row)

        self.blur_row = Adw.SpinRow(
            title="Blur strength",
            adjustment=Gtk.Adjustment(value=20, lower=2, upper=80,
                                      step_increment=1, page_increment=5))
        self.blur_row.set_visible(False)
        out_grp.add(self.blur_row)

        self.alpha_row = Adw.SwitchRow(
            title="Alpha matting", subtitle="Cleaner edges, a little slower")
        out_grp.add(self.alpha_row)
        box.append(out_grp)
        return scroller

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
        self.footer_progress.set_show_text(True)
        self.footer_progress.set_visible(False)
        centre.append(self.footer_progress)
        bar.set_center_widget(centre)

        self.footer_context = Gtk.Label(label="")
        self.footer_context.add_css_class("dim-label")
        self.footer_context.add_css_class("status-label")
        bar.set_end_widget(self.footer_context)
        return bar

    def _footer_progress(self, done, total):
        """Show frame/step progress in the shared footer bar (GIF + segmentation)."""
        total = max(int(total), 1)
        self.footer_progress.set_fraction(min(1.0, done / total))
        self.footer_progress.set_text("%d / %d" % (done, total))
        self.footer_progress.set_visible(True)

    def _hide_footer_progress(self):
        if getattr(self, "footer_progress", None) is not None:
            self.footer_progress.set_visible(False)

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
            it = self.seg_model_row.get_selected_item()
            model = it.get_string() if it is not None else ""
            mode = "Click" if self._seg_mode() == "point" else "Everything"
            return "Segment · %s · %s" % (mode, model) if model else "Segment · %s" % mode
        if page == "batch":
            return "Batch"
        i = self.subject_row.get_selected()
        name = (MODELS[self.model_row.get_selected()][0] if i >= len(SUBJECTS)
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
        self.model_adv.set_visible(shown)

    def _on_engine_changed(self, *_):
        label, engine = ENGINES[self.engine_row.get_selected()]
        try:
            selection.save_setting(engine, self._settings_path)
        except OSError as err:
            self._toast("Couldn't save the engine setting: %s" % err)
            return
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
        self._mark_stale()
        if getattr(self, "footer_context", None) is not None:
            self._update_footer()

    def _sync_color(self):
        bg = BGS[self.bg_row.get_selected()][1]
        self.color_row.set_sensitive(bg == "custom")
        self.blur_row.set_visible(bg == "blur")
        self._mark_stale()
        self._seg_bg_changed()

    def _seg_bg_changed(self, *_):
        if getattr(self, "seg_panel", None) is not None and \
                self.stack.get_visible_child_name() == "segment":
            self._update_seg_preview()

    def get_settings(self):
        i = self.subject_row.get_selected()
        if i >= len(SUBJECTS):                      # "Custom model…"
            model = MODELS[self.model_row.get_selected()][1]
        else:
            model = SUBJECTS[i][1]
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
        self.seg_model_row = Adw.ComboRow(
            title="Model", model=Gtk.StringList.new(["Auto"]))
        self.seg_model_row.set_subtitle("Auto — by GPU / VRAM")
        self.seg_model_row.connect("notify::selected", self._on_seg_model_changed)
        self.seg_adv.add_row(self.seg_model_row)

        self.seg_detail_row = Adw.ComboRow(
            title="Detail", model=Gtk.StringList.new([d[0] for d in SEG_DETAIL]))
        self.seg_detail_row.set_subtitle("Finer finds more, but is slower")
        self.seg_detail_row.connect("notify::selected", lambda *_: None)
        self.seg_adv.add_row(self.seg_detail_row)
        grp.add(self.seg_adv)
        grp.set_visible(False)
        return grp

    def _vram_str(self, mb):
        if not mb:
            return "CPU · low VRAM"
        return ("~%.1f GB" % (mb / 1000)).replace(".0", "")

    def _apply_seg_models(self):
        names = ["Auto — best for your GPU"]
        for m in self._seg_models:
            names.append("%s · %s" % (m["label"], self._vram_str(m.get("vram", 0))))
        self._syncing_seg = True
        self.seg_model_row.set_model(Gtk.StringList.new(names))
        self.seg_model_row.set_selected(0)
        self._syncing_seg = False

    def _seg_model_choice(self):
        i = self.seg_model_row.get_selected()
        if i <= 0 or i > len(self._seg_models):
            return "auto"
        return self._seg_models[i - 1]["rung"]

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
        cached encode + overlays and invalidate any in-flight load so a later
        pass re-encodes cleanly."""
        if scope == "seg":
            self.seg_loaded_for = None
            self.seg_pending = None
            self._abandon_jobs({Job.SEG_LOAD})  # its reply would re-mark it loaded
            if getattr(self, "seg_panel", None) is not None:
                self._seg_clear_overlays()
        if self.worker.ok:
            self.worker.send({"op": "unload", "scope": scope})

    def _sync_sidebar_page(self):
        seg = self.stack.get_visible_child_name() == "segment"
        self.model_grp.set_visible(not seg)
        self.seg_grp.set_visible(seg and self._seg_available)
        self.alpha_row.set_visible(not seg)

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
            self._on_seg_everything()
        elif p == "batch":
            self._on_run_batch()
        return True

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
            title="Tip: hold Space to pan · click to keep objects · Ctrl-click removes")
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
        self._seg_clear_overlays()
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

    def _seg_clear_overlays(self):
        self.seg_panel.view.clear_seg(keep_mode=True)
        self.seg_objects = []
        self.seg_points = []
        self.seg_point_mask = None
        self.seg_result_output = None
        self.seg_res_panel.view.clear()
        self.seg_save_btn.set_sensitive(False)
        if not self.seg_busy:
            self.seg_status.set_text("Cleared.")

    def _update_seg_selection_ui(self):
        self._update_seg_preview()
        n = len(self.seg_panel.view.get_seg_selection())
        if not self.seg_busy:
            self._seg_base_status = (
                "%d selected — Save selection." % n if n
                else "Click objects to keep them (they light up on hover).")
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
        req = {"op": "seg_extract", "bg": bg, "blur": blur, "output": outp}
        if self._seg_mode() == "point":
            req["mask"] = self.seg_point_mask
        else:
            req["ids"] = list(self.seg_panel.view.get_seg_selection())
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
        self._seg_set_busy(True)
        self.seg_status.set_text("Preparing export…")
        req = self._seg_extract_req(os.path.join(self.tmpdir, "seg_result.png"))
        self._send_job(Job.SEG_SAVE, req, alpha=_alpha_of(req["bg"]))

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
        self.seg_panel.view.set_seg_mode(mode)
        self._seg_clear_overlays()
        self.seg_everything_btn.set_visible(mode == "everything")
        self.seg_selall_btn.set_visible(mode == "everything")
        if mode == "point":
            self.seg_status.set_text(
                "Click an object (Ctrl-click or right-click removes).")
        else:
            self.seg_status.set_text(
                "Click Segment everything, then pick the objects to keep.")
        self._update_footer()

    def _on_seg_model_changed(self, *_):
        if getattr(self, "_syncing_seg", False):
            return
        self.seg_loaded_for = None  # force re-encode with the new model
        self.seg_adv.set_subtitle("Reloads on next segment")
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
        paned.set_vexpand(True)
        self.src_panel = Panel("Source", on_change=self._on_source_change)
        self.res_panel = Panel(
            "Result", empty_icon="emblem-photos-symbolic",
            empty_title="Your cut-out appears here", empty_hint="Press Generate")
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
        grp.add(self.pattern_row)
        box.append(grp)

        hint = Gtk.Label(xalign=0, wrap=True)
        hint.add_css_class("dim-label")
        hint.set_markup(
            "<small>Tokens: <tt>{name}</tt> original filename, "
            "<tt>{n}</tt> index. Output is PNG in the output folder.</small>")
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

        def done(d, res):
            try:
                f = d.open_finish(res)
            except GLib.Error:
                return
            self._load_source(f.get_path())
        dlg.open(self, None, done)

    def _on_drop(self, target, value, x, y):
        try:
            files = value.get_files()
        except Exception:
            return False
        if files and files[0].get_path():
            self._load_source(files[0].get_path())
            self.stack.set_visible_child_name("single")
            return True
        return False

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
        self.seg_save_btn.set_sensitive(self.seg_result_output is not None)

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
        self.result_output = None
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
            self._seg_clear_overlays()
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
            self._ask_save_path("_nobg", (fmt,), lambda target, choices: self._save_result(
                target, choices, lambda path, _: shutil.copyfile(result, path), fmt))
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

    def _save_result(self, target, choices, render, rendered):
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
            if isinstance(result, (OSError, GLib.Error)):
                self._report_save_failure(target, result)
            elif isinstance(result, Exception):
                raise result                 # a bug, not a save failure
            elif direct:
                self._report_saved(target, choices)
            else:
                self._save_via_worker(result, target, choices, staging=result)
        offmain.run(work, written)

    def _ask_save_path(self, stem_suffix, choices, save):
        """Ask where to save, offering `choices`, then call save(target, choices).

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

        def done(d, res):
            try:
                f = d.save_finish(res)
            except GLib.Error:
                return
            target = formats.save_target(f.get_path(), choices, default)
            if formats.overwrites_unasked(target):
                self._confirm_overwrite(target, lambda: save(target, choices))
            else:
                save(target, choices)
        dlg.save(self, None, done)

    def _confirm_overwrite(self, target, save):
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
        dlg.connect("response", answered)
        dlg.present(self)

    def _report_saved(self, target, choices):
        self._toast(_saved_notice(target, choices))

    def _report_save_failure(self, target, exc):
        self._report_failure("saving " + target.path, exc, "Could not save %s: %s." % (
            os.path.basename(target.path), plain_reason(exc)))

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

    def _on_save_reply(self, pending, msg):
        _remove_quietly(pending.staging)
        if msg["type"] == "error":
            _remove_quietly(pending.part)
            _log.warning("saving %s failed: %s", pending.target.path, msg["message"])
            self._toast_error("Could not save %s: %s" % (
                os.path.basename(pending.target.path), msg["message"]))
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

        def done(d, res):
            try:
                f = d.select_folder_finish(res)
            except GLib.Error:
                return
            self.batch_input = f.get_path()
            self.in_row.set_subtitle(self.batch_input)
            self._sync_batch()
        dlg.select_folder(self, None, done)

    def _on_pick_output(self, *_):
        dlg = Gtk.FileDialog(title="Output folder")

        def done(d, res):
            try:
                f = d.select_folder_finish(res)
            except GLib.Error:
                return
            self.batch_output = f.get_path()
            self.out_row.set_subtitle(self.batch_output)
            self._sync_batch()
        dlg.select_folder(self, None, done)

    def _sync_batch(self):
        self.run_btn.set_sensitive(
            bool(self.batch_input and self.batch_output) and not self.busy
            and self._worker_state is WorkerState.UP)

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
        model, bg, alpha, blur = self.get_settings()
        if self._source_animation is formats.Animation.ANIMATED:
            self._generate_animation(model, bg, alpha, blur)
            return
        out = os.path.join(self.tmpdir, "result.png")
        self._set_busy(True, cancel=True)
        self.src_panel.view.set_scanning(True)      # scan shimmer over the source
        self.status.set_text("Generating…")

        def send(inp):
            # The input is deleted once the job ends, if we encoded it.
            self._send_job(Job.SINGLE, {
                "op": "single", "input": inp, "output": out,
                "model": model, "alpha": alpha, "bg": bg, "blur": blur},
                alpha=_alpha_of(bg), input=inp)

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
        out = os.path.join(self.tmpdir, "result" + fmt.suffix)
        v = self.src_panel.view
        self._set_busy(True, cancel=True)
        self.src_panel.view.set_scanning(True)
        self.status.set_text("Removing background from %s…" % fmt.name)
        self._send_job(Job.SINGLE, {
            "op": "gif", "input": self.source_path, "output": out,
            "model": model, "alpha": alpha, "bg": bg, "blur": blur,
            "rot": v.rot, "fh": v.fh, "fv": v.fv}, alpha=_alpha_of(bg))

    def _on_bg_cancel(self):
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
                self._remove_temp_input(self._abandoned.pop(rid).input)
            return False
        if t in _JOB_ENDS:
            req = self._jobs.pop(rid, None)
            if req is not None:
                self._remove_temp_input(req.input)
                # An Engine change waits for every job; after this reply is handled.
                GLib.idle_add(self._replace_worker_when_idle)
        else:
            req = self._jobs.get(rid)
        job = req.job if req is not None else None
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
            self._offer_engines(msg.get("engines") or ["ort"], msg.get("settings"))
            provs = msg.get("providers") or []
            if provs:
                self._set_device(provs[0],
                                 PROVIDER_LABELS.get(provs[0], provs[0]))
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
            if job is Job.BATCH:
                self.batch_status.set_text("Loading model %s…" % msg["model"])
            else:
                self.status.set_text(
                    "Loading %s… (the first run is slower)" % msg["model"])
        elif t == "notice":
            self._toast(msg["message"])
        elif t == "device":
            self._set_device(msg["provider"], msg["label"])
            if msg.get("gpu") and not self._gpu_notified:
                self._gpu_notified = True
                self._toast("%s ready — the first run optimises GPU kernels and "
                            "is slower; later runs are fast." % msg["label"])
        elif t == "done_single":
            self.result_output = msg["output"]
            self._result_alpha = req.alpha
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
            self.result_output = msg["output"]
            self._result_alpha = req.alpha
            self.save_btn.set_sensitive(True)
            self.status.set_text("Done: %d frames in %.1fs — Save the %s."
                                 % (msg["frames"], msg["seconds"],
                                    formats.format_of(msg["output"]).name))
            self._set_busy(False)
        elif t == "canceled":
            self._set_busy(False, spin=(msg.get("scope") != "batch"))
            if msg.get("scope") == "batch":
                self.progress.set_text("cancelled")
                self.batch_status.set_text(
                    "Cancelled after %d / %d." % (msg.get("done", 0),
                                                  msg.get("total", 0)))
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
            self.batch_status.set_text(
                "Finished %d images in %.1fs → %s" %
                (msg["count"], msg["seconds"], msg["outdir"]))
            self._set_busy(False, spin=False)
            self._toast("Batch finished: %d images" % msg["count"])
        elif t == "seg_ready":
            self._seg_set_busy(False)
            self.seg_loaded_for = req.loads
            note = {"auto": " · auto", "fallback": " · fallback"}.get(
                msg.get("mode"), "")
            self.seg_model_row.set_subtitle(msg["model"] + note)
            # When on "Auto", tell the user which model that resolved to.
            if self.seg_model_row.get_selected() <= 0:
                self.seg_model_row.set_tooltip_text(
                    "Auto selected: %s on %s" % (msg["model"], msg["label"]))
            self._set_device(msg["provider"], msg["label"])
            cb, self.seg_pending = self.seg_pending, None
            if cb:
                cb()
            else:
                self.seg_status.set_text("Ready: %s on %s." %
                                         (msg["model"], msg["label"]))
        elif t == "seg_download":
            total = max(msg.get("total", 0), 1)
            self.seg_status.set_text("Downloading model… %d%%" %
                                     (100 * msg["done"] // total))
        elif t == "seg_step":
            self.seg_status.set_text(
                "%s unavailable — trying a lighter model…" % msg["rung"])
        elif t == "seg_progress":
            # The scan shimmer conveys activity; keep a calm static label, but
            # also drive the shared footer progress bar.
            self.seg_status.set_text("Finding objects…")
            self._footer_progress(msg["done"], msg["total"])
        elif t == "seg_objects":
            self.seg_objects = msg["objects"]
            status = "%d objects — hover to highlight, click to keep." % msg["count"]

            def shown(loaded):
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
            self.seg_panel.view.set_point_mask(msg["mask"], msg.get("contour"))
            self._update_seg_preview()
            self.seg_status.set_text(
                "Object selected (%.2f) — Save, or click to refine." %
                msg["score"])
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
            self._seg_set_busy(False)
            self._unload("seg")            # free the SAM model + cached encode
            self.seg_status.set_text("Cancelled.")
        elif t == "error":
            self._on_job_error(job, msg["message"])
        return False

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
            ("document-open-symbolic", "Open or drop an image",
             "Drag a file in, or click Open in the header."),
            ("applications-graphics-symbolic", "Pick a subject or model",
             "General, portrait, anime — or an exact model under Advanced."),
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
            if req.job is Job.SAVE:
                self._on_save_reply(req.save, {
                    "type": "error", "message": "the background process stopped."})

    def _restart_worker(self):
        self.worker.shutdown()
        # Before the new worker exists, so none of the old requests waits on it.
        self._end_requests_in_flight()
        self._start_worker(first_id=self.worker.next_id)
        if not self.worker.ok:
            self._show_worker_down("Could not start the background process (%s)."
                                   % self.worker.error)
            return
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
        self.worker.shutdown()
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        return False


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
