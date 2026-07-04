#!/usr/bin/env python3
"""bg-be-gone — GTK/libadwaita frontend.

Runs on the system Python. Image processing happens in a persistent worker
subprocess (worker.py) inside the bundled virtualenv, which keeps the model
resident on the GPU. Frontend and worker talk over line-delimited JSON.
"""
import os
import sys
import json
import shutil
import tempfile
import threading
import subprocess

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Gdk, Gio, GLib, Adw  # noqa: E402

from viewer import ImageView  # noqa: E402

APP_ID = "io.github.cristeigabriela.BgBeGone"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PY = (os.environ.get("BGBG_VENV_PYTHON")
           or os.path.expanduser("~/.local/share/bg-be-gone/venv/bin/python"))
WORKER = os.environ.get("BGBG_WORKER", os.path.join(APP_DIR, "worker.py"))
LOG = os.path.join(GLib.get_user_cache_dir(), "bg-be-gone-worker.log")
APP_VERSION = os.environ.get("BGBG_VERSION", "1.0.0")

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
BGS = [
    ("Transparent", "transparent"),
    ("Blur background", "blur"),
    ("White", "#ffffff"),
    ("Black", "#000000"),
    ("Green screen", "#00b140"),
    ("Custom…", "custom"),
]
IMG_PATTERNS = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tif", "*.tiff"]
PROVIDER_LABELS = {
    "CUDAExecutionProvider": "NVIDIA (CUDA)",
    "ROCMExecutionProvider": "AMD (ROCm)",
    "MIGraphXExecutionProvider": "AMD (MIGraphX)",
    "DmlExecutionProvider": "DirectML",
    "CPUExecutionProvider": "CPU",
}

CSS = b"""
.view-frame { background: #1c1c1e; border-radius: 8px; }
.zoom-badge {
  background: alpha(black, 0.55); color: white; border-radius: 6px;
  padding: 2px 8px; margin: 8px; font-size: 0.8em;
}
.panel-title { font-weight: bold; }
"""


class Worker:
    """Owns the venv subprocess and dispatches replies to the main loop."""

    def __init__(self, on_message):
        self._on_message = on_message
        self._next_id = 1
        self.ok = False
        self.error = None
        try:
            self._logf = open(LOG, "a")
            self.proc = subprocess.Popen(
                [VENV_PY, WORKER], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self._logf, text=True, bufsize=1)
            threading.Thread(target=self._reader, daemon=True).start()
            self.ok = True
        except Exception as e:
            self.error = str(e)

    def _reader(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            GLib.idle_add(self._on_message, msg)

    def send(self, req):
        if not self.ok:
            return -1
        rid = self._next_id
        self._next_id += 1
        req["id"] = rid
        try:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        return rid

    def shutdown(self):
        if not self.ok:
            return
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

    def __init__(self, title, on_change=None):
        self._ext_change = on_change
        self._last_pct = -1
        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.box.set_hexpand(True)

        header = Gtk.CenterBox()
        lbl = Gtk.Label(label=title)
        lbl.add_css_class("panel-title")
        header.set_start_widget(lbl)
        tools = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
        for icon, tip, cb in (
            ("object-rotate-left-symbolic", "Rotate left",
             lambda *_: self.view._rotate(-1)),
            ("object-rotate-right-symbolic", "Rotate right",
             lambda *_: self.view._rotate(1)),
            ("object-flip-horizontal-symbolic", "Flip horizontal",
             lambda *_: self.view._flip(True)),
            ("object-flip-vertical-symbolic", "Flip vertical",
             lambda *_: self.view._flip(False)),
            ("zoom-fit-best-symbolic", "Reset view", self._reset),
        ):
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

        self.placeholder = Gtk.Label()
        self.placeholder.set_markup(
            "<span alpha='55%'>Drop an image here\nor click Open</span>")
        self.placeholder.set_justify(Gtk.Justification.CENTER)
        self.placeholder.set_can_target(False)
        overlay.add_overlay(self.placeholder)

        self.box.append(overlay)

    def _reset(self, *_):
        self.view.reset_view()

    def _changed(self):
        has = self.view.has_image()

        def apply():
            self.placeholder.set_visible(not has)
            self.zoom_badge.set_visible(has)
            return False
        GLib.idle_add(apply)
        if self._ext_change:
            self._ext_change()

    def _update_zoom(self, percent):
        # Called from the paint callback; defer the label mutation to idle so we
        # never relayout mid-snapshot.
        if percent != self._last_pct:
            self._last_pct = percent
            GLib.idle_add(self.zoom_badge.set_text, "%d%%" % percent)


class Window(Adw.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app)
        self.set_title("bg-be-gone")
        self.set_default_size(1300, 820)

        self.tmpdir = os.environ.get("BGBG_TMPDIR") or tempfile.mkdtemp(
            prefix="bg-be-gone-")
        os.makedirs(self.tmpdir, exist_ok=True)
        self.source_path = None
        self.result_output = None
        self.busy = False
        self.batch_input = None
        self.batch_output = None
        self._advanced = False
        self._syncing = False
        self._gpu_notified = False
        self.worker = Worker(self._on_worker_message)

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
        menu = Gio.Menu()
        menu.append("About bg-be-gone", "app.about")
        menu_btn = Gtk.MenuButton(icon_name="open-menu-symbolic",
                                  menu_model=menu)
        header.pack_end(menu_btn)
        tv.add_top_bar(header)

        body = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        body.append(self._build_sidebar())
        body.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))
        self.stack.set_hexpand(True)
        body.append(self.stack)
        tv.set_content(body)

        self.stack.add_titled_with_icon(
            self._build_single(), "single", "Single", "image-x-generic-symbolic")
        self.stack.add_titled_with_icon(
            self._build_batch(), "batch", "Batch", "view-grid-symbolic")
        if os.environ.get("BGBG_START_PAGE"):
            self.stack.set_visible_child_name(os.environ["BGBG_START_PAGE"])

        # window-wide drag and drop
        drop = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop.connect("drop", self._on_drop)
        self.add_controller(drop)

        self.connect("close-request", self._on_close)
        if not self.worker.ok:
            self._toast("Could not start the worker. Is it installed?")

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

        model_grp = Adw.PreferencesGroup(title="Model")
        self.subject_row = Adw.ComboRow(
            title="Subject",
            model=Gtk.StringList.new([s[0] for s in SUBJECTS]))
        self.subject_row.set_subtitle("What the image mainly contains")
        self.subject_row.connect("notify::selected", self._on_subject_changed)
        model_grp.add(self.subject_row)

        self.adv_row = Adw.ExpanderRow(title="Advanced model")
        self.model_row = Adw.ComboRow(
            title="Model", model=Gtk.StringList.new([m[0] for m in MODELS]))
        self.model_row.connect("notify::selected", self._on_model_changed)
        self.adv_row.add_row(self.model_row)
        model_grp.add(self.adv_row)
        box.append(model_grp)

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

        self.device_lbl = Gtk.Label(xalign=0, label="Device: detecting…")
        self.device_lbl.add_css_class("dim-label")
        self.device_lbl.set_wrap(True)
        box.append(self.device_lbl)
        return scroller

    def _on_subject_changed(self, *_):
        self._advanced = False
        target = SUBJECTS[self.subject_row.get_selected()][1]
        for i, (_, m) in enumerate(MODELS):
            if m == target:
                self._syncing = True
                self.model_row.set_selected(i)
                self._syncing = False
                break
        self._mark_stale()

    def _on_model_changed(self, *_):
        if self._syncing:
            return
        self._advanced = True
        self.adv_row.set_expanded(True)
        self._mark_stale()

    def _sync_color(self):
        bg = BGS[self.bg_row.get_selected()][1]
        self.color_row.set_sensitive(bg == "custom")
        self.blur_row.set_visible(bg == "blur")
        self._mark_stale()

    def get_settings(self):
        if self._advanced:
            model = MODELS[self.model_row.get_selected()][1]
        else:
            model = SUBJECTS[self.subject_row.get_selected()][1]
        bg = BGS[self.bg_row.get_selected()][1]
        if bg == "custom":
            c = self.color_btn.get_rgba()
            bg = "#%02x%02x%02x" % (round(c.red * 255), round(c.green * 255),
                                    round(c.blue * 255))
        return model, bg, self.alpha_row.get_active(), int(self.blur_row.get_value())

    # ---------- single page ----------
    def _build_single(self):
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        page.set_margin_top(10)
        page.set_margin_bottom(10)
        page.set_margin_start(10)
        page.set_margin_end(10)

        bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.gen_btn = Gtk.Button(label="Generate")
        self.gen_btn.add_css_class("suggested-action")
        self.gen_btn.set_sensitive(False)
        self.gen_btn.connect("clicked", lambda *_: self._on_generate())
        self.save_btn = Gtk.Button(label="Save result…")
        self.save_btn.set_sensitive(False)
        self.save_btn.connect("clicked", lambda *_: self._on_save())
        self.spinner = Gtk.Spinner()
        bar.append(self.gen_btn)
        bar.append(self.save_btn)
        bar.append(self.spinner)
        self.status = Gtk.Label(xalign=0, label="Open or drop an image to begin.")
        self.status.add_css_class("dim-label")
        self.status.set_hexpand(True)
        self.status.set_ellipsize(3)  # PANGO_ELLIPSIZE_END
        bar.append(self.status)
        page.append(bar)

        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL,
                          wide_handle=True)
        paned.set_vexpand(True)
        self.src_panel = Panel("Source", on_change=self._on_source_change)
        self.res_panel = Panel("Result")
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

        self.run_btn = Gtk.Button(label="Run batch")
        self.run_btn.add_css_class("suggested-action")
        self.run_btn.add_css_class("pill")
        self.run_btn.set_halign(Gtk.Align.START)
        self.run_btn.set_sensitive(False)
        self.run_btn.connect("clicked", lambda *_: self._on_run_batch())
        box.append(self.run_btn)

        self.progress = Gtk.ProgressBar(show_text=True)
        box.append(self.progress)
        self.batch_status = Gtk.Label(xalign=0, label="")
        self.batch_status.add_css_class("dim-label")
        self.batch_status.set_wrap(True)
        box.append(self.batch_status)
        return outer

    # ---------- file dialogs ----------
    def _image_filters(self):
        store = Gio.ListStore.new(Gtk.FileFilter)
        f = Gtk.FileFilter(name="Images")
        for p in IMG_PATTERNS:
            f.add_pattern(p)
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

    def _load_source(self, path):
        if not path:
            return
        if not self.src_panel.view.load_file(path):
            self._toast("Could not open that image.")
            return
        self.source_path = path
        self.res_panel.view.clear()
        self.result_output = None
        self.save_btn.set_sensitive(False)
        self.gen_btn.set_sensitive(True)
        self.status.set_text(os.path.basename(path) + " — press Generate.")

    def _on_save(self):
        pb = self.res_panel.view.export_pixbuf()
        if pb is None:
            return
        dlg = Gtk.FileDialog(title="Save result")
        base = os.path.splitext(os.path.basename(self.source_path or "image"))[0]
        dlg.set_initial_name(base + "_nobg.png")

        def done(d, res):
            try:
                f = d.save_finish(res)
            except GLib.Error:
                return
            path = f.get_path()
            if not path.lower().endswith(".png"):
                path += ".png"
            try:
                pb.savev(path, "png", [], [])
                self._toast("Saved " + os.path.basename(path))
            except Exception as e:
                self._toast("Save failed: %s" % e)
        dlg.save(self, None, done)

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
            bool(self.batch_input and self.batch_output) and not self.busy)

    # ---------- actions ----------
    def _set_busy(self, busy, spin=True):
        self.busy = busy
        self.gen_btn.set_sensitive(not busy and self.src_panel.view.has_image())
        self._sync_batch()
        if spin:
            (self.spinner.start if busy else self.spinner.stop)()

    def _on_source_change(self):
        if self.src_panel.view.has_image() and self.result_output:
            self.status.set_text("Source changed — press Generate to update.")

    def _mark_stale(self, *_):
        if getattr(self, "result_output", None):
            self.status.set_text("Settings changed — press Generate to update.")

    def _on_generate(self):
        if self.busy or not self.src_panel.view.has_image():
            return
        model, bg, alpha, blur = self.get_settings()
        inp = os.path.join(self.tmpdir, "input.png")
        try:
            self.src_panel.view.export_pixbuf().savev(inp, "png", [], [])
        except Exception as e:
            self._toast("Could not prepare image: %s" % e)
            return
        out = os.path.join(self.tmpdir, "result.png")
        self._set_busy(True)
        self.status.set_text("Generating…")
        self.worker.send({"op": "single", "input": inp, "output": out,
                          "model": model, "alpha": alpha, "bg": bg, "blur": blur})

    def _on_run_batch(self):
        if self.busy or not (self.batch_input and self.batch_output):
            return
        model, bg, alpha, blur = self.get_settings()
        pattern = self.pattern_row.get_text().strip() or "{name}_nobg"
        self._set_busy(True, spin=False)
        self.progress.set_fraction(0)
        self.progress.set_text("starting…")
        self.batch_status.set_text("Preparing…")
        self.worker.send({"op": "batch", "input_dir": self.batch_input,
                          "output_dir": self.batch_output, "model": model,
                          "alpha": alpha, "bg": bg, "blur": blur,
                          "pattern": pattern})

    # ---------- worker replies ----------
    def _on_worker_message(self, msg):
        t = msg.get("type")
        if t == "ready":
            provs = msg.get("providers") or []
            if provs:
                self.device_lbl.set_text(
                    "Device: " + PROVIDER_LABELS.get(provs[0], provs[0]))
        elif t == "loading":
            self.status.set_text(
                "Loading %s… (the first run is slower)" % msg["model"])
            self.batch_status.set_text("Loading model %s…" % msg["model"])
        elif t == "device":
            self.device_lbl.set_text("Device: " + msg["label"])
            if msg.get("gpu") and not self._gpu_notified:
                self._gpu_notified = True
                self._toast("%s ready — the first run optimises GPU kernels and "
                            "is slower; later runs are fast." % msg["label"])
        elif t == "done_single":
            self.res_panel.view.load_file(msg["output"])
            self.result_output = msg["output"]
            self.save_btn.set_sensitive(True)
            self.status.set_text("Done in %.2fs." % msg["seconds"])
            self._set_busy(False)
        elif t == "progress":
            total = max(msg["total"], 1)
            self.progress.set_fraction(msg["done"] / total)
            self.progress.set_text("%d / %d" % (msg["done"], total))
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
        elif t == "error":
            first = msg["message"].splitlines()[0]
            self.status.set_text("Error: " + first)
            self.batch_status.set_text("Error: " + first)
            self._toast("Error: " + first)
            self._set_busy(False)
            self._set_busy(False, spin=False)
        return False

    def _toast(self, text):
        self.toasts.add_toast(Adw.Toast(title=text, timeout=3))

    def _on_close(self, *_):
        self.worker.shutdown()
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        return False


class App(Adw.Application):
    def __init__(self):
        super().__init__(application_id=APP_ID,
                         flags=Gio.ApplicationFlags.HANDLES_OPEN)
        self._pending_open = None

    def do_startup(self):
        Adw.Application.do_startup(self)
        provider = Gtk.CssProvider()
        provider.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        about = Gio.SimpleAction.new("about", None)
        about.connect("activate", self._on_about)
        self.add_action(about)

    def do_activate(self):
        win = Window(self)
        if self._pending_open:
            win._load_source(self._pending_open)
            self._pending_open = None
        win.present()
        # Env-gated demo helper: auto-run Generate once, for screenshots/tests.
        if os.environ.get("BGBG_AUTOGEN") and win.src_panel.view.has_image():
            GLib.timeout_add(1400, lambda: (win._on_generate(), False)[1])

    def do_open(self, files, n, hint):
        if files:
            self._pending_open = files[0].get_path()
        self.do_activate()

    def _on_about(self, *_):
        about = Adw.AboutWindow(
            application_name="bg-be-gone",
            application_icon="io.github.cristeigabriela.BgBeGone",
            developer_name="bg-be-gone",
            version=APP_VERSION,
            comments="Local image background remover (BiRefNet / rembg).",
            license_type=Gtk.License.MIT_X11,
            website="https://github.com/cristeigabriela/bg-be-gone")
        about.set_transient_for(self.get_active_window())
        about.present()


def main():
    return App().run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
