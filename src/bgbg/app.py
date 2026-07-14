#!/usr/bin/env python3
"""bg-be-gone — GTK/libadwaita frontend.

Runs on the system Python. Image processing happens in a persistent worker
subprocess (compute/service.py) inside the bundled virtualenv, which keeps the model
resident on the GPU. Frontend and worker talk over line-delimited JSON.
"""
import os
import sys
import shutil
import tempfile

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Gdk, GdkPixbuf, Gio, GLib, Adw  # noqa: E402

from shell_gtk.canvas import ImageView  # noqa: E402
from shell_gtk.sidebar import Sidebar  # noqa: E402
from shell_gtk.worker_port import WorkerPort  # noqa: E402
from engine import interactables as IX  # noqa: E402
from engine import outputs as OUT  # noqa: E402

APP_ID = "io.github.cristeigabriela.BgBeGone"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PY = (os.environ.get("BGBG_VENV_PYTHON")
           or os.path.expanduser("~/.local/share/bg-be-gone/venv/bin/python"))
WORKER = os.environ.get("BGBG_WORKER",
                        os.path.join(APP_DIR, "compute", "service.py"))
LOG = os.path.join(GLib.get_user_cache_dir(), "bg-be-gone-worker.log")
APP_VERSION = os.environ.get("BGBG_VERSION", "1.1.0")

# Every setting the user can touch — the models, the backgrounds, the
# segmentation knobs — is declared once in engine/interactables.py, and the
# sidebar below is built from it. Nothing here knows what a "model" is.
IMG_PATTERNS = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tif", "*.tiff",
                "*.gif"]
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
        # Every user-facing setting, and its current value. The sidebar is built
        # from this; nothing in this file holds a list of models or backgrounds.
        self.settings = IX.new_settings()
        self.source_path = None
        self._source_is_gif = False
        self.result_output = None
        self.busy = False
        self.batch_input = None
        self.batch_output = None
        self._gpu_notified = False
        # segmentation state
        self._seg_available = False
        self._seg_models = []          # [{"rung","label"}] from the worker
        self.seg_loaded_for = None     # source path currently encoded
        self.seg_pending = None        # callback to run once seg_ready arrives
        self._seg_load_rid = None      # id of the in-flight seg_load, or None
        self.seg_busy = False
        self.seg_objects = []
        self.seg_points = []           # accumulated [x, y, label] in point mode
        self.seg_point_mask = None     # last point-mode mask path
        self.seg_result_output = None
        self._seg_rows = {}            # object id -> layer-list row widgets
        self.worker = WorkerPort(VENV_PY, WORKER, self._on_worker_message,
                                 log=LOG)

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
            self._toast("Could not start the worker. Is it installed?")

    # ---------- sidebar ----------
    def _build_sidebar(self):
        scroller = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER)
        scroller.set_size_request(280, -1)
        self.sidebar_root = scroller
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        box.set_margin_top(16)
        box.set_margin_bottom(16)
        box.set_margin_start(12)
        box.set_margin_end(12)
        scroller.set_child(box)

        # The whole sidebar is a walk over the engine's schema: every group,
        # row, option, default and visibility rule comes from
        # engine/interactables.py, so the browser's sidebar can be the same walk
        # over the same declaration.
        self.sidebar = Sidebar(self.settings, on_change=self._on_setting_changed)
        for grp in self.sidebar.widgets:
            box.append(grp)

        # Named handles for the rows the rest of the window still talks to.
        r = self.sidebar.rows
        self.subject_row = r["subject"]
        self.model_row = r["model"]
        self.bg_row = r["bg"]
        self.color_row = r["bg_color"]
        self.color_btn = self.sidebar.buttons["bg_color"]
        self.blur_row = r["blur"]
        self.alpha_row = r["alpha"]
        self.seg_focus_row = r["seg_focus"]
        self.seg_adv = r["seg_advanced"]
        self.seg_model_row = r["seg_model"]
        self.seg_detail_row = r["seg_detail"]
        self.model_grp = self.sidebar.groups["model"]
        self.seg_grp = self.sidebar.groups["seg"]
        return scroller

    def _on_setting_changed(self, sid):
        """A setting the user actually changed. Visibility already re-applied
        itself from the schema; this is only the consequences beyond the row."""
        if sid == "subject":
            self._mark_stale()
            if getattr(self, "footer_context", None) is not None:
                self._update_footer()
        elif sid == "model":
            self._mark_stale()
        elif sid == "bg":
            self._mark_stale()
            self._seg_bg_changed()
        elif sid == "bg_color":
            self._seg_bg_changed()
        elif sid == "seg_focus":
            if getattr(self, "seg_panel", None) is not None:
                self.seg_panel.view.set_dwell_ms(IX.dwell_ms(self.settings))
        elif sid == "seg_model":
            self._on_seg_model_changed()

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
        st = self.settings
        if page == "segment":
            model = st.label("seg_model")
            mode = "Click" if self._seg_mode() == "point" else "Everything"
            return "Segment · %s · %s" % (mode, model) if model else "Segment · %s" % mode
        if page == "batch":
            return "Batch"
        name = (st.label("model") if st.get("subject") == "custom"
                else st.label("subject"))
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

    def _seg_bg_changed(self, *_):
        if getattr(self, "seg_panel", None) is not None and \
                self.stack.get_visible_child_name() == "segment":
            self._update_seg_preview()

    def get_settings(self):
        st = self.settings
        return (IX.model_id(st), IX.background(st),
                IX.alpha_matting(st), IX.blur_strength(st))

    # ---------- segmentation sidebar + page ----------
    def _apply_seg_models(self):
        """The worker told us which SAM models this GPU can actually hold."""
        self.settings.set_options("seg_model",
                                  IX.seg_model_choices(self._seg_models))
        self.sidebar.reload_options("seg_model")

    def _seg_model_choice(self):
        return IX.seg_model(self.settings)

    def _seg_detail(self):
        return IX.seg_detail(self.settings)

    def _seg_mode(self):
        return IX.seg_mode(self.settings)

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

    def _release_domain(self, domain):
        if domain == "bg":
            self._unload("bg")
            return
        # Stop any in-flight pass, then unload unconditionally. We can't rely on a
        # seg_canceled reply — only "segment everything" is cancellable; load /
        # point / extract finish normally — so unload authoritatively here.
        if self.seg_busy:
            self.worker.send({"op": "seg_cancel"})
        self._unload("seg")

    def _unload(self, scope):
        """Ask the worker to release a model. For segmentation, also drop the
        cached encode + overlays and invalidate any in-flight load so a later
        pass re-encodes cleanly."""
        if scope == "seg":
            self.seg_loaded_for = None
            self.seg_pending = None
            self._seg_load_rid = None      # ignore a load reply that's now stale
            if getattr(self, "seg_panel", None) is not None:
                self._seg_clear_overlays()
        if self.worker.ok:
            self.worker.send({"op": "unload", "scope": scope})

    def _sync_sidebar_page(self):
        # Which page is showing and whether the worker has SAM are the only two
        # things the schema's visibility rules need from the app.
        self.settings.set_context(page=self.stack.get_visible_child_name(),
                                  seg_available=self._seg_available)
        self.sidebar.sync()

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
        # Page furniture, but the same catalogue: the modes are declared once,
        # in the engine, and rendered here as a segmented control rather than a
        # sidebar row (engine.settings.PAGE).
        self.seg_mode_toggle = Adw.ToggleGroup()
        for c in self.settings.options("seg_mode"):
            self.seg_mode_toggle.add(Adw.Toggle(name=c.value, label=c.label))
        self.seg_mode_toggle.set_active_name(self.settings.get("seg_mode"))
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
        self.seg_everything_btn.set_sensitive(not busy)
        self.seg_selall_btn.set_sensitive(not busy)
        self.seg_cancel_btn.set_sensitive(busy and cancel)
        (self.seg_spinner.start if busy else self.seg_spinner.stop)()
        if not busy:                        # any terminal state stops the shimmer
            self.seg_panel.view.set_scanning(False)
            self._hide_footer_progress()

    def _extract_bg(self):
        return IX.background(self.settings), IX.blur_strength(self.settings)

    def _ensure_seg_loaded(self, cb):
        if not self.source_path:
            self._toast("Open an image first.")
            return
        if self.seg_loaded_for == self.source_path:
            cb()
            return
        if self.seg_busy:
            return
        inp = os.path.join(self.tmpdir, "seg_input.png")
        try:
            # The RAW pixbuf, not export_pixbuf(): the overlays must come back in
            # un-rotated image space so they can ride the pane transform instead
            # of being invalidated by it (step 12).
            self.seg_panel.view.pixbuf.savev(inp, "png", [], [])
        except Exception as e:
            self._toast("Could not prepare image: %s" % e)
            return
        self.seg_pending = cb
        self._seg_set_busy(True, cancel=True)
        self.seg_status.set_text("Preparing segmentation model…")
        # Track this load so a reply that lands after we've left/unloaded (below)
        # can be ignored instead of wrongly re-marking the model as resident.
        self._seg_load_rid = self.worker.send({"op": "seg_load", "input": inp,
                                               "model": self._seg_model_choice()})

    def _on_seg_everything(self):
        if self.seg_busy:
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
        self.worker.send(req)

    def _on_seg_click(self, ix, iy, value, kind):
        if kind == "toggle":
            self._update_seg_selection_ui()
        else:  # point mode
            if self.seg_busy:
                return
            self.seg_points.append([ix, iy, value])
            self._ensure_seg_loaded(self._do_seg_point)

    def _do_seg_point(self):
        self._seg_set_busy(True)
        self.seg_status.set_text("Refining selection…")
        self.worker.send({"op": "seg_point", "points": self.seg_points,
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
        # The masks are in un-rotated source space; the user may have rotated the
        # view. Bake their transform into the output so the file matches what they
        # were looking at when they picked the objects.
        req.update(self.seg_panel.view.pane.export_transform())
        if self._seg_mode() == "point":
            req["mask"] = self.seg_point_mask
        else:
            req["ids"] = list(self.seg_panel.view.get_seg_selection())
        return req

    def _cancel_seg_preview(self):
        if getattr(self, "_seg_preview_timer", None):
            GLib.source_remove(self._seg_preview_timer)
            self._seg_preview_timer = None

    def _update_seg_preview(self):
        """Show the real output — the selected objects over the chosen
        background — in the right panel.

        A transparent or solid background is just display-list ops (a mask and a
        fill), so the renderer draws it *now*: no worker, no temp PNG, no 150 ms
        debounce. Only Blur genuinely needs the pixels, so only Blur still costs
        a round-trip. See engine/outputs.py.
        """
        if not self._seg_has_selection():
            self._cancel_seg_preview()
            self.seg_res_panel.view.clear()
            self.seg_result_output = None
            self.seg_save_btn.set_sensitive(False)
            return

        eff = OUT.resolve(IX.background(self.settings),
                          IX.blur_strength(self.settings))
        if eff.local:
            self._cancel_seg_preview()      # a pending blur prerender is stale
            self._seg_live_preview(eff)
            return

        self._cancel_seg_preview()
        self._seg_preview_timer = GLib.timeout_add(150, self._do_seg_prerender)

    def _seg_preview_masks(self):
        """The mask textures the cutout is clipped to."""
        v = self.seg_panel.view
        if self._seg_mode() == "point":
            tex = v.point_texture()
            return [tex] if tex is not None else []
        return [t for t in (v.seg_texture(i) for i in v.get_seg_selection())
                if t is not None]

    def _seg_live_preview(self, eff):
        """Draw the cutout on the canvas, this frame. The masks are already
        uploaded (they are the same textures the Segment panel highlights with),
        so this is a display list, not a render."""
        src = self.seg_panel.view.pixbuf
        masks = self._seg_preview_masks()
        if src is None or not masks:
            return
        rv = self.seg_res_panel.view
        if rv.pixbuf is src and rv.session.objects.clip_active:
            rv.update_composite(masks, eff.fill)      # keep zoom/pan
        else:
            rv.set_composite(src, masks, eff.fill)
        # Mirror the source panel's rotate/flip, so the preview shows the same
        # orientation the extract will bake into the saved file.
        sp = self.seg_panel.view.pane
        rv.pane.rot, rv.pane.fh, rv.pane.fv = sp.rot, sp.fh, sp.fv
        # Save still renders the real PNG in the worker — the canvas preview is
        # not a file.
        self.seg_result_output = None
        self.seg_save_btn.set_sensitive(True)

    def _do_seg_prerender(self):
        self._seg_preview_timer = None
        if self._seg_has_selection():
            self._seg_preview_rid = self.worker.send(self._seg_extract_req(
                os.path.join(self.tmpdir, "seg_preview.png")))
        return False

    def _on_seg_save(self):
        """Render the final PNG in the worker, then offer a save dialog."""
        if self.seg_busy:
            return
        if not self._seg_has_selection():
            self._toast("Pick at least one object first.")
            return
        self._seg_set_busy(True)
        self.seg_status.set_text("Preparing export…")
        self.worker.send(self._seg_extract_req(
            os.path.join(self.tmpdir, "seg_result.png")))

    def _seg_save_file(self, src):
        dlg = Gtk.FileDialog(title="Save result")
        base = os.path.splitext(os.path.basename(self.source_path or "image"))[0]
        dlg.set_initial_name(base + "_object.png")

        def done(d, res):
            try:
                f = d.save_finish(res)
            except GLib.Error:
                return
            path = f.get_path()
            if not path.lower().endswith(".png"):
                path += ".png"
            try:
                shutil.copyfile(src, path)
                self._toast("Saved " + os.path.basename(path))
            except Exception as e:
                self._toast("Save failed: %s" % e)
        dlg.save(self, None, done)

    def _on_seg_cancel(self):
        self.worker.send({"op": "seg_cancel"})
        self._unload("seg")            # free the SAM model even if it isn't cancellable
        self.seg_status.set_text("Cancelling…")

    def _on_seg_mode_changed(self, *_):
        self.settings.set("seg_mode",
                          self.seg_mode_toggle.get_active_name() or "everything")
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

    def _is_animated_gif(self, path):
        if not path.lower().endswith(".gif"):
            return False
        try:
            anim = GdkPixbuf.PixbufAnimation.new_from_file(path)
        except GLib.Error:
            return False
        return not anim.is_static_image()

    def _load_source(self, path):
        if not path:
            return
        if not self.src_panel.view.load_file(path):
            self._toast("Could not open that image.")
            return
        self.source_path = path
        self._source_is_gif = self._is_animated_gif(path)
        self.res_panel.view.clear()
        self.result_output = None
        self.save_btn.set_sensitive(False)
        self.gen_btn.set_sensitive(True)
        tail = " — animated GIF, press Generate." if self._source_is_gif \
            else " — press Generate."
        self.status.set_text(os.path.basename(path) + tail)
        # Mirror into the Segment page and reset its state for the new image.
        if getattr(self, "seg_panel", None) is not None:
            self.seg_panel.view.load_file(path)
            self.seg_res_panel.view.clear()
            self._seg_clear_overlays()
            self.seg_loaded_for = None
            self.seg_result_output = None
            self.seg_save_btn.set_sensitive(False)
            self.seg_status.set_text(os.path.basename(path) + " — Segment.")

    def _on_save(self):
        # A GIF result is the animated file itself (all frames); a still result
        # is the current pixbuf.
        is_gif = bool(self.result_output
                      and self.result_output.lower().endswith(".gif"))
        pb = None if is_gif else self.res_panel.view.export_pixbuf()
        if not is_gif and pb is None:
            return
        ext = ".gif" if is_gif else ".png"
        dlg = Gtk.FileDialog(title="Save result")
        base = os.path.splitext(os.path.basename(self.source_path or "image"))[0]
        dlg.set_initial_name(base + "_nobg" + ext)

        def done(d, res):
            try:
                f = d.save_finish(res)
            except GLib.Error:
                return
            path = f.get_path()
            if not path.lower().endswith(ext):
                path += ext
            try:
                if is_gif:
                    shutil.copyfile(self.result_output, path)
                else:
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
    def _set_busy(self, busy, spin=True, cancel=False):
        self.busy = busy
        self.gen_btn.set_sensitive(not busy and self.src_panel.view.has_image())
        self.gen_cancel_btn.set_sensitive(busy and cancel)
        self.batch_cancel_btn.set_sensitive(busy and cancel)
        self._sync_batch()
        if spin:
            (self.spinner.start if busy else self.spinner.stop)()
        if not busy:
            self.src_panel.view.set_scanning(False)   # end the scan shimmer
            self._hide_footer_progress()

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
        if self._source_is_gif:
            self._generate_gif(model, bg, alpha, blur)
            return
        inp = os.path.join(self.tmpdir, "input.png")
        try:
            self.src_panel.view.export_pixbuf().savev(inp, "png", [], [])
        except Exception as e:
            self._toast("Could not prepare image: %s" % e)
            return
        out = os.path.join(self.tmpdir, "result.png")
        self._set_busy(True, cancel=True)
        self.src_panel.view.set_scanning(True)      # scan shimmer over the source
        self.status.set_text("Generating…")
        self.worker.send({"op": "single", "input": inp, "output": out,
                          "model": model, "alpha": alpha, "bg": bg, "blur": blur})

    def _generate_gif(self, model, bg, alpha, blur):
        # Same options as a still; the worker processes each frame and reports
        # per-frame progress. Pass the current rotate/flip so it bakes in like a
        # still would; process the original file (all frames) not the preview.
        out = os.path.join(self.tmpdir, "result.gif")
        v = self.src_panel.view
        self._set_busy(True, cancel=True)
        self.src_panel.view.set_scanning(True)
        self.status.set_text("Removing background from GIF…")
        self.worker.send({"op": "gif", "input": self.source_path, "output": out,
                          "model": model, "alpha": alpha, "bg": bg, "blur": blur,
                          "rot": v.rot, "fh": v.fh, "fv": v.fv})

    def _on_bg_cancel(self):
        if not self.busy:
            return
        self.worker.send({"op": "cancel"})
        self.gen_cancel_btn.set_sensitive(False)
        self.batch_cancel_btn.set_sensitive(False)
        self.status.set_text("Cancelling…")
        if self.stack.get_visible_child_name() == "batch":
            self.batch_status.set_text("Cancelling…")

    def _on_run_batch(self):
        if self.busy or not (self.batch_input and self.batch_output):
            return
        model, bg, alpha, blur = self.get_settings()
        pattern = self.pattern_row.get_text().strip() or "{name}_nobg"
        self._set_busy(True, spin=False, cancel=True)
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
        elif t == "loading":
            self.status.set_text(
                "Loading %s… (the first run is slower)" % msg["model"])
            self.batch_status.set_text("Loading model %s…" % msg["model"])
        elif t == "notice":
            self._toast(msg["message"])
        elif t == "device":
            self._set_device(msg["provider"], msg["label"])
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
        elif t == "gif_progress":
            self._footer_progress(msg["done"], msg["total"])
            if msg["done"] == 0:
                self.status.set_text("Removing background — %d frames…"
                                     % msg["total"])
            else:
                self.status.set_text("Removing background — frame %d / %d"
                                     % (msg["done"], msg["total"]))
        elif t == "gif_done":
            self.res_panel.view.load_file(msg["output"])   # first-frame preview
            self.result_output = msg["output"]
            self.save_btn.set_sensitive(True)
            self.status.set_text("Done: %d frames in %.1fs — Save the GIF."
                                 % (msg["frames"], msg["seconds"]))
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
            if msg.get("id") != getattr(self, "_seg_load_rid", None):
                return False           # left/unloaded before this load finished
            self.seg_loaded_for = self.source_path
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
            self._seg_set_busy(False)
            self.seg_objects = msg["objects"]
            self.seg_panel.view.set_seg_layers(
                msg["objects"], msg["label_map"],
                msg.get("general_map"), msg.get("depth_map"))
            self._update_seg_preview()
            self._seg_base_status = (
                "%d objects — hover to highlight, click to keep." % msg["count"])
            self.seg_status.set_text(self._seg_base_status)
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
            if out.endswith("seg_preview.png"):     # live prerender → right panel
                # Ignore a stale reply: a prerender sent for an earlier selection
                # can land during response lag, after we've cleared the panel.
                if (msg.get("id") != getattr(self, "_seg_preview_rid", None)
                        or not self._seg_has_selection()):
                    return False
                self.seg_res_panel.view.load_file(out, keep_transform=True)
                self.seg_result_output = out
                self.seg_save_btn.set_sensitive(True)
            else:                                    # explicit Save → file dialog
                self._seg_set_busy(False)
                self.seg_status.set_text("Ready — choose where to save.")
                self._seg_save_file(out)
        elif t == "seg_canceled":
            self._seg_set_busy(False)
            self._unload("seg")            # free the SAM model + cached encode
            self.seg_status.set_text("Cancelled.")
        elif t == "error":
            first = msg["message"].splitlines()[0]
            self.status.set_text("Error: " + first)
            self.batch_status.set_text("Error: " + first)
            if getattr(self, "seg_busy", False):
                self._seg_set_busy(False)
                self.seg_status.set_text("Error: " + first)
            self._toast("Error: " + first)
            self._set_busy(False)
            self._set_busy(False, spin=False)
        return False

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
             "Your picks composite live on the right; Save exports a PNG."),
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
        provider.load_from_string(CSS)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
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
            comments="Local image background remover (BiRefNet / rembg) and "
                     "object segmentation (Segment Anything / SAM 2.1).",
            license_type=Gtk.License.MIT_X11,
            website="https://github.com/cristeigabriela/bg-be-gone")
        about.set_transient_for(self.get_active_window())
        about.present()


def main():
    return App().run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
