"""The Batch page: a folder's images in a gallery, how to write them, and Run.

The window (app.py) owns the worker and every request sent to it. This page
builds its batch, thumbnail and preview requests and shows their replies; it
reaches the window only through BatchHost.

Top to bottom: the folders and the rename pattern; the gallery beside the
Output card and its live preview; the run bar. The page's output choices are
its own, remembered apart from the Single page's.
"""
import enum
import logging
import math
import os
from dataclasses import dataclass
from typing import Optional, Protocol

from gi.repository import Adw, Gdk, GLib, Gtk

import background
import batchnames
import formats
import offmain
from gallery import (Conversion, Editing, Gallery, Selection, Thumbnails, TileState,
                     WorkerQueue)
from widgets import action_bar, pill_button, primary_button

_log = logging.getLogger("bgbg")

# The preview art (tools/make_preview_art.py): a photo and its exact cut-out,
# named as the worker's recomposite op reads them from a keep_dir.
PREVIEW_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "preview")

DEFAULT_PATTERN = "{name}_nobg"
_PATTERN_HINT = ("<small>Tokens: <tt>{name}</tt> the file's name, <tt>{n}</tt> its number "
                 "in the run. A name that's taken gets a number, so nothing is "
                 "overwritten.</small>")
# Failed or skipped files a summary lists by name, so the dialog fits on screen.
_MAX_LISTED = 15
# The Output card's width: room for [PNG | WebP | JPEG] beside its row title.
_OUTPUT_CARD_WIDTH = 330


class BatchHost(Protocol):
    """What the page needs from the window; app.Window provides it."""

    tmpdir: str

    def run_batch(self, req: dict) -> int:
        """Send a batch and mark the window busy; the request id, or -1."""

    def cancel_batch(self) -> None: ...

    def send_batch_thumb(self, req: dict) -> int:
        """Send a thumbnail conversion; replies go to on_thumb_reply."""

    def send_batch_preview(self, req: dict) -> int:
        """Send a preview render; replies go to BatchPreview.on_reply."""

    def batch_model(self) -> str:
        """The background-removal model the sidebar picks."""

    def worker_is_up(self) -> bool: ...

    def is_busy(self) -> bool:
        """Whether a Single or Batch job holds the worker."""

    def setting(self, key: str):
        """A remembered value, or None."""

    def remembered_folder(self, key: str):
        """A remembered folder that still exists, as a Gio.File, or None."""

    def remember(self, **values) -> None: ...

    def open_folder(self, path: str) -> None: ...

    def toast(self, text: str) -> None: ...


class Background(enum.Enum):
    """What a batch puts behind each cut-out."""

    REMOVE = "remove"        # nothing: transparent
    BLUR = "blur"            # the photo itself, blurred
    COLOUR = "colour"        # one flat colour


class Matting(enum.Enum):
    """Alpha matting: cleaner edges, a little slower."""

    ON = "on"
    OFF = "off"


class Run(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"


class Cancel(enum.Enum):
    """Whether the window offers to cancel the job it runs."""

    OFFERED = "offered"
    NOT_OFFERED = "not_offered"


# Colours offered beside the custom colour button: (tooltip, "#rrggbb").
SWATCHES = (("White", "#ffffff"), ("Black", "#000000"), ("Green screen", "#00b140"))


@dataclass(frozen=True)
class Output:
    """How a batch writes each image: the Output card's choices."""

    background: Background
    colour: str              # "#rrggbb", for Background.COLOUR
    blur: int
    chosen: formats.ImageFormat   # the format picked; `format` is what is written
    matting: Matting

    @property
    def bg(self):
        """The worker's background setting."""
        if self.background is Background.REMOVE:
            return "transparent"
        if self.background is Background.BLUR:
            return "blur"
        return self.colour

    @property
    def alpha(self):
        if self.background is Background.REMOVE:
            return formats.Alpha.TRANSPARENT
        return formats.Alpha.FLATTENED

    @property
    def format(self):
        """The chosen format, or PNG while it can't keep the transparency
        (JPEG with Remove). The choice comes back with a flat background."""
        if self.chosen in formats.save_formats(self.alpha):
            return self.chosen
        return formats.PNG

    def look(self):
        """What the preview depends on: format and matting don't show on it."""
        return (self.bg, self.blur if self.background is Background.BLUR else 0)


DEFAULT_OUTPUT = Output(Background.REMOVE, "#ffffff", background.STRENGTH_DEFAULT,
                        formats.PNG, Matting.OFF)


def _hex(rgba):
    return "#%02x%02x%02x" % (round(rgba.red * 255), round(rgba.green * 255),
                              round(rgba.blue * 255))


def _restored_output(setting):
    """The remembered Output choices; each bad or missing one keeps its default."""
    d = DEFAULT_OUTPUT
    behind = next((b for b in Background if b.value == setting("batch_background")),
                  d.background)
    colour = d.colour
    rgba = Gdk.RGBA()
    if isinstance(setting("batch_colour"), str) and rgba.parse(setting("batch_colour")):
        colour = _hex(rgba)
    blur = setting("batch_blur_strength")
    if not (isinstance(blur, (int, float)) and not isinstance(blur, bool) and math.isfinite(blur)):
        blur = d.blur
    blur = int(min(max(blur, background.STRENGTH_MIN), background.STRENGTH_MAX))
    chosen = formats.BATCH_OUTPUTS.get(setting("batch_format"), d.chosen)
    matting = Matting.ON if setting("batch_alpha_matting") is True else Matting.OFF
    return Output(behind, colour, blur, chosen, matting)


def scan_folder(folder):
    """The images in `folder`, by name: [(path, formats.Animation)].

    Hidden files are left out, as the worker's own folder scan leaves them
    out: a batch's unfinished part files are hidden. A GIF counts as
    animated even when it is a still, since batch leaves every GIF out; any
    other file is judged by its header, as the worker judges it by content.
    """
    found = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.name.startswith(".") or not entry.is_file():
                continue
            fmt = formats.format_of(entry.name)
            if fmt is None:
                continue
            if fmt is formats.GIF:
                animation = formats.Animation.ANIMATED
            else:
                animation = formats.animation_in(entry.path)
            found.append((entry.path, animation))
    found.sort(key=lambda image: os.path.basename(image[0]).casefold())
    return found


def _count_text(images):
    """'23 images · 2 animated, open those in Single'."""
    animated = sum(1 for _, a in images if a is formats.Animation.ANIMATED)
    text = "1 image" if len(images) == 1 else "%d images" % len(images)
    if animated:
        text += " · %d animated, open %s in Single" % (animated,
                                                       "it" if animated == 1 else "those")
    return text


class BatchPreview:
    """The Output card's picture: what a batch writes, shown on the preview art.

    A change waits DEBOUNCE_MS to settle, and only the latest choices are
    rendered: one request is out at a time, and when it answers, the newest
    choices go next if they changed meanwhile.
    """

    DEBOUNCE_MS = 150

    def __init__(self, host, picture):
        self._host = host
        self._picture = picture
        self._wanted = None          # the Output to show
        self._sent = None            # (look, png path) of the request out
        self._shown = None           # the look on screen
        self._shown_file = None
        self._rid = None
        self._timer = 0
        self._made = 0

    def show(self, output):
        self._wanted = output
        if self._timer:
            GLib.source_remove(self._timer)
        self._timer = GLib.timeout_add(self.DEBOUNCE_MS, self._settled)

    def _settled(self):
        self._timer = 0
        if self._rid is None:
            self._send()
        return False

    def _send(self):
        """Ask the worker to render the wanted look, unless it is on screen.

        The worker re-composites the preview art as it re-composites a kept
        Generate result, through the compositing a batch writes with.
        """
        if self._wanted is None or self._wanted.look() == self._shown:
            return
        look = self._wanted.look()
        self._made += 1
        path = os.path.join(self._host.tmpdir, "batch-preview-%d.png" % self._made)
        rid = self._host.send_batch_preview({
            "op": "recomposite", "keep_dir": PREVIEW_ART, "output": path,
            "bg": self._wanted.bg, "blur": self._wanted.blur})
        if rid < 0:
            return                   # no worker: worker_ready() asks again
        self._rid, self._sent = rid, (look, path)

    def on_reply(self, rid, msg):
        if rid != self._rid:
            return
        self._rid = None
        look, path = self._sent
        # A transparent result also gets a checkered copy to show, as in Single.
        written = (path, path + ".preview.png")
        if msg["type"] != "recomposited":
            _log.warning("the batch preview failed: %s", msg.get("message"))
            for f in written:
                _remove_quietly(f)
            self._send_if_changed(look)
            return
        shown = msg["preview"]

        def decode():
            return Gdk.Texture.new_from_filename(shown)

        def decoded(result):
            for f in written:
                _remove_quietly(f)
            if isinstance(result, Exception):
                _log.warning("can't show the batch preview %s: %s", shown, result)
            else:
                self._picture.set_paintable(result)
                self._shown = look
            self._send_if_changed(look)
        offmain.run(decode, decoded)

    def _send_if_changed(self, look):
        if self._rid is None and self._wanted is not None and self._wanted.look() != look:
            self._send()

    def requests_lost(self):
        self._rid = None

    def worker_ready(self):
        if self._rid is None:
            self._send()


def _remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


_CSS = """
gridview.batch-gallery { background: none; }
gridview.batch-gallery > child { padding: 5px; background: none; border-radius: 17px; }
.batch-tile {
  border-radius: 14px;
  border: 3px solid transparent;
  background-color: alpha(currentColor, 0.07);
}
.batch-tile.kept { border-color: @accent_bg_color; }
.batch-tile.left-out .tile-picture { opacity: 0.35; }
.tile-name {
  padding: 16px 8px 5px;
  color: white; font-size: 0.85em;
  background-image: linear-gradient(to top, alpha(black, 0.7), alpha(black, 0));
  opacity: 0;
  transition: opacity 150ms ease;
}
.batch-tile:hover .tile-name { opacity: 1; }
.tile-check {
  margin: 5px; padding: 2px; border-radius: 999px;
  background-color: alpha(black, 0.35);
}
.tile-flip { margin: 5px; min-width: 28px; min-height: 28px; padding: 0; }
.tile-badge {
  margin: 6px; padding: 2px 8px; border-radius: 999px;
  background-color: alpha(black, 0.65); color: white;
  font-size: 0.78em; font-weight: bold;
}
.tile-state {
  margin: 6px; padding: 6px; border-radius: 999px;
  background-color: alpha(black, 0.6); color: white;
  box-shadow: 0 2px 8px alpha(black, 0.3);
}
.tile-state.done { background-color: @accent_bg_color; color: @accent_fg_color; }
.tile-state.failed { background-color: @error_bg_color; color: @error_fg_color; }
.more-tile {
  border-radius: 14px;
  border: 2px dashed alpha(currentColor, 0.25);
}
.batch-preview {
  border-radius: 12px;
  border: 1px solid alpha(currentColor, 0.15);
  background-color: alpha(currentColor, 0.07);
}
.batch-output { padding: 12px; }
button.swatch {
  min-width: 24px; min-height: 24px; padding: 0; border-radius: 999px;
  border: 1px solid alpha(currentColor, 0.35);
}
button.swatch image { -gtk-icon-size: 14px; }
"""
_css_loaded = False


def _load_css():
    """The page's own look, added once beside the app's."""
    global _css_loaded
    if _css_loaded:
        return
    provider = Gtk.CssProvider()
    css = _CSS + "".join(
        "button.swatch-%d { background: %s; color: %s; }\n"
        % (i, colour, "white" if colour == "#000000" else "black")
        for i, (_, colour) in enumerate(SWATCHES))
    provider.load_from_string(css)
    Gtk.StyleContext.add_provider_for_display(
        Gdk.Display.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
    _css_loaded = True


class BatchPage:
    """The Batch page's widgets and state. `widget` goes into the window's stack."""

    def __init__(self, host: BatchHost):
        self._host = host
        _load_css()
        self._thumbs = Thumbnails(host.tmpdir, self._convert_for_thumbnail)
        self.gallery = Gallery(self._thumbs, self._on_selection_changed)
        self.input: Optional[str] = None
        self.output: Optional[str] = None
        self._images = []                  # the scanned folder: [(path, Animation)]
        self._scan_token = None
        self._run = Run.IDLE
        self._runs = 0                     # runs so far, to name each one's thumbnail folder
        self._last_outdir = None           # where the last run wrote
        self._thumb_jobs = {}              # request id -> done(ok)
        self._cancel = Cancel.NOT_OFFERED
        self._out = DEFAULT_OUTPUT
        self._restoring = True
        # Detached: the window's footer shows the page's status.
        self.status = Gtk.Label(xalign=0, label="")
        self.widget = self._build()
        self.preview = BatchPreview(host, self._preview_picture)
        self._sync()

    # ---------- building ----------
    def _build(self):
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        for side in ("top", "bottom", "start", "end"):
            getattr(page, "set_margin_" + side)(10)
        page.append(self._build_folders())
        middle = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        middle.set_vexpand(True)
        middle.append(self._build_gallery())
        middle.append(self._build_output())
        page.append(middle)
        page.append(self._build_run_bar())
        return page

    def _build_folders(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        grp = Adw.PreferencesGroup()
        self.in_row = Adw.ActionRow(title="Input folder", subtitle="Not set — or drop one here")
        self.in_row.set_subtitle_lines(1)
        self.count_label = Gtk.Label()
        self.count_label.add_css_class("dim-label")
        self.count_label.set_valign(Gtk.Align.CENTER)
        self.in_row.add_suffix(self.count_label)
        self.in_btn = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        self.in_btn.connect("clicked", self._on_pick_input)
        self.in_row.add_suffix(self.in_btn)
        grp.add(self.in_row)

        self.out_row = Adw.ActionRow(title="Output folder", subtitle="Not set")
        self.out_row.set_subtitle_lines(1)
        self.out_btn = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        self.out_btn.connect("clicked", self._on_pick_output)
        self.out_row.add_suffix(self.out_btn)
        grp.add(self.out_row)

        self.pattern_row = Adw.EntryRow(title="Rename pattern")
        self.pattern_row.set_text(DEFAULT_PATTERN)
        self.example = Gtk.Label()
        self.example.add_css_class("dim-label")
        self.example.add_css_class("monospace")
        self.example.set_valign(Gtk.Align.CENTER)
        self.pattern_row.add_suffix(self.example)
        self.pattern_row.connect("changed", lambda *_: self._on_pattern_changed())
        grp.add(self.pattern_row)
        # A long folder path would otherwise ask the window for its whole
        # width, and the sidebar is what gives way.
        narrow = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER,
                                    vscrollbar_policy=Gtk.PolicyType.NEVER,
                                    propagate_natural_height=True)
        narrow.set_child(grp)
        box.append(narrow)

        self.pattern_hint = Gtk.Label(xalign=0, wrap=True)
        self.pattern_hint.add_css_class("dim-label")
        self.pattern_hint.set_markup(_PATTERN_HINT)
        box.append(self.pattern_hint)
        return box

    def _build_gallery(self):
        column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        column.set_hexpand(True)
        head = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        title = Gtk.Label(label="Images", xalign=0)
        title.add_css_class("heading")
        head.append(title)
        self.selected_label = Gtk.Label(xalign=0)
        self.selected_label.add_css_class("dim-label")
        head.append(self.selected_label)
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        head.append(spacer)
        self.all_btn = Gtk.Button(label="Select all")
        self.all_btn.add_css_class("flat")
        self.all_btn.connect("clicked", lambda *_: self.gallery.select_all(Selection.KEPT))
        self.none_btn = Gtk.Button(label="Select none")
        self.none_btn.add_css_class("flat")
        self.none_btn.connect("clicked", lambda *_: self.gallery.select_all(Selection.LEFT_OUT))
        head.append(self.all_btn)
        head.append(self.none_btn)
        column.append(head)

        self.gallery_stack = Gtk.Stack()
        self.gallery_stack.set_vexpand(True)
        self.empty = Adw.StatusPage(
            icon_name="folder-pictures-symbolic", title="Choose or drop a folder",
            description="Its images show up here. Untick the ones to leave out.")
        self.empty.add_css_class("compact")
        self.gallery_stack.add_named(self.empty, "empty")
        scroller = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER)
        scroller.set_child(self.gallery.view)
        self.gallery_stack.add_named(scroller, "grid")
        column.append(self.gallery_stack)
        return column

    def _build_output(self):
        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        card.add_css_class("card")
        card.add_css_class("batch-output")
        title = Gtk.Label(label="Output", xalign=0)
        title.add_css_class("heading")
        card.append(title)

        self._preview_picture = Gtk.Picture(content_fit=Gtk.ContentFit.COVER, can_shrink=True)
        self._preview_picture.set_tooltip_text(
            "What a batch writes, on a sample picture: only the background changes")
        frame = Gtk.AspectFrame(ratio=1.5, obey_child=False, child=self._preview_picture)
        frame.add_css_class("batch-preview")
        frame.set_overflow(Gtk.Overflow.HIDDEN)
        card.append(frame)

        self.bg_toggle = Adw.ToggleGroup()
        self.bg_toggle.add(Adw.Toggle(name=Background.REMOVE.value, label="Remove",
                                      tooltip="Transparent background"))
        self.bg_toggle.add(Adw.Toggle(name=Background.BLUR.value, label="Blur",
                                      tooltip="Keep the photo, blur its background"))
        self.bg_toggle.add(Adw.Toggle(name=Background.COLOUR.value, label="Colour",
                                      tooltip="Put the subject on a flat colour"))
        self.bg_toggle.set_homogeneous(True)
        card.append(self.bg_toggle)

        rows = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        rows.add_css_class("boxed-list")
        self.blur_row = Adw.SpinRow(
            title="Blur strength",
            adjustment=Gtk.Adjustment(value=DEFAULT_OUTPUT.blur,
                                      lower=background.STRENGTH_MIN,
                                      upper=background.STRENGTH_MAX, step_increment=1,
                                      page_increment=5))
        rows.append(self.blur_row)

        self.colour_row = Adw.ActionRow(title="Colour")
        self._swatches = []
        for i, (name, colour) in enumerate(SWATCHES):
            swatch = Gtk.Button(valign=Gtk.Align.CENTER)
            swatch.add_css_class("swatch")
            swatch.add_css_class("swatch-%d" % i)
            swatch.set_tooltip_text(name)
            mark = Gtk.Image.new_from_icon_name("object-select-symbolic")
            swatch.set_child(mark)
            swatch.connect("clicked", lambda _b, c=colour: self._set_colour(c))
            self.colour_row.add_suffix(swatch)
            self._swatches.append((colour, mark))
        self.colour_btn = Gtk.ColorDialogButton.new(Gtk.ColorDialog(with_alpha=False))
        self.colour_btn.set_valign(Gtk.Align.CENTER)
        self.colour_btn.set_tooltip_text("Any colour")
        self.colour_row.add_suffix(self.colour_btn)
        rows.append(self.colour_row)

        self.format_row = Adw.ActionRow(title="Format")
        self.format_toggle = Adw.ToggleGroup(valign=Gtk.Align.CENTER)
        for key, fmt in formats.BATCH_OUTPUTS.items():
            self.format_toggle.add(Adw.Toggle(name=key, label=fmt.name))
        self.format_row.add_suffix(self.format_toggle)
        rows.append(self.format_row)

        self.matting_row = Adw.SwitchRow(title="Alpha matting",
                                         subtitle="Cleaner edges, a little slower")
        rows.append(self.matting_row)
        card.append(rows)

        self.format_hint = Gtk.Label(xalign=0, wrap=True)
        self.format_hint.add_css_class("dim-label")
        self.format_hint.add_css_class("caption")
        card.append(self.format_hint)

        self.output_card = card
        scroller = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER,
                                      propagate_natural_height=True)
        scroller.set_child(card)
        scroller.set_size_request(_OUTPUT_CARD_WIDTH, -1)
        scroller.set_valign(Gtk.Align.START)
        return scroller

    def _build_run_bar(self):
        self.run_btn = primary_button("Run batch", icon="view-grid-symbolic")
        self.run_btn.set_tooltip_text("Run the batch (Ctrl+Return)")
        self.run_btn.connect("clicked", lambda *_: self.run())
        self.cancel_btn = pill_button("Cancel")
        self.cancel_btn.connect("clicked", lambda *_: self._host.cancel_batch())
        self.open_btn = Gtk.Button()
        self.open_btn.set_child(Adw.ButtonContent(icon_name="folder-open-symbolic",
                                                  label="Open output folder"))
        self.open_btn.add_css_class("pill")
        self.open_btn.set_visible(False)
        self.open_btn.connect("clicked", lambda *_: self._host.open_folder(self._last_outdir))
        bar, _status, self.spinner = action_bar(
            primary=self.run_btn, cancel=self.cancel_btn, save=self.open_btn)
        return bar

    # ---------- remembered choices ----------
    def restore(self):
        """Put back the last launch's folders and Output choices, then save each
        change. The handlers connect after, so restoring writes nothing."""
        for key in ("batch_input", "batch_output"):
            folder = self._host.remembered_folder(key)
            if folder is not None:
                self._set_folder(key, folder.get_path())
        self._show_output(_restored_output(self._host.setting))
        self.bg_toggle.connect("notify::active-name", lambda *_: self._on_output_changed())
        self.format_toggle.connect("notify::active-name", lambda *_: self._on_output_changed())
        self.blur_row.connect("notify::value", lambda *_: self._on_output_changed())
        self.colour_btn.connect("notify::rgba", lambda *_: self._on_output_changed())
        self.matting_row.connect("notify::active", lambda *_: self._on_output_changed())
        self._restoring = False
        self._on_output_changed()

    def _show_output(self, out):
        """Set the Output card's widgets to `out`."""
        self.bg_toggle.set_active_name(out.background.value)
        self.blur_row.set_value(out.blur)
        rgba = Gdk.RGBA()
        rgba.parse(out.colour)
        self.colour_btn.set_rgba(rgba)
        self.format_toggle.set_active_name(formats.batch_output_key(out.chosen))
        self.matting_row.set_active(out.matting is Matting.ON)
        self._out = out

    def _read_output(self):
        """The Output card's choices, as its widgets show them."""
        behind = Background(self.bg_toggle.get_active_name() or Background.REMOVE.value)
        chosen = formats.BATCH_OUTPUTS.get(self.format_toggle.get_active_name(), formats.PNG)
        # A disabled JPEG toggle can still be the active one: keep the person's
        # choice, which `format` sets aside while the background is transparent.
        if (chosen is formats.PNG and self._out.chosen is formats.JPEG
                and behind is Background.REMOVE):
            chosen = formats.JPEG
        return Output(behind, _hex(self.colour_btn.get_rgba()),
                      int(self.blur_row.get_value()), chosen,
                      Matting.ON if self.matting_row.get_active() else Matting.OFF)

    def _on_output_changed(self):
        if self._restoring:
            return
        out = self._read_output()
        self._out = out
        self.blur_row.set_visible(out.background is Background.BLUR)
        self.colour_row.set_visible(out.background is Background.COLOUR)
        for colour, mark in self._swatches:
            mark.set_visible(colour == out.colour)
        # JPEG can't hold the transparency: offered only on a flat background.
        jpeg = self.format_toggle.get_toggle_by_name(formats.batch_output_key(formats.JPEG))
        jpeg.set_enabled(out.alpha is formats.Alpha.FLATTENED)
        shown = formats.batch_output_key(out.format)
        if self.format_toggle.get_active_name() != shown:
            self.format_toggle.set_active_name(shown)
        if out.format is not out.chosen:
            self.format_hint.set_text("JPEG can't keep a transparent background, so this "
                                      "writes PNG. Pick Blur or Colour to write JPEG.")
        else:
            self.format_hint.set_text("")
        self.format_hint.set_visible(out.format is not out.chosen)
        self._host.remember(batch_background=out.background.value, batch_colour=out.colour,
                            batch_blur_strength=out.blur,
                            batch_format=formats.batch_output_key(out.chosen),
                            batch_alpha_matting=out.matting is Matting.ON)
        self.preview.show(out)
        self._update_example()

    def _set_colour(self, colour):
        rgba = Gdk.RGBA()
        rgba.parse(colour)
        self.colour_btn.set_rgba(rgba)

    # ---------- folders ----------
    def _on_pick_input(self, *_):
        self._pick_folder("Input folder", ("batch_input",), "batch_input")

    def _on_pick_output(self, *_):
        self._pick_folder("Output folder", ("batch_output", "batch_input"), "batch_output")

    def _pick_folder(self, title, starts, key):
        dlg = Gtk.FileDialog(title=title)
        start = next((f for f in map(self._host.remembered_folder, starts) if f), None)
        if start is not None:
            dlg.set_initial_folder(start)

        def done(d, res):
            try:
                f = d.select_folder_finish(res)
            except GLib.Error:
                return
            self._set_folder(key, f.get_path())
        dlg.select_folder(self.widget.get_root(), None, done)

    def set_input(self, folder):
        """A folder dropped on the page or picked: show its images."""
        if self._run is Run.RUNNING:
            self._host.toast("A batch is running: change the folder once it's done.")
            return
        self._set_folder("batch_input", folder)
        self.status.set_text("Input folder: " + folder)

    def _set_folder(self, key, folder):
        self._host.remember(**{key: folder})
        if key == "batch_output":
            self.output = folder
            self.out_row.set_subtitle(folder)
            self._update_example()
            self._sync()
            return
        self.input = folder
        self.in_row.set_subtitle(folder)
        self.count_label.set_text("Reading…")
        token = self._scan_token = object()

        def scanned(result):
            if self._scan_token is not token:
                return               # another folder was picked meanwhile
            images = [] if isinstance(result, Exception) else result
            self._images = images
            self.count_label.set_text(_count_text(images))
            self.gallery.set_images(images)
            if images:
                self.gallery_stack.set_visible_child_name("grid")
                return
            if isinstance(result, Exception):
                _log.warning("can't read the folder %s", folder, exc_info=result)
                reason = getattr(result, "strerror", None) or str(result)
                self.empty.set_title("Can't read this folder")
                self.empty.set_description(reason)
                self.status.set_text("Can't read %s: %s." % (folder, reason))
            else:
                self.empty.set_title("No images here")
                self.empty.set_description("bg-be-gone reads PNG, JPEG, WebP, AVIF, HEIC, "
                                           "TIFF, BMP and GIF.")
            self.gallery_stack.set_visible_child_name("empty")
        offmain.run(lambda: scan_folder(folder), scanned)

    # ---------- the rename pattern ----------
    def _pattern(self):
        return self.pattern_row.get_text().strip() or DEFAULT_PATTERN

    def _pattern_problem(self):
        """Why the rename pattern can't name files, or None."""
        try:
            batchnames.check_pattern(self._pattern())
        except batchnames.BatchPatternError as e:
            return e.user_message
        return None

    def _on_pattern_changed(self):
        problem = self._pattern_problem()
        if problem:
            self.pattern_row.add_css_class("error")
            self.pattern_hint.set_text(problem)
        else:
            self.pattern_row.remove_css_class("error")
            self.pattern_hint.set_markup(_PATTERN_HINT)
        self._update_example()
        self._sync()

    def _update_example(self):
        """'cat.jpg → cat_nobg.png' for the run's first image, with the name the
        run would give it now, taken names and all."""
        if self._pattern_problem():
            self.example.set_text("")
            return
        selected = self.gallery.selected()
        source = selected[0].path if selected else "photo.jpg"
        stem = batchnames.stem_for(self._pattern(), source, 1)
        suffix = self._out.format.suffix
        name = stem + suffix
        if self.output:
            taken = batchnames.taken_by([item.path for item in selected])
            name = os.path.basename(batchnames.first_free(self.output, stem, suffix, taken))
        self.example.set_text("%s → %s" % (os.path.basename(source), name))
        self.example.set_tooltip_text(
            None if name == stem + suffix
            else "%s is taken, so the result gets a number." % (stem + suffix))

    # ---------- selection ----------
    def _on_selection_changed(self):
        selectable = len(self.gallery.selectable())
        selected = len(self.gallery.selected())
        self.selected_label.set_text("%d of %d selected" % (selected, selectable)
                                     if self.gallery.items else "")
        self.run_btn.get_child().set_label(
            "Run batch on %d of %d" % (selected, selectable) if self.gallery.items
            else "Run batch")
        self._update_example()
        self._sync()

    # ---------- running ----------
    def can_run(self):
        return (self._run is Run.IDLE and not self._host.is_busy()
                and self._host.worker_is_up() and self.input and self.output
                and self.gallery.selected() and self._pattern_problem() is None)

    def run(self):
        """Run the batch on the selected images, as the Output card says."""
        if not self.can_run():
            return
        out = self._out
        selected = self.gallery.selected()
        self._runs += 1
        req = {
            "op": "batch", "input_dir": self.input, "output_dir": self.output,
            "files": [item.path for item in selected], "model": self._host.batch_model(),
            "alpha": out.matting is Matting.ON, "bg": out.bg, "blur": out.blur,
            "format": formats.batch_output_key(out.format), "pattern": self._pattern(),
            "thumbs": os.path.join(self._host.tmpdir, "batch-run-%d" % self._runs)}
        for item in self.gallery.items:
            item.state = TileState.QUEUED if item.selection is Selection.KEPT \
                else TileState.IDLE
            item.message = item.output = item.result_thumb = None
            item.reset_result()
            item.changed()
        if self._host.run_batch(req) < 0:
            self._end_run()
            return
        self._run = Run.RUNNING
        self._last_outdir = self.output
        self.open_btn.set_visible(False)
        self.status.set_text("Preparing…")
        self._lock()

    def offer_cancel(self, cancel):
        """The window's busy state changed (a Single or Batch job)."""
        self._cancel = cancel
        self._sync()

    def on_file(self, msg):
        """One image's state in the running batch."""
        item = self.gallery.item_for(msg.get("input"))
        if item is None:
            return
        item.state = TileState(msg["state"])
        item.message = msg.get("message")
        if item.state is TileState.DONE:
            item.output = msg.get("output")
            item.result_thumb = msg.get("thumb")
        item.changed()

    def on_done(self, msg):
        self._end_run()
        self._summarise(msg)

    def on_canceled(self, msg):
        self._end_run()

    def on_failed(self):
        """The batch request failed as a whole (a bad pattern, no images)."""
        self._end_run()

    def _end_run(self):
        """The run is over, however it ended: tiles it never reached go back to idle."""
        was_running = self._run is Run.RUNNING
        self._run = Run.IDLE
        for item in self.gallery.items:
            if item.state in (TileState.QUEUED, TileState.WORKING):
                item.state = TileState.IDLE
                item.changed()
        self.open_btn.set_visible(was_running and self._last_outdir is not None)
        self._lock()
        self._pause_thumbnails()

    def _lock(self):
        """While a run goes, what it was started with can't change."""
        editing = Editing.LOCKED if self._run is Run.RUNNING else Editing.OPEN
        self.gallery.set_editing(editing)
        for widget in (self.output_card, self.in_btn, self.out_btn, self.pattern_row):
            widget.set_sensitive(editing is Editing.OPEN)
        self._pause_thumbnails()
        self._sync()

    def _sync(self):
        choosing = self._run is Run.IDLE and bool(self.gallery.items)
        self.all_btn.set_sensitive(choosing)
        self.none_btn.set_sensitive(choosing)
        self.run_btn.set_sensitive(bool(self.can_run()))
        self.cancel_btn.set_sensitive(self._run is Run.RUNNING
                                      and self._cancel is Cancel.OFFERED)

    def _summarise(self, msg):
        """Say how the run went; list what failed or was skipped."""
        failed, skipped = msg["failed"], msg["skipped"]
        parts = ["%d done" % msg["count"]]
        if failed:
            parts.append("%d failed" % len(failed))
        if skipped:
            parts.append("%d skipped" % len(skipped))
        summary = ", ".join(parts)
        if msg["renamed"]:
            summary += " (%d numbered, so nothing was overwritten)" % msg["renamed"]
        self.status.set_text("%s in %.1fs → %s" % (summary, msg["seconds"], msg["outdir"]))
        if not (failed or skipped):
            self._host.toast("Batch finished: " + summary)
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
        dlg.connect("response", lambda d, r: self._host.open_folder(msg["outdir"])
                    if r == "open" else None)
        dlg.present(self.widget)

    # ---------- the worker ----------
    def _convert_for_thumbnail(self, path, out_png, side, done):
        """Ask the worker for a thumbnail GdkPixbuf can't decode itself."""
        rid = self._host.send_batch_thumb(
            {"op": "convert", "input": path, "output": out_png, "max_size": side})
        if rid < 0:
            # No worker to ask: hold the rest until one is ready again.
            self._thumbs.set_worker_queue(WorkerQueue.PAUSED)
            done(Conversion.LOST)
            return
        self._thumb_jobs[rid] = done

    def on_thumb_reply(self, rid, msg):
        done = self._thumb_jobs.pop(rid, None)
        if done is not None:
            done(Conversion.DONE if msg["type"] == "converted" else Conversion.FAILED)

    def _pause_thumbnails(self):
        # A run, or no worker: conversions would only wait in its queue.
        if self._run is Run.RUNNING or not self._host.worker_is_up():
            self._thumbs.set_worker_queue(WorkerQueue.PAUSED)
        else:
            self._thumbs.set_worker_queue(WorkerQueue.OPEN)

    def worker_ready(self):
        self._pause_thumbnails()
        self.preview.worker_ready()
        self._sync()

    def requests_lost(self):
        """The worker went away: nothing it was asked will answer."""
        self._thumbs.set_worker_queue(WorkerQueue.PAUSED)
        jobs, self._thumb_jobs = self._thumb_jobs, {}
        for done in jobs.values():
            done(Conversion.LOST)
        self.preview.requests_lost()
        if self._run is Run.RUNNING:
            self._end_run()
        self._pause_thumbnails()
        self._sync()

    def shutdown(self):
        self._thumbs.shutdown()
