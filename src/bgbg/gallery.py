"""The Batch page's gallery: a folder's images as tiles to keep or leave out.

The tiles live in a Gtk.GridView, which builds widgets only for the tiles on
screen and reuses them while scrolling. The gallery shows the folder a page
at a time; a "+" tile at the end reveals the next page.

Thumbnails decode on a small thread pool, at thumbnail size, and come back
to the main loop in batches (Thumbnails). A format GdkPixbuf can't decode
(HEIC, AVIF, WebP on Windows) is shrunk by the worker instead.
"""
import collections
import concurrent.futures
import enum
import os
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from gi.repository import Adw, Gdk, GdkPixbuf, Gio, GLib, GObject, Gtk, Pango

import formats


class Selection(enum.Enum):
    """Whether the next run includes an image."""

    KEPT = "kept"
    LEFT_OUT = "left_out"


class TileState(enum.Enum):
    """Where an image is in the current or last run."""

    IDLE = "idle"            # not in a run
    QUEUED = "queued"        # in the run, not reached yet
    WORKING = "working"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"      # the worker left it out (an animation it found)


class Face(enum.Enum):
    """Which picture a finished tile shows."""

    SOURCE = "source"
    RESULT = "result"


class Thumb(enum.Enum):
    """A thumbnail that isn't a picture (yet)."""

    LOADING = "loading"
    BROKEN = "broken"        # neither GdkPixbuf nor the worker could read it


class Editing(enum.Enum):
    """Whether the selection can change: not while a run uses it."""

    OPEN = "open"
    LOCKED = "locked"


class GalleryItem(GObject.Object):
    """One image of the folder, and everything a tile shows about it.

    Emits "changed" whenever any of it changes; the tile bound to it redraws.
    """

    __gsignals__ = {"changed": (GObject.SignalFlags.RUN_FIRST, None, ())}

    def __init__(self, path, animation):
        super().__init__()
        self.path = path
        self.name = os.path.basename(path)
        self.animation = animation
        # Batch runs stills only, so an animation is never in a run.
        self.selection = Selection.KEPT if animation is formats.Animation.STILL \
            else Selection.LEFT_OUT
        self.state = TileState.IDLE
        self.message = None          # FAILED, SKIPPED: why
        self.output = None           # DONE: the file written
        self.result_thumb = None     # DONE: the worker's small PNG of the result
        self.face = Face.SOURCE
        self.source = Thumb.LOADING  # a Gdk.Texture once decoded
        self.result = Thumb.LOADING

    @property
    def selectable(self):
        return self.animation is formats.Animation.STILL

    def reset_result(self):
        """Forget the last run's result picture, before a new run."""
        self.result = Thumb.LOADING
        self.face = Face.SOURCE

    def changed(self):
        self.emit("changed")


class MoreItem(GObject.Object):
    """The "+" tile: how many images are still hidden."""

    def __init__(self):
        super().__init__()
        self.remaining = 0


class Thumbnails:
    """Decodes thumbnails off the main thread, and keeps the recent ones.

    Each request names the folder generation it was made for: a new folder
    bumps it, so its predecessor's queued decodes are cancelled and late
    results are dropped. Results reach the main loop through one idle
    callback per batch of results, not one per image.
    """

    SIDE = 256               # longest side of a thumbnail, in pixels
    CACHE_SIZE = 160         # textures kept, most recently used first
    WORKER_IN_FLIGHT = 2     # worker conversions at once: a run waits behind them

    def __init__(self, tmpdir, worker_convert):
        """`worker_convert(path, out_png, side, done)` asks the worker for a PNG
        of `path`; done(Conversion) runs on the main thread once it answers."""
        self._tmpdir = tmpdir
        self._worker_convert = worker_convert
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="bgbg-thumbs")
        self._cache = collections.OrderedDict()     # path -> (mtime_ns, texture)
        self.generation = 0
        self._futures = set()
        self._lock = threading.Lock()
        # Decoded on the pool, waiting for the main loop: _Decoded each.
        self._results = collections.deque()
        self._drain_scheduled = False
        # Waiting for the worker: (generation, path, mtime_ns, apply) each.
        self._waiting = collections.deque()
        self._in_flight = 0
        self._queue = WorkerQueue.OPEN
        self._made = 0                              # worker PNGs so far, to name each one

    def new_generation(self):
        """A new folder: drop everything still queued for the old one."""
        self.generation += 1
        for future in list(self._futures):
            future.cancel()
        self._futures.clear()
        self._waiting.clear()

    def request(self, path, apply):
        """Call apply(texture or Thumb.BROKEN) on the main thread, once decoded.

        A cached texture is applied at once.
        """
        try:
            mtime = os.stat(path).st_mtime_ns
        except OSError:
            apply(Thumb.BROKEN)
            return
        cached = self._cache.get(path)
        if cached is not None and cached[0] == mtime:
            self._cache.move_to_end(path)
            apply(cached[1])
            return
        self._submit(self.generation, path, mtime, apply, Fallback.WORKER)

    def set_worker_queue(self, queue):
        """WorkerQueue.PAUSED while a run holds the worker, or there is none:
        conversions then wait here, not in the worker's queue."""
        self._queue = queue
        self._next_worker_job()

    def shutdown(self):
        self.new_generation()
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---- decoding ----
    def _submit(self, generation, path, mtime, apply, fallback, source=None):
        """Decode `source` (default: `path`) on the pool; cache it as `path`."""
        future = self._pool.submit(self._decode, source or path)
        self._futures.add(future)

        def finished(f):
            # On the pool's thread: hand the result to the main loop, one
            # idle callback for however many results are waiting.
            if f.cancelled():
                # Called by whoever cancelled it; the worker's PNG is ours to delete.
                if source is not None:
                    _remove_quietly(source)
                return
            with self._lock:
                self._results.append(_Decoded(generation, path, mtime, apply, fallback,
                                              source, f.result()))
                if self._drain_scheduled:
                    return
                self._drain_scheduled = True
            GLib.idle_add(self._drain, priority=GLib.PRIORITY_DEFAULT)
        future.add_done_callback(finished)

    def _decode(self, path):
        """A texture of `path` at thumbnail size, or None if GdkPixbuf can't read it."""
        try:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(path, self.SIDE, self.SIDE, True)
        except GLib.Error:
            return None
        pixbuf = pixbuf.apply_embedded_orientation() or pixbuf
        return Gdk.Texture.new_for_pixbuf(pixbuf)

    def _drain(self):
        with self._lock:
            results, self._results = self._results, collections.deque()
            self._drain_scheduled = False
        self._futures = {f for f in self._futures if not f.done()}
        for done in results:
            if done.source is not None:
                _remove_quietly(done.source)
            if done.generation != self.generation:
                continue
            if done.texture is not None:
                self._remember(done.path, done.mtime, done.texture)
                done.apply(done.texture)
            elif done.fallback is Fallback.WORKER:
                self._waiting.append((done.generation, done.path, done.mtime, done.apply))
                self._next_worker_job()
            else:
                done.apply(Thumb.BROKEN)
        return False

    def _remember(self, path, mtime, texture):
        self._cache[path] = (mtime, texture)
        self._cache.move_to_end(path)
        while len(self._cache) > self.CACHE_SIZE:
            self._cache.popitem(last=False)

    # ---- the worker, for formats GdkPixbuf can't read ----
    def _next_worker_job(self):
        while (self._queue is WorkerQueue.OPEN and self._waiting
               and self._in_flight < self.WORKER_IN_FLIGHT):
            job = self._waiting.popleft()
            generation, path, mtime, apply = job
            if generation != self.generation:
                continue
            self._made += 1
            out = os.path.join(self._tmpdir, "thumb-%d.png" % self._made)
            self._in_flight += 1

            def converted(outcome, job=job, out=out):
                self._in_flight = max(0, self._in_flight - 1)
                generation, path, mtime, apply = job
                if outcome is Conversion.DONE and generation == self.generation:
                    self._submit(generation, path, mtime, apply, Fallback.NONE, source=out)
                else:
                    _remove_quietly(out)
                    if outcome is Conversion.LOST:
                        self._waiting.appendleft(job)     # asked again once a worker is up
                    elif generation == self.generation:
                        apply(Thumb.BROKEN)
                self._next_worker_job()
            self._worker_convert(path, out, self.SIDE, converted)


@dataclass
class _Decoded:
    """A decode the pool finished, on its way to the main loop."""

    generation: int
    path: str                # the image the thumbnail is of
    mtime: int
    apply: Callable
    fallback: "Fallback"
    source: Optional[str]    # the worker's PNG that was decoded, deleted after
    texture: Optional[Gdk.Texture]   # None: GdkPixbuf couldn't read it


class Fallback(enum.Enum):
    """What a decode that GdkPixbuf can't do tries next."""

    WORKER = "worker"        # ask the worker for a PNG
    NONE = "none"            # it was the worker's PNG: show the broken picture


class WorkerQueue(enum.Enum):
    """Whether conversions go to the worker, or wait in the gallery."""

    OPEN = "open"
    PAUSED = "paused"


class Conversion(enum.Enum):
    """How a worker conversion for a thumbnail ended."""

    DONE = "done"
    FAILED = "failed"        # the worker can't read it either
    LOST = "lost"            # the worker went away before answering


def _remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


class _Square(Gtk.Widget):
    """Lays its child out as a square as wide as the grid's column.

    The grid gives each row the height its cells ask for at the column's
    width. Gtk.AspectFrame doesn't answer that question, so in one each row
    is only as tall as its pictures' natural height.
    """

    MIN_SIDE, NATURAL_SIDE = 130, 168

    def __init__(self, child):
        super().__init__()
        self.child = child
        child.set_parent(self)

    def do_get_request_mode(self):
        return Gtk.SizeRequestMode.HEIGHT_FOR_WIDTH

    def do_measure(self, orientation, for_size):
        if orientation == Gtk.Orientation.HORIZONTAL:
            return self.MIN_SIDE, self.NATURAL_SIDE, -1, -1
        side = for_size if for_size >= 0 else self.NATURAL_SIDE
        return side, side, -1, -1

    def do_size_allocate(self, width, height, baseline):
        self.child.allocate(width, height, baseline, None)

    def do_dispose(self):
        if self.child is not None:
            self.child.unparent()
            self.child = None
        Gtk.Widget.do_dispose(self)


class _Tile(Gtk.Overlay):
    """The widget that shows one GalleryItem; the grid reuses it for others."""

    def __init__(self, gallery):
        super().__init__()
        self._gallery = gallery
        self._item = None
        self._handler = 0
        self._syncing = False
        self.add_css_class("batch-tile")
        self.set_overflow(Gtk.Overflow.HIDDEN)

        self._placeholder = Gtk.Image.new_from_icon_name("image-x-generic-symbolic")
        self._placeholder.set_pixel_size(32)
        self._placeholder.add_css_class("dim-label")
        self._picture = Gtk.Picture(content_fit=Gtk.ContentFit.COVER, can_shrink=True)
        self._picture.add_css_class("tile-picture")
        under = Gtk.Overlay()
        under.set_child(self._placeholder)
        under.add_overlay(self._picture)
        self.set_child(under)

        self._name = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.END)
        self._name.add_css_class("tile-name")
        self._name.set_valign(Gtk.Align.END)
        self._name.set_can_target(False)
        self.add_overlay(self._name)

        self._badge = Gtk.Label(label="Single only")
        self._badge.add_css_class("tile-badge")
        self._badge.set_halign(Gtk.Align.START)
        self._badge.set_valign(Gtk.Align.END)
        self._badge.set_can_target(False)
        self.add_overlay(self._badge)

        # In the corner, so a done tile's tick doesn't read as a second
        # selection check, and the picture stays in view.
        self._state = Gtk.Box()
        self._state.add_css_class("tile-state")
        self._state.set_halign(Gtk.Align.END)
        self._state.set_valign(Gtk.Align.END)
        self._state_icon = Gtk.Image()
        self._state_icon.set_pixel_size(16)
        self._spinner = Adw.Spinner()
        self._spinner.set_size_request(16, 16)
        self._state.append(self._state_icon)
        self._state.append(self._spinner)
        self.add_overlay(self._state)

        self._check = Gtk.CheckButton()
        self._check.add_css_class("selection-mode")
        self._check.add_css_class("tile-check")
        self._check.set_halign(Gtk.Align.START)
        self._check.set_valign(Gtk.Align.START)
        self._check.connect("toggled", self._on_check)
        self.add_overlay(self._check)

        self._flip = Gtk.ToggleButton(icon_name="object-flip-horizontal-symbolic")
        self._flip.add_css_class("osd")
        self._flip.add_css_class("circular")
        self._flip.add_css_class("tile-flip")
        self._flip.set_halign(Gtk.Align.END)
        self._flip.set_valign(Gtk.Align.START)
        self._flip.connect("toggled", self._on_flip)
        self.add_overlay(self._flip)

    def bind(self, item):
        self._item = item
        self._handler = item.connect("changed", lambda *_: self.sync())
        self._gallery.want_thumbnail(item, Face.SOURCE)
        self.sync()

    def unbind(self):
        if self._item is not None and self._handler:
            self._item.disconnect(self._handler)
        self._item, self._handler = None, 0
        self._spinner.set_visible(False)        # a hidden spinner doesn't animate

    def sync(self):
        """Show the bound item as it is now."""
        item = self._item
        if item is None:
            return
        self._syncing = True
        kept = item.selection is Selection.KEPT
        for cls, on in (("kept", kept), ("left-out", not kept)):
            (self.add_css_class if on else self.remove_css_class)(cls)
        self._check.set_active(kept)
        self._check.set_visible(item.selectable)
        self._check.set_sensitive(self._gallery.editing is Editing.OPEN)
        self._badge.set_visible(not item.selectable)
        # Single animates by the file's name: an animated .png opens as a still there.
        self._badge.set_text("Single only" if formats.can_animate(item.path) else "Animated")
        self._name.set_text(item.name)

        face = item.face if item.state is TileState.DONE else Face.SOURCE
        shown = item.result if face is Face.RESULT else item.source
        self._picture.set_paintable(shown if isinstance(shown, Gdk.Texture) else None)
        # A dimmed picture would let it show through.
        self._placeholder.set_visible(not isinstance(shown, Gdk.Texture))
        self._placeholder.set_from_icon_name(
            "image-missing-symbolic" if shown is Thumb.BROKEN else "image-x-generic-symbolic")
        self._flip.set_visible(item.state is TileState.DONE)
        self._flip.set_active(face is Face.RESULT)
        self._flip.set_tooltip_text("Show the original" if face is Face.RESULT
                                    else "Show the result")
        self._show_state(item)
        self.set_tooltip_text(_tooltip(item))
        self._syncing = False

    def _show_state(self, item):
        icon = _STATE_ICONS.get(item.state)
        working = item.state is TileState.WORKING
        self._state.set_visible(icon is not None or working)
        self._state_icon.set_visible(icon is not None)
        if icon is not None:
            self._state_icon.set_from_icon_name(icon)
        self._spinner.set_visible(working)
        for state in TileState:
            self._state.remove_css_class(state.value)
        self._state.add_css_class(item.state.value)

    def _on_check(self, check):
        if self._syncing or self._item is None:
            return
        self._gallery.set_selection(
            self._item, Selection.KEPT if check.get_active() else Selection.LEFT_OUT)

    def _on_flip(self, button):
        if self._syncing or self._item is None:
            return
        self._gallery.flip(self._item, Face.RESULT if button.get_active() else Face.SOURCE)


# What a tile shows over its picture in each state; WORKING has a spinner.
_STATE_ICONS = {
    TileState.QUEUED: "content-loading-symbolic",
    TileState.DONE: "object-select-symbolic",
    TileState.FAILED: "dialog-warning-symbolic",
    TileState.SKIPPED: "action-unavailable-symbolic",
}


def _tooltip(item):
    """The file's name, then what happened to it, in words."""
    if not item.selectable:
        if formats.can_animate(item.path):
            return "%s\nAnimated: open it on the Single page, which keeps every frame." % item.name
        return "%s\nAnimated: a batch would keep only its first frame." % item.name
    lines = [item.name]
    if item.state is TileState.QUEUED:
        lines.append("Waiting its turn")
    elif item.state is TileState.WORKING:
        lines.append("Removing the background…")
    elif item.state is TileState.DONE:
        lines.append("Done: " + os.path.basename(item.output or ""))
    elif item.state in (TileState.FAILED, TileState.SKIPPED):
        lines.append(("Failed: " if item.state is TileState.FAILED else "Skipped: ")
                     + (item.message or ""))
    elif item.selection is Selection.LEFT_OUT:
        lines.append("Left out of the next run")
    return "\n".join(lines)


class _MoreTile(Gtk.Button):
    """The "+" tile that reveals the next page of images."""

    def __init__(self, gallery):
        super().__init__()
        self.add_css_class("more-tile")
        self.add_css_class("flat")
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.set_valign(Gtk.Align.CENTER)
        icon = Gtk.Image.new_from_icon_name("list-add-symbolic")
        icon.set_pixel_size(28)
        self._title = Gtk.Label()
        self._title.add_css_class("heading")
        self._left = Gtk.Label()
        self._left.add_css_class("dim-label")
        for child in (icon, self._title, self._left):
            box.append(child)
        self.set_child(box)
        self.connect("clicked", lambda *_: gallery.show_more())

    def unbind(self):
        pass

    def bind(self, item):
        if item.remaining > Gallery.PAGE:
            self._title.set_text("Show %d more" % Gallery.PAGE)
            self._left.set_text("%d left" % item.remaining)
        else:
            self._title.set_text("Show the last %d" % item.remaining)
            self._left.set_text("")
        self._left.set_visible(item.remaining > Gallery.PAGE)


class Gallery:
    """The grid of tiles over a folder's images, and their selection.

    `on_changed()` runs on the main thread whenever the selection changes.
    """

    PAGE = 20                # tiles revealed at once

    def __init__(self, thumbnails, on_changed):
        self._thumbs = thumbnails
        self._on_changed = on_changed
        self.items = []                      # every image of the folder, in order
        self.editing = Editing.OPEN
        self._by_path = {}
        self._shown = 0
        self._more = MoreItem()
        self._store = Gio.ListStore.new(GObject.Object)

        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._setup)
        factory.connect("bind", self._bind)
        factory.connect("unbind", self._unbind)
        self.view = Gtk.GridView(model=Gtk.NoSelection(model=self._store), factory=factory)
        self.view.add_css_class("batch-gallery")
        self.view.set_min_columns(2)
        self.view.set_max_columns(12)
        # One click toggles a tile (or opens the "+" tile); Enter and Space do too.
        self.view.set_single_click_activate(True)
        self.view.connect("activate", self._on_activate)

    # ---- the folder ----
    def set_images(self, images):
        """Show a new folder's images: [(path, formats.Animation)], in order."""
        self._thumbs.new_generation()
        self.items = [GalleryItem(path, animation) for path, animation in images]
        self._by_path = {item.path: item for item in self.items}
        self._shown = 0
        self._store.remove_all()
        self._reveal(self.PAGE)
        self._on_changed()

    def show_more(self):
        self._reveal(self.PAGE)

    def _reveal(self, count):
        new = self.items[self._shown:self._shown + count]
        self._shown += len(new)
        if self._store.get_n_items() and self._store.get_item(
                self._store.get_n_items() - 1) is self._more:
            self._store.remove(self._store.get_n_items() - 1)
        self._store.splice(self._store.get_n_items(), 0, new)
        self._more.remaining = len(self.items) - self._shown
        if self._more.remaining > 0:
            self._store.append(self._more)

    def item_for(self, path):
        return self._by_path.get(path)

    # ---- selection ----
    def set_editing(self, editing):
        if editing is self.editing:
            return
        self.editing = editing
        for item in self.items:
            item.changed()

    def set_selection(self, item, selection):
        if self.editing is Editing.LOCKED or not item.selectable or item.selection is selection:
            return
        item.selection = selection
        item.changed()
        self._on_changed()

    def select_all(self, selection):
        """Keep or leave out every image, shown or not."""
        if self.editing is Editing.LOCKED:
            return
        for item in self.items:
            if item.selectable and item.selection is not selection:
                item.selection = selection
                item.changed()
        self._on_changed()

    def selected(self):
        return [item for item in self.items if item.selection is Selection.KEPT]

    def selectable(self):
        return [item for item in self.items if item.selectable]

    # ---- thumbnails ----
    def want_thumbnail(self, item, face):
        """Decode the picture a tile needs, unless it is there already."""
        if face is Face.SOURCE:
            if item.source is not Thumb.LOADING:
                return

            def apply(texture):
                item.source = texture
                item.changed()
            self._thumbs.request(item.path, apply)
        elif item.result is Thumb.LOADING and item.result_thumb:
            def apply(texture):
                item.result = texture
                item.changed()
            self._thumbs.request(item.result_thumb, apply)

    def flip(self, item, face):
        item.face = face
        if face is Face.RESULT:
            self.want_thumbnail(item, Face.RESULT)
        item.changed()

    # ---- the grid ----
    def _setup(self, _factory, list_item):
        list_item.set_child(_Square(_Tile(self)))

    def _bind(self, _factory, list_item):
        # A cell is a tile, or the one "+" tile: swapped here when the grid
        # reuses a cell for the other kind, which is rare.
        item = list_item.get_item()
        kind = _MoreTile if isinstance(item, MoreItem) else _Tile
        cell = list_item.get_child()
        if not isinstance(cell.child, kind):
            cell = _Square(kind(self))
            list_item.set_child(cell)
        cell.child.bind(item)

    def _unbind(self, _factory, list_item):
        list_item.get_child().child.unbind()

    def _on_activate(self, _view, position):
        item = self._store.get_item(position)
        if isinstance(item, MoreItem):
            self.show_more()
        elif item.selectable:
            self.set_selection(item, Selection.LEFT_OUT if item.selection is Selection.KEPT
                               else Selection.KEPT)
