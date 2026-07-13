"""The sidebar, built from the engine's schema rather than from hardcoded rows.

This file knows what a ComboRow is. It does not know what a "model" or a
"background" is, and it contains no list of either — it walks
`engine.settings.Settings` and makes a widget per setting. Adding a setting is a
line in `engine/interactables.py`; this grows the control for free, and so will
the browser's sidebar, from the same declaration.

The reactive rules come from the schema too: after any change we re-apply every
setting's `visible_when` / `sensitive_when`, which is what used to be a hand-wired
`_on_subject_changed` -> `model_row.set_visible(...)` per pair of widgets.
"""
import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Gdk, Adw  # noqa: E402

from engine.settings import (  # noqa: E402
    CHOICE, BOOL, INT, COLOR, EXPANDER, SIDEBAR,
)


def _hex(rgba):
    return "#%02x%02x%02x" % (round(rgba.red * 255), round(rgba.green * 255),
                              round(rgba.blue * 255))


class Sidebar:
    """Builds the rows, keeps them in step with the settings, reports changes.

    `on_change(setting_id)` fires only when a value actually changed, and only
    for user edits — never for the programmatic updates this class makes itself
    (reloading the SAM ladder would otherwise look like the user picking a model).
    """

    def __init__(self, settings, on_change=None):
        self.settings = settings
        self.on_change = on_change
        self.groups = {}        # group id   -> Adw.PreferencesGroup
        self.rows = {}          # setting id -> Adw row
        self.buttons = {}       # setting id -> the colour button, for COLOR rows
        self._syncing = False

        self.widgets = [self._group(g) for g in settings.schema.groups]
        self.sync()

    # ---------- building ----------
    def _group(self, g):
        grp = Adw.PreferencesGroup(title=g.title)
        self.groups[g.id] = grp
        for s in g.settings:
            if s.surface != SIDEBAR:      # page furniture, not a sidebar row
                continue
            row = self._row(s)
            if row is not None:
                grp.add(row)
        return grp

    def _row(self, s):
        row = self._make(s)
        if row is None:
            return None
        if s.subtitle:
            row.set_subtitle(s.subtitle)
        self.rows[s.id] = row
        if s.kind == EXPANDER:
            for child in s.children:
                sub = self._row(child)
                if sub is not None:
                    row.add_row(sub)
        return row

    def _make(self, s):
        st = self.settings
        if s.kind == CHOICE:
            row = Adw.ComboRow(
                title=s.label,
                model=Gtk.StringList.new([c.label for c in st.options(s.id)]))
            row.set_selected(st.index(s.id))
            row.connect("notify::selected", self._on_combo, s.id)
            return row

        if s.kind == BOOL:
            row = Adw.SwitchRow(title=s.label)
            row.set_active(bool(st.get(s.id)))
            row.connect("notify::active", self._on_switch, s.id)
            return row

        if s.kind == INT:
            row = Adw.SpinRow(
                title=s.label,
                adjustment=Gtk.Adjustment(
                    value=st.get(s.id), lower=s.minimum, upper=s.maximum,
                    step_increment=s.step, page_increment=s.page_step))
            row.connect("notify::value", self._on_spin, s.id)
            return row

        if s.kind == COLOR:
            row = Adw.ActionRow(title=s.label)
            btn = Gtk.ColorDialogButton.new(Gtk.ColorDialog())
            rgba = Gdk.RGBA()
            rgba.parse(st.get(s.id))
            btn.set_rgba(rgba)
            btn.set_valign(Gtk.Align.CENTER)
            btn.connect("notify::rgba", self._on_color, s.id)
            row.add_suffix(btn)
            self.buttons[s.id] = btn
            return row

        if s.kind == EXPANDER:
            return Adw.ExpanderRow(title=s.label)

        raise ValueError("unknown setting kind %r" % (s.kind,))

    # ---------- user edits ----------
    def _changed(self, sid, changed):
        if not changed:
            return
        self.sync()                       # a change can reveal or disable a row
        if self.on_change:
            self.on_change(sid)

    def _on_combo(self, row, _p, sid):
        if self._syncing:
            return
        self._changed(sid, self.settings.set_index(sid, row.get_selected()))

    def _on_switch(self, row, _p, sid):
        if self._syncing:
            return
        self._changed(sid, self.settings.set(sid, row.get_active()))

    def _on_spin(self, row, _p, sid):
        if self._syncing:
            return
        self._changed(sid, self.settings.set(sid, int(row.get_value())))

    def _on_color(self, btn, _p, sid):
        if self._syncing:
            return
        self._changed(sid, self.settings.set(sid, _hex(btn.get_rgba())))

    # ---------- keeping the widgets in step ----------
    def sync(self):
        """Re-apply every visibility/sensitivity rule from the schema."""
        st = self.settings
        for gid, grp in self.groups.items():
            grp.set_visible(st.group_visible(gid))
        for sid, row in self.rows.items():
            row.set_visible(st.visible(sid))
            row.set_sensitive(st.sensitive(sid))

    def reload_options(self, sid):
        """The options of a choice changed under us (the worker reported which
        SAM models this GPU can hold). Rebuild the list without it looking like
        the user picked something."""
        self._syncing = True
        try:
            row = self.rows[sid]
            row.set_model(Gtk.StringList.new(
                [c.label for c in self.settings.options(sid)]))
            row.set_selected(self.settings.index(sid))
        finally:
            self._syncing = False
