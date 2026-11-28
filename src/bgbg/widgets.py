"""The buttons and action row every page shares, so the pages look alike.

Each page builds its own with these, wherever its module lives. Their look
is in app.py's CSS (`.pill`, `.action-bar`).
"""
from gi.repository import Adw, Gtk


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
