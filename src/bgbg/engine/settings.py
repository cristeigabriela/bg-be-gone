"""Settings as data: the schema, the value store, the visibility rules.

Stdlib only.

The UI does not know what settings exist. It asks `describe()` and builds itself:
a `choice` becomes an Adw.ComboRow on the desktop and a `<select>` on the web, and
neither one has a hardcoded list of models or backgrounds anywhere in it. Adding a
setting is a line in `interactables.py`, and both UIs grow the control.

The reactive rules are data too. "The exact-model row only appears for Custom
model…" used to be an `_on_subject_changed` handler wiring one widget to another;
it is now `visible_when=[eq("subject", "custom")]` on the setting itself, which is
the same fact stated once instead of once per UI.

Conditions read from an *environment*: every setting's current value, plus the
host's context (which page is showing, whether the worker reported SAM). That is
the only coupling between a setting and the app around it.
"""

CHOICE = "choice"      # one of a list          -> ComboRow / <select>
BOOL = "bool"          # on/off                 -> SwitchRow / <input type=checkbox>
INT = "int"            # a bounded number       -> SpinRow / <input type=number>
COLOR = "color"        # an #rrggbb colour      -> ColorDialogButton / <input type=color>
EXPANDER = "expander"  # a disclosure of others -> ExpanderRow / <details>

#: Where a setting is shown. Most live in the sidebar; a few are page furniture
#: (the Everything/Click toggle), and they still belong to the catalogue.
SIDEBAR = "sidebar"
PAGE = "page"


class Choice:
    """One option of a `choice` setting: the value we store, the label we show."""

    __slots__ = ("value", "label")

    def __init__(self, value, label):
        self.value = value
        self.label = label

    def __repr__(self):
        return "Choice(%r, %r)" % (self.value, self.label)


class Cond:
    """A visibility/sensitivity rule, as data rather than as a signal handler."""

    __slots__ = ("key", "op", "value")

    EQ = "=="
    NE = "!="
    TRUTHY = "truthy"

    def __init__(self, key, op, value=None):
        self.key = key
        self.op = op
        self.value = value

    def test(self, env):
        v = env.get(self.key)
        if self.op == Cond.EQ:
            return v == self.value
        if self.op == Cond.NE:
            return v != self.value
        return bool(v)

    def __repr__(self):
        return "Cond(%r %s %r)" % (self.key, self.op, self.value)


def eq(key, value):
    return Cond(key, Cond.EQ, value)


def ne(key, value):
    return Cond(key, Cond.NE, value)


def truthy(key):
    return Cond(key, Cond.TRUTHY)


class Setting:
    __slots__ = ("id", "kind", "label", "subtitle", "default", "options",
                 "minimum", "maximum", "step", "page_step", "children",
                 "visible_when", "sensitive_when", "surface")

    def __init__(self, id, kind, label, subtitle=None, default=None,
                 options=(), minimum=None, maximum=None, step=1, page_step=5,
                 children=(), visible_when=(), sensitive_when=(),
                 surface=SIDEBAR):
        self.id = id
        self.kind = kind
        self.label = label
        self.subtitle = subtitle
        self.default = default
        self.options = list(options)
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.page_step = page_step      # how far a PageUp moves an INT
        self.children = list(children)
        self.visible_when = list(visible_when)
        self.sensitive_when = list(sensitive_when)
        self.surface = surface

    def walk(self):
        """This setting and, recursively, anything nested under it."""
        yield self
        for c in self.children:
            yield from c.walk()


class Group:
    __slots__ = ("id", "title", "settings", "visible_when")

    def __init__(self, id, title, settings=(), visible_when=()):
        self.id = id
        self.title = title
        self.settings = list(settings)
        self.visible_when = list(visible_when)


class Schema:
    def __init__(self, groups=()):
        self.groups = list(groups)

    def walk(self):
        for g in self.groups:
            for s in g.settings:
                yield from s.walk()

    def get(self, sid):
        for s in self.walk():
            if s.id == sid:
                return s
        raise KeyError(sid)


class Settings:
    """The live values, plus the host context the rules read."""

    def __init__(self, schema, context=None):
        self.schema = schema
        self.values = {s.id: s.default for s in schema.walk()
                       if s.kind != EXPANDER}
        # Options can be replaced at runtime — the SAM ladder only exists once
        # the worker has told us which models this GPU can actually hold.
        self._options = {s.id: list(s.options) for s in schema.walk()}
        self.context = dict(context or {})

    # ---------- values ----------
    def get(self, sid):
        return self.values[sid]

    def set(self, sid, value):
        """Returns True if the value actually changed."""
        if self.values.get(sid) == value:
            return False
        self.values[sid] = value
        return True

    def env(self):
        e = dict(self.values)
        e.update(self.context)
        return e

    def set_context(self, **kw):
        self.context.update(kw)

    # ---------- options ----------
    def options(self, sid):
        return self._options[sid]

    def set_options(self, sid, choices):
        """Replace a choice list at runtime, keeping the value valid."""
        self._options[sid] = list(choices)
        if not any(c.value == self.values.get(sid) for c in self._options[sid]):
            self.values[sid] = (self._options[sid][0].value
                                if self._options[sid] else None)

    def index(self, sid):
        """The position of the current value in the option list (0 if absent)."""
        for i, c in enumerate(self._options[sid]):
            if c.value == self.values.get(sid):
                return i
        return 0

    def set_index(self, sid, i):
        opts = self._options[sid]
        if 0 <= i < len(opts):
            return self.set(sid, opts[i].value)
        return False

    def label(self, sid):
        """The label of the current value — what a status line wants to show."""
        for c in self._options[sid]:
            if c.value == self.values.get(sid):
                return c.label
        return ""

    # ---------- rules ----------
    def visible(self, sid):
        env = self.env()
        return all(c.test(env) for c in self.schema.get(sid).visible_when)

    def sensitive(self, sid):
        env = self.env()
        return all(c.test(env) for c in self.schema.get(sid).sensitive_when)

    def group_visible(self, gid):
        env = self.env()
        for g in self.schema.groups:
            if g.id == gid:
                return all(c.test(env) for c in g.visible_when)
        raise KeyError(gid)

    # ---------- the UI contract ----------
    def describe(self, surface=None):
        """Everything a UI needs to build itself, as plain JSON-able data.

        This is the whole contract: both the GTK sidebar and the web sidebar are
        a loop over this, and neither knows what a "background" is.
        """
        return {"groups": [self._group(g, surface) for g in self.schema.groups]}

    def _group(self, g, surface):
        return {
            "id": g.id,
            "title": g.title,
            "visible": self.group_visible(g.id),
            "rows": [self._row(s) for s in g.settings
                     if surface is None or s.surface == surface],
        }

    def _row(self, s):
        d = {"id": s.id, "kind": s.kind, "label": s.label,
             "visible": self.visible(s.id), "sensitive": self.sensitive(s.id)}
        if s.subtitle:
            d["subtitle"] = s.subtitle
        if s.kind == EXPANDER:
            d["rows"] = [self._row(c) for c in s.children]
            return d
        d["value"] = self.values.get(s.id)
        if s.kind == CHOICE:
            d["options"] = [{"value": c.value, "label": c.label}
                            for c in self._options[s.id]]
            d["selected"] = self.index(s.id)
        elif s.kind == INT:
            d["min"] = s.minimum
            d["max"] = s.maximum
            d["step"] = s.step
        return d
