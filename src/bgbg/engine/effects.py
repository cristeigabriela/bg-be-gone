"""What the engine asks the host to do. Stdlib only.

The engine never calls GTK and never touches the DOM. It *returns* its side
effects and the shell applies them — so the same event sequence can be replayed
in a test with no UI at all, and asserted against an expected list of effects.
That is the whole point: interaction becomes data.

Effects compare by value, so a test can write the expectation literally:

    assert sess.feed(PointerMove(120, 90)) == [RequestTick(), Redraw(),
                                               HoverChanged(1, False, 120, 90)]
"""


class Effect:
    """Base: value equality + a readable repr, driven by __slots__."""

    __slots__ = ()

    def _fields(self):
        return tuple(getattr(self, f) for f in self.__slots__)

    def __eq__(self, other):
        return (type(self) is type(other)
                and self._fields() == other._fields())

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((type(self).__name__,) + self._fields())

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, ", ".join(
            "%s=%r" % (f, getattr(self, f)) for f in self.__slots__))


class Redraw(Effect):
    """The frame is stale — repaint."""

    __slots__ = ()


class RequestTick(Effect):
    """Something is animating: start the frame clock if it is idle.

    Separate from Redraw because arming the clock is not the same as painting
    once, and because a host may have no clock yet (an unrealized widget) and has
    to re-arm later — see AnimState.needs_tick.
    """

    __slots__ = ()


class StopTick(Effect):
    """Nothing is animating any more — retire the frame clock.

    The tick would retire itself on its next pass anyway (`needs_tick` goes
    false), so this only saves a frame; it is here so that "stop animating" is
    something the engine can *say*, rather than something the host has to infer.
    """

    __slots__ = ()


class SetCursor(Effect):
    """`name` is a CSS/GTK cursor name, or None to restore the default."""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name


class GrabFocus(Effect):
    """Take keyboard focus (so Space-to-pan reaches us, not a focused button)."""

    __slots__ = ()


class ViewChanged(Effect):
    """The image transform changed — the host's toolbar/labels should refresh."""

    __slots__ = ()


class ContextMenu(Effect):
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


class HoverChanged(Effect):
    """The hovered object stack changed (or the cursor moved within it).

    `depth` is how many objects overlap under the cursor and `drilled` whether we
    have dwelled long enough to focus the *part* rather than the whole; the host
    uses them to place the "3 objects here" badge.
    """

    __slots__ = ("depth", "drilled", "x", "y")

    def __init__(self, depth, drilled, x, y):
        self.depth = depth
        self.drilled = drilled
        self.x = x
        self.y = y


class SelectionChanged(Effect):
    """`ids` is the full selection after the change, sorted."""

    __slots__ = ("ids",)

    def __init__(self, ids):
        self.ids = tuple(ids)


class SegClick(Effect):
    """A click while a segmentation mode is active.

    everything mode: `value` is the object id that was toggled (0 if none).
    point mode:      `value` is the label — 1 positive, 0 negative.
    """

    __slots__ = ("ix", "iy", "value", "kind")

    def __init__(self, ix, iy, value, kind):
        self.ix = ix
        self.iy = iy
        self.value = value
        self.kind = kind
