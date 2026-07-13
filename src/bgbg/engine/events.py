"""Input events the host feeds the engine. Stdlib only.

A GTK event controller and a DOM listener produce different objects; both can
produce *these*. Coordinates are always view pixels (widget-relative on the
desktop, canvas-relative in the browser).
"""

PRIMARY = 1
SECONDARY = 3

SPACE = "space"


class Event:
    __slots__ = ()

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, ", ".join(
            "%s=%r" % (f, getattr(self, f)) for f in self.__slots__))


class PointerEnter(Event):
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


class PointerMove(Event):
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


class PointerLeave(Event):
    __slots__ = ()


class PointerDown(Event):
    __slots__ = ("x", "y", "button", "n_press", "ctrl")

    def __init__(self, x, y, button=PRIMARY, n_press=1, ctrl=False):
        self.x = x
        self.y = y
        self.button = button
        self.n_press = n_press
        self.ctrl = ctrl


class PointerUp(Event):
    __slots__ = ("x", "y", "button", "ctrl")

    def __init__(self, x, y, button=PRIMARY, ctrl=False):
        self.x = x
        self.y = y
        self.button = button
        self.ctrl = ctrl


class Scroll(Event):
    """A wheel notch. Positive `dy` scrolls down (zoom out)."""

    __slots__ = ("dy",)

    def __init__(self, dy):
        self.dy = dy


class DragBegin(Event):
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


class DragUpdate(Event):
    """Offsets are cumulative from the drag's start, as GtkGestureDrag reports."""

    __slots__ = ("dx", "dy")

    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy


class KeyDown(Event):
    __slots__ = ("key",)

    def __init__(self, key):
        self.key = key


class KeyUp(Event):
    __slots__ = ("key",)

    def __init__(self, key):
        self.key = key
