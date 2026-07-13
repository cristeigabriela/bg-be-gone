"""The animation state machine. Stdlib only.

One clock drives everything the canvas animates: the breathing glow, the marching
ants, the scan shimmer, the overlay reveal, the select "pop", the hover-dwell
drill, the press-and-hold ripple, and the outline morph.

Every timestamp is *absolute* microseconds from the host's frame clock, never a
delta, so a renderer can be asked for "the scene at time T" and get the same
answer twice. That is what makes the golden corpus in `spec/` possible: the
offscreen rasteriser has no frame clock (now == 0), so a fixture states an age
("pressed 300 ms ago") and it becomes a negative timestamp.

The derived getters (`pop_scale`, `press`, `waves`, `morph_progress`) are here
rather than in the renderer on purpose: they are pure maths that both the GSK
backend and the browser backend need, so they belong to the shared core.
"""
import math

from .geometry import ease_out, ease_out_back

# ---- timings (the single source of truth; the TS mirror reads the same numbers)
REVEAL_MS = 300.0        # overlays fade + scale in
POP_MS = 300.0           # tactile bump when an object is selected
DWELL_MS = 1600.0        # hover this long to drill general -> specific
MORPH_MS = 240.0         # outline tween when the focused object changes
MORPH_N = 64             # resampled outline vertices (lerp correspondence)

# press-and-hold "swizzle": a wave spawns every SPAWN ms while held and lives
# WAVE ms, so a quick tap still plays one full wave that outlives the click.
PRESS_WAVE_MS = 720.0
PRESS_SPAWN_MS = 450.0
PRESS_DECAY_MS = 460.0   # glow/tint fade after release
PRESS_SPRING_MS = 360.0  # scale springs back with overshoot after release

PULSE_HZ = 1.1           # breathing glow
ANT_SPEED = 24.0         # marching-ants dash travel, px/s
SCAN_PERIOD_S = 1.2      # shimmer sweep loop


class Tick:
    """What one `advance()` changed."""

    __slots__ = ("animating", "dwell_fired", "press_retired", "morph_retired")

    def __init__(self, animating=False, dwell_fired=False,
                 press_retired=False, morph_retired=False):
        self.animating = animating
        self.dwell_fired = dwell_fired
        self.press_retired = press_retired
        self.morph_retired = morph_retired

    @property
    def changed(self):
        return self.dwell_fired or self.press_retired or self.morph_retired


class Press:
    """The press-and-hold state at a moment in time."""

    __slots__ = ("held", "fade", "scale", "glow", "tint", "age_ms")

    def __init__(self, held, fade, scale, glow, tint, age_ms):
        self.held = held        # still holding the button down
        self.fade = fade        # 1 while held, decays to 0 after release
        self.scale = scale      # 0.97 held; springs back past 1.0 on release
        self.glow = glow        # glow alpha
        self.tint = tint        # fill alpha
        self.age_ms = age_ms    # ms since the press began


class AnimState:
    def __init__(self, dwell_ms=DWELL_MS):
        self.t0 = None           # animation epoch (µs)
        # time-derived values, refreshed every advance()
        self.pulse = 0.0         # 0..1 breathing
        self.ant = 0.0           # dash phase, px
        self.scan_phase = 0.0    # 0..1 shimmer sweep
        self.reveal = 1.0        # 0..1 overlay fade/scale-in

        self.scanning = False
        self.reveal_t0 = None
        self.pop = {}            # object id -> start µs

        self.dwell_ms = float(dwell_ms)
        self.dwell_t0 = None
        self.drilled = False

        self.press_obj = 0
        self.press_pt = None     # (ix, iy) image px
        self.press_t0 = None
        self.release_t0 = None   # None while held

        self.morph_t0 = None

    # ---------- lifecycle ----------
    def reset(self):
        self.t0 = None
        self.reveal, self.reveal_t0 = 1.0, None
        self.pop = {}
        self.scanning = False
        self.cancel_dwell()
        self.clear_press()
        self.morph_t0 = None

    def begin_reveal(self, now):
        self.reveal = 0.0
        self.reveal_t0 = now

    def begin_pop(self, oid, now):
        self.pop[oid] = now

    def begin_dwell(self, now):
        self.dwell_t0 = now
        self.drilled = False

    def cancel_dwell(self):
        self.dwell_t0 = None
        self.drilled = False

    def begin_press(self, oid, pt, now):
        self.press_obj = oid
        self.press_pt = pt
        self.press_t0 = now
        self.release_t0 = None

    def release_press(self, now):
        """Stamp the release; the ripple keeps playing until it has decayed."""
        if self.press_obj and self.release_t0 is None:
            self.release_t0 = now
            return True
        return False

    def clear_press(self):
        self.press_obj = 0
        self.press_pt = None
        self.press_t0 = None
        self.release_t0 = None

    def begin_morph(self, now):
        self.morph_t0 = now

    def clear_morph(self):
        self.morph_t0 = None

    @property
    def pressing(self):
        """A press animation is alive (held, or still decaying after release)."""
        return bool(self.press_obj)

    @property
    def held(self):
        return bool(self.press_obj) and self.release_t0 is None

    def needs_tick(self, active=False):
        """Is anything animating? `active` is whatever the host owns (a hovered
        object, a selection, a point mask). The host uses this both to stop an
        idle tick and to *re-arm* one — a clock can be started while the widget
        is unrealized (no frame clock yet), and something has to notice later.
        """
        return bool(active or self.press_obj or self.scanning
                    or self.morph_t0 is not None
                    or self.reveal_t0 is not None or self.pop)

    # ---------- the pump ----------
    def advance(self, now, active=False):
        """Refresh time-derived values and retire finished animations.

        `active` is whatever else wants the clock running (a hovered object, a
        selection, a point mask) — state the engine doesn't own yet. Returns a
        Tick; when `animating` is false the host should stop the frame callback,
        which is what keeps an idle window from waking 60 times a second.
        """
        if self.t0 is None:
            self.t0 = now
        el = (now - self.t0) / 1_000_000.0

        self.pulse = 0.5 * (1.0 + math.sin(el * 2.0 * math.pi * PULSE_HZ))
        self.ant = el * ANT_SPEED
        if self.scanning:
            self.scan_phase = (el / SCAN_PERIOD_S) % 1.0

        if self.reveal_t0 is not None:
            p = (now - self.reveal_t0) / 1000.0 / REVEAL_MS
            if p >= 1.0:
                self.reveal, self.reveal_t0 = 1.0, None
            else:
                self.reveal = ease_out(p)

        for oid in [k for k, t in self.pop.items()
                    if (now - t) / 1000.0 > POP_MS]:
            del self.pop[oid]

        dwell_fired = False
        if (self.dwell_t0 is not None and not self.drilled
                and (now - self.dwell_t0) / 1000.0 > self.dwell_ms):
            self.drilled = True
            dwell_fired = True

        press_retired = False
        if (self.press_obj and self.release_t0 is not None
                and (now - self.release_t0) / 1000.0 > PRESS_WAVE_MS):
            self.clear_press()
            press_retired = True

        morph_retired = False
        if (self.morph_t0 is not None
                and (now - self.morph_t0) / 1000.0 >= MORPH_MS):
            self.morph_t0 = None
            morph_retired = True

        return Tick(self.needs_tick(active), dwell_fired,
                    press_retired, morph_retired)

    # ---------- derived values the renderer needs ----------
    def pop_scale(self, oid, now):
        """The tactile bump applied to a freshly selected object."""
        t = self.pop.get(oid)
        if t is None:
            return 1.0
        p = (now - t) / 1000.0 / POP_MS
        if p <= 0.0 or p >= 1.0:
            return 1.0
        return 1.0 + 0.09 * math.sin(p * math.pi) * (1.0 - p)

    def press(self, now):
        """The press-and-hold envelope at `now`, or None if nothing is pressed."""
        if not self.press_obj or self.press_t0 is None:
            return None
        held = self.release_t0 is None
        rel_ms = 0.0 if held else (now - self.release_t0) / 1000.0
        fade = 1.0 if held else max(0.0, 1.0 - rel_ms / PRESS_DECAY_MS)
        if held:
            scale = 0.97
        else:
            scale = 0.97 + 0.03 * ease_out_back(
                min(1.0, rel_ms / PRESS_SPRING_MS))
        glow = (0.34 + 0.24 * self.pulse) if held else 0.40 * fade
        tint = 0.30 * (1.0 if held else fade)
        return Press(held, fade, scale, glow, tint,
                     (now - self.press_t0) / 1000.0)

    def waves(self, now):
        """The ripple rings alive at `now`: a list of (phase, alpha).

        Waves spawn every PRESS_SPAWN_MS from the press until release; each lives
        PRESS_WAVE_MS. Only the ~2 that can still be on screen are considered, so
        a long hold does not walk a growing list.
        """
        p = self.press(now)
        if p is None:
            return []
        age_total = p.age_ms
        last_spawn = (age_total if p.held
                      else (self.release_t0 - self.press_t0) / 1000.0)
        n = int(last_spawn / PRESS_SPAWN_MS) + 1
        first = max(0, int((age_total - PRESS_WAVE_MS) / PRESS_SPAWN_MS))
        out = []
        for i in range(first, n):
            age = age_total - i * PRESS_SPAWN_MS
            if age < 0.0 or age >= PRESS_WAVE_MS:
                continue
            ph = age / PRESS_WAVE_MS
            out.append((ph, (1.0 - ph) * 0.5 * (1.0 if p.held else p.fade)))
        return out

    def morph_progress(self, now):
        """0..1 eased progress of the outline tween, or None when idle."""
        if self.morph_t0 is None:
            return None
        raw = (now - self.morph_t0) / 1000.0 / MORPH_MS
        if raw >= 1.0:
            return None
        return ease_out(raw)
