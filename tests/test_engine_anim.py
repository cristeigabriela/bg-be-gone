#!/usr/bin/env python3
"""engine.anim — the animation state machine, driven by an injected clock.

No GTK, no frame clock: we hand it timestamps, so every curve, every retirement
deadline and the idle-shutdown gate are all directly assertable. That is the
whole point of moving the clocks into the engine.

Run: python tests/test_engine_anim.py
"""
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src", "bgbg"))
from engine.anim import (  # noqa: E402
    AnimState, REVEAL_MS, POP_MS, MORPH_MS,
    PRESS_WAVE_MS, PRESS_SPAWN_MS, PRESS_DECAY_MS, PRESS_SPRING_MS,
    ANT_SPEED, SCAN_PERIOD_S,
)

MS = 1000            # µs per ms
S = 1_000_000        # µs per second


def test_clocks():
    a = AnimState()
    a.scanning = True
    a.advance(0)                       # sets the epoch
    assert a.ant == 0.0

    a.advance(1 * S)
    assert abs(a.ant - ANT_SPEED) < 1e-9, "ants travel 24 px/s"
    # pulse is a 1.1 Hz sine: after 1s it is 0.5*(1+sin(2pi*1.1))
    assert abs(a.pulse - 0.5 * (1 + math.sin(2 * math.pi * 1.1))) < 1e-12
    # the shimmer wraps on a 1.2 s loop
    assert abs(a.scan_phase - (1.0 / SCAN_PERIOD_S) % 1.0) < 1e-12

    a.advance(int(SCAN_PERIOD_S * S))
    assert a.scan_phase < 1e-9, "shimmer wraps exactly at the period"
    print("  clocks: ants, pulse, shimmer wrap  OK")


def test_reveal():
    a = AnimState()
    a.advance(0)
    a.begin_reveal(0)
    assert a.reveal == 0.0

    a.advance(int(REVEAL_MS / 2 * MS))
    mid = a.reveal
    assert 0.0 < mid < 1.0
    assert mid > 0.5, "ease-out: more than half way by the half-way point"

    a.advance(int(REVEAL_MS * MS))      # exactly at the deadline -> done
    assert a.reveal == 1.0 and a.reveal_t0 is None
    print("  reveal: eases out and retires at %.0f ms  OK" % REVEAL_MS)


def test_pop():
    a = AnimState()
    a.advance(0)
    a.begin_pop(7, 0)
    assert a.pop_scale(7, 0) == 1.0                     # starts at rest
    assert a.pop_scale(7, int(POP_MS / 2 * MS)) > 1.0   # bumps out
    assert a.pop_scale(9, 0) == 1.0                     # unknown id -> no bump

    t = a.advance(int((POP_MS + 1) * MS))
    assert 7 not in a.pop, "a finished pop is retired"
    assert not t.animating, "and nothing else is running, so the tick stops"
    print("  pop: bumps then retires at %.0f ms  OK" % POP_MS)


def test_dwell():
    a = AnimState(dwell_ms=1600.0)
    a.advance(0)
    a.begin_dwell(0)

    t = a.advance(int(1599 * MS))
    assert not t.dwell_fired and not a.drilled

    t = a.advance(int(1601 * MS))
    assert t.dwell_fired and a.drilled, "drills once past the dwell time"

    t = a.advance(int(3000 * MS))
    assert not t.dwell_fired, "and only ONCE — not every frame afterwards"

    a.cancel_dwell()
    assert a.dwell_t0 is None and not a.drilled
    t = a.advance(int(9000 * MS))
    assert not t.dwell_fired, "a cancelled dwell never fires"
    print("  dwell: fires exactly once, cancellable  OK")


def test_press_envelope():
    a = AnimState()
    a.advance(0)
    a.begin_press(3, (10, 20), 0)
    assert a.pressing and a.held

    p = a.press(int(200 * MS))
    assert p.held and p.scale == 0.97, "held: pressed in"
    assert p.fade == 1.0

    assert a.release_press(int(300 * MS)) is True
    assert a.release_press(int(400 * MS)) is False, "a second release is a no-op"
    assert a.pressing and not a.held, "still alive, now decaying"

    # springs back past 1.0 (ease-out-back overshoot), then settles
    scales = [a.press(int((300 + d) * MS)).scale
              for d in range(0, int(PRESS_SPRING_MS) + 40, 20)]
    assert max(scales) > 1.0, "the spring must overshoot, not just ease to 1"
    assert abs(scales[-1] - 1.0) < 0.02, "and settle at 1.0"

    # glow/tint decay to nothing
    assert a.press(int((300 + PRESS_DECAY_MS + 1) * MS)).fade == 0.0

    t = a.advance(int((300 + PRESS_WAVE_MS + 1) * MS))
    assert t.press_retired and not a.pressing, "retired once fully decayed"
    print("  press: hold 0.97, spring overshoot, decay, retire  OK")


def test_press_waves():
    a = AnimState()
    a.advance(0)

    # a quick TAP must still play one full wave that outlives the click
    a.begin_press(1, (0, 0), 0)
    a.release_press(int(10 * MS))
    assert len(a.waves(int(10 * MS))) == 1, "the tap's wave exists at release"
    assert len(a.waves(int((PRESS_WAVE_MS - 20) * MS))) == 1, \
        "and is STILL playing long after the button came up"
    assert len(a.waves(int((PRESS_WAVE_MS + 20) * MS))) == 0, "then it's gone"

    # a long hold keeps spawning, but only the ~2 on-screen ones are considered
    a.clear_press()
    a.begin_press(1, (0, 0), 0)
    for sec in (1, 5, 30):
        n = len(a.waves(sec * S))
        assert n <= 2, "a 30 s hold must not walk a growing list (got %d)" % n
    assert len(a.waves(int(PRESS_SPAWN_MS * 1.5 * MS))) >= 1
    print("  waves: tap outlives the click; a long hold stays bounded  OK")


def test_morph():
    a = AnimState()
    a.advance(0)
    a.begin_morph(0)
    assert a.morph_progress(0) == 0.0
    mid = a.morph_progress(int(MORPH_MS / 2 * MS))
    assert 0.0 < mid < 1.0 and mid > 0.5, "eased, not linear"
    assert a.morph_progress(int(MORPH_MS * MS)) is None, "done at the deadline"

    t = a.advance(int((MORPH_MS + 1) * MS))
    assert t.morph_retired and a.morph_t0 is None
    print("  morph: eases and retires at %.0f ms  OK" % MORPH_MS)


def test_idle_gate():
    """The tick must stop when nothing is animating — otherwise an idle window
    wakes 60 times a second."""
    a = AnimState()
    assert not a.advance(0).animating

    a.scanning = True
    assert a.advance(1 * S).animating
    a.scanning = False
    assert not a.advance(2 * S).animating

    assert a.advance(3 * S, active=True).animating, "a hover keeps it alive"

    a.begin_press(1, (0, 0), 3 * S)
    assert a.advance(4 * S).animating
    a.release_press(4 * S)
    assert not a.advance(int(4 * S + (PRESS_WAVE_MS + 1) * MS)).animating
    print("  idle gate: stops when idle, re-arms on scan/hover/press  OK")


def main():
    print("engine.anim")
    test_clocks()
    test_reveal()
    test_pop()
    test_dwell()
    test_press_envelope()
    test_press_waves()
    test_morph()
    test_idle_gate()
    print("ANIM OK")


if __name__ == "__main__":
    main()
