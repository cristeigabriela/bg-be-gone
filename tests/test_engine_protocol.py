#!/usr/bin/env python3
"""The compute protocol: the catalogue, the codec, and conformance.

Two different things are checked here.

**The codec** — jobs and events round-trip, required fields are enforced, and an
undeclared job is rejected rather than shipped to the venv to fail there. Plus
`ImagePayload`: a bare string is a path (which is the desktop's current wire
format, unchanged), and bytes/handles encode as dicts so the browser — which has
no filesystem — can use the same protocol.

**Conformance** — every `{"op": ...}` app.py actually sends and every
`{"type": ...}` the worker actually emits must be declared in `engine/protocol.py`.
This is not busywork: when the protocol was written, the worker's own docstring
had already drifted from its code (missing the `unload` job and the `notice` and
`unloaded` events). A docstring cannot fail a build. This can.

Run: python tests/test_engine_protocol.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "src", "bgbg")
sys.path.insert(0, SRC)

from engine import protocol as P                                  # noqa: E402

APP = os.path.join(SRC, "app.py")
SERVICE = os.path.join(SRC, "compute", "service.py")

FAILED = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAILED.append(label)
    print("  %-50s %s" % (label, "PASS" if ok else "FAIL"))
    if not ok:
        print("      want: %r" % (want,))
        print("      got:  %r" % (got,))


def ok(label, cond):
    check(label, bool(cond), True)


def raises(label, fn):
    try:
        fn()
    except P.ProtocolError:
        check(label, True, True)
    else:
        check(label, "no error", "ProtocolError")


# ------------------------------------------------------------------ codec ----
def test_jobs():
    j = P.job("single", input="/in.png", output="/out.png",
              model="birefnet-general", bg="transparent", alpha=False)
    check("a job encodes to the wire format the worker already speaks",
          j, {"op": "single", "input": "/in.png", "output": "/out.png",
              "model": "birefnet-general", "bg": "transparent", "alpha": False})

    check("None fields are dropped, not sent as null",
          "blur" in P.job("single", input="/i", output="/o", model="m", blur=None),
          False)

    raises("an undeclared job is rejected", lambda: P.job("teleport"))
    raises("a job missing a required field is rejected",
           lambda: P.job("single", input="/i"))          # no output, no model
    raises("... and validate_job catches it on the way in too",
           lambda: P.validate_job({"op": "seg_load"}))   # no input

    ok("every job the app can send is declared",
       set(P.JOBS) == {"single", "batch", "gif", "cancel", "models", "unload",
                       "seg_load", "seg_everything", "seg_point", "seg_extract",
                       "seg_cancel", "shutdown"})
    check("12 jobs", len(P.JOBS), 12)


def test_events():
    e = P.event("seg_mask", mask="/m.png", score=0.91, bbox=[1, 2, 3, 4])
    check("an event round-trips", P.validate_event(e).name, "seg_mask")
    raises("an undeclared event is rejected",
           lambda: P.validate_event({"type": "vibes"}))
    raises("an event missing a required field is rejected",
           lambda: P.event("error"))                     # no message
    check("20 events", len(P.EVENTS), 20)

    # strict mode is what the conformance test below leans on
    raises("strict mode rejects an undeclared field",
           lambda: P.validate_event(
               {"type": "seg_canceled", "surprise": 1}, strict=True))
    P.validate_event({"type": "seg_canceled", "surprise": 1})   # lenient by default
    ok("... but the default is lenient, so a newer worker can add fields", True)


def test_image_payload():
    p = P.ImagePayload.path("/tmp/a.png")
    check("a path encodes as a bare string — today's wire format, unchanged",
          p.encode(), "/tmp/a.png")
    check("... and decodes back", P.ImagePayload.decode("/tmp/a.png"), p)

    b = P.ImagePayload.data(b"\x89PNG", "image/png")
    check("bytes encode as a dict (the browser has no paths)",
          b.encode(), {"kind": "bytes", "value": b"\x89PNG", "mime": "image/png"})
    check("... and decode back", P.ImagePayload.decode(b.encode()), b)

    h = P.ImagePayload.handle("tex:7")
    check("a handle round-trips", P.ImagePayload.decode(h.encode()), h)

    # the job codec accepts payloads directly
    j = P.job("seg_load", input=P.ImagePayload.path("/x.png"))
    check("a job carrying a payload still writes a plain path",
          j["input"], "/x.png")
    check("images_of() finds the payloads a job carries",
          P.images_of(j), {"input": P.ImagePayload.path("/x.png")})


# ------------------------------------------------------------ conformance ----
def _literals(path, key):
    with open(path) as f:
        src = f.read()
    # only the sends/emits, not the docstring block
    src = re.sub(r'^""".*?"""', "", src, count=1, flags=re.S)
    return set(re.findall(r'"%s":\s*"([a-z_]+)"' % key, src))


def test_conformance():
    sends = _literals(APP, "op")
    ok("app.py sends at least a dozen distinct jobs", len(sends) >= 10)
    undeclared = sends - set(P.JOBS)
    check("every job app.py sends is declared in the protocol",
          sorted(undeclared), [])

    emits = _literals(SERVICE, "type")
    ok("service.py emits at least a dozen distinct events", len(emits) >= 15)
    undeclared = emits - set(P.EVENTS)
    check("every event service.py emits is declared in the protocol",
          sorted(undeclared), [])

    # and the other direction: nothing declared but dead
    check("every event the protocol declares is actually emitted",
          sorted(set(P.EVENTS) - emits), [])

    # app.py dispatches on `t == "..."`; 'unloaded' is emitted and deliberately
    # ignored (releasing a model needs no UI response).
    with open(APP) as f:
        handled = set(re.findall(r't == "([a-z_]+)"', f.read())) | {"unloaded"}
    check("every event emitted is handled by the app (or knowingly ignored)",
          sorted(emits - handled), [])


def main():
    print("jobs")
    test_jobs()
    print("events")
    test_events()
    print("ImagePayload (no filesystem in the contract)")
    test_image_payload()
    print("conformance: the code vs the catalogue")
    test_conformance()
    print()
    if FAILED:
        print("PROTOCOL FAILED (%d): %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("PROTOCOL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
