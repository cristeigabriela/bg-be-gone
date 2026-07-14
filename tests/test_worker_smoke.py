#!/usr/bin/env python3
"""The real worker subprocess, over its real protocol. No GTK.

This harness cannot run a GTK app and a worker subprocess together, so the worker
is tested the other way round: spawn it exactly as the app does and talk to it on
the pipe.

It exists to catch one specific class of failure that is otherwise **silent**.
`service.py` imports `segmentation` inside a try/except that degrades to "this
worker has no segmentation" — so when service.py moved into `compute/` in step 8,
a broken sys.path would not have raised anything. It would have shipped a worker
that quietly cannot segment, and every test that stubs the worker would still
have passed.

So: start it, read `ready`, and assert it actually found both halves.

Needs the venv (it does NOT need a model or the network).
Run: python tests/test_worker_smoke.py
"""
import os
import sys
import json
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "src", "bgbg")
sys.path.insert(0, SRC)

from engine import protocol as P  # noqa: E402

VENV_PY = (os.environ.get("BGBG_VENV_PYTHON")
           or os.path.expanduser("~/.local/share/bg-be-gone/venv/bin/python"))
SERVICE = os.environ.get("BGBG_WORKER", os.path.join(SRC, "compute", "service.py"))

FAILED = []


def check(label, got, want=True):
    okk = got == want
    if not okk:
        FAILED.append(label)
    print("  %-52s %s" % (label, "PASS" if okk else "FAIL (%r)" % (got,)))


def main():
    if not os.path.exists(VENV_PY):
        print("SKIP: no venv at %s" % VENV_PY)
        return 0

    proc = subprocess.Popen([VENV_PY, SERVICE],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    ready = None
    try:
        # The worker announces itself before it is asked anything.
        for _ in range(50):
            line = proc.stdout.readline()
            if not line:
                break
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") == "ready":
                ready = ev
                break

        if ready is None:
            err = proc.stderr.read()[-800:] if proc.stderr else ""
            print("  worker never sent 'ready'")
            print(err)
            FAILED.append("ready")
        else:
            check("the worker starts and announces itself", True)
            P.validate_event(ready)
            check("... and 'ready' conforms to the declared protocol", True)
            # The whole point of this file:
            check("... it found SEGMENTATION (the silent-failure import)",
                  bool(ready.get("seg")))
            check("... it found BACKGROUND REMOVAL (rembg)",
                  bool(ready.get("bgremove")))
            check("... and it reports at least one execution provider",
                  bool(ready.get("providers")))
            check("... and the SAM model ladder",
                  bool(ready.get("seg_models")))
            print("  providers: %s" % (ready.get("providers"),))
            print("  seg rungs: %s" % ([m.get("rung")
                                        for m in ready.get("seg_models") or []],))

        # and it shuts down cleanly when asked, over the same protocol
        proc.stdin.write(json.dumps(P.job("shutdown")) + "\n")
        proc.stdin.flush()
        try:
            proc.wait(timeout=15)
            check("... and it exits on the 'shutdown' job", True)
        except subprocess.TimeoutExpired:
            check("... and it exits on the 'shutdown' job", False)
    finally:
        if proc.poll() is None:
            proc.kill()

    print()
    if FAILED:
        print("WORKER SMOKE FAILED (%d): %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("WORKER SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
