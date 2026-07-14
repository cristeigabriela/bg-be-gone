"""The desktop ComputePort: a venv subprocess on a pipe.

The heavy half of the app (numpy, PIL, onnxruntime, the models) cannot live in
the GTK process — the shell runs on the *system* Python and deliberately has none
of it. So compute is a subprocess, and this is the port to it: jobs go out as
line-delimited JSON on stdin, events come back on stdout and are marshalled onto
the GTK main loop.

The browser's port will be a Web Worker and `postMessage`, speaking the same
`engine.protocol`. Nothing above this line knows which one it has.
"""
import json
import subprocess
import threading

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib  # noqa: E402

from engine import protocol as P  # noqa: E402
from engine.ports import ComputePort  # noqa: E402


class WorkerPort(ComputePort):
    def __init__(self, python, script, on_event, log=None):
        self._on_event = on_event
        self._next_id = 1
        self._ok = False
        self.error = None
        self._logf = None
        try:
            self._logf = open(log, "a") if log else subprocess.DEVNULL
            self.proc = subprocess.Popen(
                [python, script], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self._logf, text=True, bufsize=1)
            threading.Thread(target=self._reader, daemon=True).start()
            self._ok = True
        except Exception as e:  # noqa: BLE001
            self.error = str(e)

    @property
    def ok(self):
        return self._ok

    def _reader(self):
        """Runs on a reader thread — never touch GTK from here."""
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            GLib.idle_add(self._on_event, ev)

    def send(self, job):
        if not self._ok:
            return -1
        P.validate_job(job)          # catch a malformed job here, not in the venv
        rid = self._next_id
        self._next_id += 1
        job["id"] = rid
        try:
            self.proc.stdin.write(json.dumps(job) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass                     # the worker died; `error`/timeouts surface it
        return rid

    def shutdown(self):
        if not self._ok:
            return
        try:
            self.proc.stdin.write(json.dumps(P.job("shutdown")) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            try:
                self.proc.terminate()
            except Exception:  # noqa: BLE001
                pass
