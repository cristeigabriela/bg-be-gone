"""The ports: how the engine reaches the world it is not allowed to know about.

Stdlib only.

A `ComputePort` is anything that takes a job (`engine.protocol`) and eventually
delivers events back. On the desktop that is a venv subprocess on a pipe
(`shell_gtk.worker_port`); in the browser it will be a Web Worker running ORT-Web
behind `postMessage`. The engine cannot tell the difference, which is the whole
point — it is the same seam as `render.ops` (what to draw) and `effects` (what to
do), applied to compute.

Nothing here imports numpy, onnxruntime or gi. A port *implementation* may.
"""
from . import protocol as P


class ComputePort:
    """Send jobs, receive events. Implementations live in the shells."""

    def send(self, job):
        """Dispatch a job dict (see `protocol.job`). Returns a request id."""
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError

    @property
    def ok(self):
        """False if the port could not be started at all."""
        raise NotImplementedError


class NullPort(ComputePort):
    """A port with nothing behind it — a segmentation-only or broken install.

    The app is expected to stay usable (you can still open, zoom and inspect an
    image), so "no compute" is a state, not a crash.
    """

    def __init__(self, error="no compute backend"):
        self.error = error
        self._next = 1

    def send(self, job):
        self._next += 1
        return -1

    def shutdown(self):
        pass

    @property
    def ok(self):
        return False


class RecordingPort(ComputePort):
    """A port that records jobs and replays canned events. For tests.

    This is what lets a whole app-level flow be driven with no model, no
    subprocess and no GPU — see tests/test_live_app_segment.py.
    """

    def __init__(self, on_event=None):
        self.jobs = []
        self.on_event = on_event
        self.error = None
        self._next = 1

    def send(self, job):
        P.validate_job(job)          # a test port that accepts junk proves nothing
        rid = self._next
        self._next += 1
        job = dict(job, id=rid)
        self.jobs.append(job)
        return rid

    def emit(self, ev):
        """Deliver an event as if the worker had sent it."""
        P.validate_event(ev)
        if self.on_event:
            self.on_event(ev)

    def sent(self, op):
        return [j for j in self.jobs if j.get("op") == op]

    def shutdown(self):
        pass

    @property
    def ok(self):
        return True
