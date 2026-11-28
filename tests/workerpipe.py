"""The worker process, spoken to over its real JSON pipe the way the frontend does.

Shared by every test that drives the worker. Runs it on the interpreter running
pytest, which needs the worker's CPU stack (rembg, onnxruntime, Pillow).
"""
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKER = REPO / "src" / "bgbg" / "worker.py"
TESTDATA = Path(__file__).parent / "testdata"
# subject.png: a dark disc centred at (128, 128), radius >= 64, on light grey.
SUBJECT_PIXEL = (128, 128)
BACKGROUND_PIXEL = (8, 8)
# Generous: the first request may download the model.
REPLY_TIMEOUT_S = 180


class Worker:
    """The worker process, with its stderr (the app's log) in `log_path`."""

    def __init__(self, log_path, env=None, command=None):
        self.log_path = log_path
        self._log = open(log_path, "w")
        self.proc = subprocess.Popen(
            command or [sys.executable, str(WORKER)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self._log, text=True, bufsize=1, env=env)
        # A reader thread, so a worker that stalls fails the test at the
        # deadline instead of blocking readline() forever.
        self._replies = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.ready = self.next_reply()
        self._next_id = 1

    def _read(self):
        for line in self.proc.stdout:
            if line.startswith("{"):
                self._replies.put(json.loads(line))
        self._replies.put(None)

    def next_reply(self):
        try:
            reply = self._replies.get(timeout=REPLY_TIMEOUT_S)
        except queue.Empty:
            raise AssertionError(f"no reply from the worker in {REPLY_TIMEOUT_S} s") from None
        assert reply is not None, "the worker exited (see %s)" % self.log_path
        return reply

    def reply_within(self, seconds):
        """The next reply if one comes within `seconds`, else None."""
        try:
            reply = self._replies.get(timeout=seconds)
        except queue.Empty:
            return None
        assert reply is not None, "the worker exited (see %s)" % self.log_path
        return reply

    def send(self, request):
        """Send a request as is, giving it the next id if it has none; return its id."""
        request.setdefault("id", self._next_id)
        self._next_id += 1
        self.write(request)
        return request["id"]

    def write(self, message):
        """Write one line to the worker, exactly as given (a cancel names another id)."""
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def ask(self, request, until):
        """Send a request; return (the reply whose type is in `until`, all replies)."""
        self.send(request)
        seen = []
        while True:
            reply = self.next_reply()
            seen.append(reply)
            if reply.get("type") in until:
                return reply, seen

    def log_text(self):
        self._log.flush()
        return Path(self.log_path).read_text(encoding="utf-8", errors="replace")

    def close(self):
        try:
            self.proc.stdin.write('{"op": "shutdown"}\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
            self.proc.wait()
        self._log.close()
