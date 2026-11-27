"""The worker, over its real JSON pipe, on the CPU.

Uses rembg's smallest model (u2netp, ~5 MB), downloaded on first use into
U2NET_HOME; CI caches it.
"""
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
WORKER = REPO / "src" / "bgbg" / "worker.py"
TESTDATA = Path(__file__).parent / "testdata"

# subject.png: a dark disc centred at (128, 128), radius >= 64, on light grey.
SUBJECT_PIXEL = (128, 128)
BACKGROUND_PIXEL = (8, 8)
# Generous: the first request may download the model.
REPLY_TIMEOUT_S = 180


class Worker:
    """The worker process, spoken to the way the frontend does."""

    def __init__(self, log_path):
        self._log = open(log_path, "w")
        self.proc = subprocess.Popen(
            [sys.executable, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._log, text=True, bufsize=1)
        # A reader thread, so a worker that stalls fails the test at the
        # deadline instead of blocking readline() forever.
        self._replies = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.ready = self.next_reply()

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
        assert reply is not None, "the worker exited (see its log)"
        return reply

    def ask(self, request, until):
        """Send a request; return (the reply whose type is in `until`, all replies)."""
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        seen = []
        while True:
            reply = self.next_reply()
            seen.append(reply)
            if reply.get("type") in until:
                return reply, seen

    def close(self):
        try:
            self.proc.stdin.write('{"op": "shutdown"}\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
            self.proc.wait()
        self._log.close()


@pytest.fixture
def worker(tmp_path):
    w = Worker(tmp_path / "worker.log")
    yield w
    w.close()


def test_a_single_cutout_succeeds(worker, tmp_path):
    """rembg 2.0.61+ rejected the worker's session options with a TypeError,
    so every background removal failed before a model loaded."""
    out = tmp_path / "cutout.png"
    reply, seen = worker.ask(
        {"op": "single", "model": "u2netp", "id": 1,
         "input": str(TESTDATA / "subject.png"), "output": str(out)},
        until={"done_single", "error"})
    assert reply["type"] == "done_single", "the cutout must succeed: %s" % reply.get("message")
    devices = [r["provider"] for r in seen if r.get("type") == "device"]
    assert devices, "the worker reports the device it ran on"
    alpha = Image.open(out).getchannel("A")
    assert alpha.getpixel(SUBJECT_PIXEL) > 224, "the subject stays opaque"
    assert alpha.getpixel(BACKGROUND_PIXEL) < 32, "the background becomes transparent"


def test_an_unknown_model_is_named_in_the_error(worker, tmp_path):
    reply, _ = worker.ask(
        {"op": "single", "model": "u2net_typo", "id": 2,
         "input": str(TESTDATA / "subject.png"), "output": str(tmp_path / "x.png")},
        until={"done_single", "error"})
    assert reply["type"] == "error", "an unknown model can't produce a cutout"
    assert "u2net_typo" in reply["message"], "the error names the model it couldn't find"


def test_rembg_sessions_turn_graph_optimisation_off():
    """The worker's session options must reach onnxruntime: its graph
    optimiser miscompiles a fused op in these models into NaN.

    Runs in its own interpreter: importing the worker can re-exec Python to
    put CUDA libraries on the loader path, which must not hit pytest itself.
    """
    probe = (
        "import sys, onnxruntime as ort; sys.path.insert(0, sys.argv[1]); import worker; "
        "o = worker._new_session('u2netp', ['CPUExecutionProvider'])"
        ".inner_session.get_session_options(); "
        "print(o.graph_optimization_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "
        "o.intra_op_num_threads)")
    env = dict(os.environ, OMP_NUM_THREADS="2")
    done = subprocess.run([sys.executable, "-c", probe, str(WORKER.parent)],
                          capture_output=True, text=True, env=env, timeout=REPLY_TIMEOUT_S)
    assert done.returncode == 0, done.stderr
    optimisation_off, threads = done.stdout.split()
    assert optimisation_off == "True", "graph optimisation must be off"
    assert threads == "2", "OMP_NUM_THREADS must still cap the threads, as rembg.new_session did"


def test_subject_fixture_matches_the_pixels_the_tests_check():
    """Guards the coordinates above against a regenerated fixture."""
    image = np.asarray(Image.open(TESTDATA / "subject.png").convert("L"))
    assert image[SUBJECT_PIXEL[1], SUBJECT_PIXEL[0]] < 100, "the subject pixel is on the disc"
    assert image[BACKGROUND_PIXEL[1], BACKGROUND_PIXEL[0]] > 200, "the background pixel is not"
