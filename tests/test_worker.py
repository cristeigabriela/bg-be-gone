"""The worker, over its real JSON pipe, on the CPU.

Uses rembg's smallest model (u2netp, ~5 MB), downloaded on first use into
U2NET_HOME; CI caches it.
"""
import json
import os
import queue
import shutil
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

    def __init__(self, log_path, env=None, command=None):
        self._log = open(log_path, "w")
        self.proc = subprocess.Popen(
            command or [sys.executable, str(WORKER)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self._log, text=True, bufsize=1, env=env)
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


@pytest.fixture
def engine_env(tmp_path):
    """An environment that names no engine anywhere, whatever this machine has set.

    An empty config folder, no BGBG_ENGINE, a profile that isn't there, and a
    gpu.env that sets nothing.
    """
    env = {k: v for k, v in os.environ.items() if k != "BGBG_ENGINE"}
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    env["BGBG_PROFILE"] = str(tmp_path / "no-profile.json")
    env["BGBG_GPU_ENV"] = str(TESTDATA / "gpu-env" / "empty.env")
    return env


@pytest.fixture
def start_worker(tmp_path):
    """Start workers with a given environment; every one is shut down after the test."""
    started = []

    def start(env, command=None):
        w = Worker(tmp_path / f"worker-{len(started)}.log", env=env, command=command)
        started.append(w)
        return w

    yield start
    for w in started:
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


def test_the_ready_message_carries_every_field_the_frontend_reads(worker):
    assert set(worker.ready) == {"type", "models", "providers", "bgremove", "seg",
                                 "seg_models"}, "the frontend reads each of these on start-up"
    assert worker.ready["providers"][-1] == "CPUExecutionProvider", \
        "the CPU is always the last provider, so the status bar always has one to show"
    assert "u2netp" in worker.ready["models"], "rembg's models are listed"


def test_with_no_engine_set_start_up_says_nothing_but_ready(start_worker, engine_env):
    """Every install today has no engine setting, and must start as it always did."""
    w = start_worker(engine_env)
    # A second "ready" marks the end of whatever start-up sent after the first.
    _, seen = w.ask({"op": "models"}, until={"ready"})
    assert [r["type"] for r in seen] == ["ready"], "no notice or other message at start-up"


def test_asking_for_pytorch_keeps_onnx_runtime_and_says_so(start_worker, engine_env, tmp_path):
    """Only ONNX Runtime exists so far: a torch request must not break removal.

    The worker sees only the CPU, so the provider that runs is known.
    """
    cpu_only = [sys.executable, str(TESTDATA / "cpu_only_worker.py"), str(WORKER.parent)]
    w = start_worker(dict(engine_env, BGBG_ENGINE="torch"), command=cpu_only)
    reply, seen = w.ask(
        {"op": "single", "model": "u2netp", "id": 3,
         "input": str(TESTDATA / "subject.png"), "output": str(tmp_path / "cutout.png")},
        until={"done_single", "error"})
    notices = [r["message"] for r in seen if r.get("type") == "notice"]
    assert any("PyTorch" in m for m in notices), "the user learns PyTorch isn't used"
    assert reply["type"] == "done_single", "the cutout still succeeds: %s" % reply.get("message")
    devices = [r["provider"] for r in seen if r.get("type") == "device"]
    assert devices == ["CPUExecutionProvider"], "the run goes through onnxruntime, on the CPU"


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
    env = dict(os.environ, OMP_NUM_THREADS="2")
    done = subprocess.run(
        [sys.executable, str(TESTDATA / "session_options_probe.py"), str(WORKER.parent)],
        capture_output=True, text=True, env=env, timeout=REPLY_TIMEOUT_S)
    assert done.returncode == 0, done.stderr
    options = json.loads(done.stdout.strip().splitlines()[-1])
    assert options["optimisation_off"], "graph optimisation must be off"
    assert options["intra_op_threads"] == 2, \
        "OMP_NUM_THREADS must still cap the threads, as rembg.new_session did"


def test_subject_fixture_matches_the_pixels_the_tests_check():
    """Guards the coordinates above against a regenerated fixture."""
    image = np.asarray(Image.open(TESTDATA / "subject.png").convert("L"))
    assert image[SUBJECT_PIXEL[1], SUBJECT_PIXEL[0]] < 100, "the subject pixel is on the disc"
    assert image[BACKGROUND_PIXEL[1], BACKGROUND_PIXEL[0]] > 200, "the background pixel is not"


# subject.png saved in each format the worker reads (make_format_fixtures.py).
FORMAT_FIXTURES = pytest.mark.parametrize("fixture", [
    "subject.png",
    "subject.jpg",
    "subject.webp",
    # AVIF: Pillow's own plugin, no extra package.
    "subject.avif",
    # HEIC: needs pi-heif; Pillow alone can't read it.
    "subject.heic",
    "subject.tif",
    # 16-bit greyscale: Pillow's own conversion clips it to white.
    "subject16.tif",
    # A two-colour palette BMP, not RGB: its preview must still come out RGB.
    "subject.bmp",
    "subject.gif",
], ids=["png", "jpeg", "webp", "avif", "heic", "tiff", "tiff-16-bit-grey", "bmp", "gif"])


def assert_cut_out(path):
    alpha = Image.open(path).convert("RGBA").getchannel("A")
    assert alpha.getpixel(SUBJECT_PIXEL) > 224, "the subject stays opaque"
    assert alpha.getpixel(BACKGROUND_PIXEL) < 32, "the background becomes transparent"


@FORMAT_FIXTURES
def test_a_cutout_succeeds_for_every_format_the_worker_reads(worker, tmp_path, fixture):
    out = tmp_path / "cutout.png"
    reply, _ = worker.ask(
        {"op": "single", "model": "u2netp", "id": 1,
         "input": str(TESTDATA / fixture), "output": str(out)},
        until={"done_single", "error"})
    assert reply["type"] == "done_single", "the cutout must succeed: %s" % reply.get("message")
    assert_cut_out(out)


def test_batch_cuts_out_every_still_format_and_skips_gifs_and_other_files(worker, tmp_path):
    """A GIF in batch would lose all but its first frame, so batch leaves it out."""
    inp, outdir = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    stills = ["subject.png", "subject.jpg", "subject.webp", "subject.avif",
              "subject.heic", "subject.tif", "subject.bmp"]
    for name in stills + ["subject.gif"]:
        # Named after the format, so no two outputs collide; upper case, as
        # cameras write them, so the scan must ignore case.
        suffix = name.rsplit(".", 1)[1]
        shutil.copyfile(TESTDATA / name, inp / f"{suffix}.{suffix.upper()}")
    (inp / "notes.txt").write_text("not an image\n")
    reply, _ = worker.ask(
        {"op": "batch", "model": "u2netp", "id": 1,
         "input_dir": str(inp), "output_dir": str(outdir)},
        until={"done_batch", "error"})
    assert reply["type"] == "done_batch", "the batch must succeed: %s" % reply.get("message")
    want = {name.rsplit(".", 1)[1] + "_nobg.png" for name in stills}
    assert {p.name for p in outdir.iterdir()} == want, \
        "every still format is cut out; the GIF and the text file are not"
    for out in outdir.iterdir():
        assert_cut_out(out)


def test_an_unreadable_image_is_named_in_the_error_and_the_log(worker, tmp_path):
    bad = TESTDATA / "not_an_image.webp"
    reply, _ = worker.ask(
        {"op": "single", "model": "u2netp", "id": 1,
         "input": str(bad), "output": str(tmp_path / "x.png")},
        until={"done_single", "error"})
    assert reply["type"] == "error", "a text file can't be decoded"
    assert reply["message"].startswith("can't read not_an_image.webp: "), \
        "the error's first line, which the frontend shows, names the file"
    assert str(bad) in (tmp_path / "worker.log").read_text(), \
        "the worker log has the file's full path"
