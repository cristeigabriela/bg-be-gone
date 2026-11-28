"""The worker, over its real JSON pipe, on the CPU.

Uses rembg's smallest model (u2netp, ~5 MB), downloaded on first use into
U2NET_HOME; CI caches it.
"""
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

import modelinfo
from workerpipe import BACKGROUND_PIXEL, REPLY_TIMEOUT_S, SUBJECT_PIXEL, TESTDATA, WORKER


@pytest.fixture
def engine_env(tmp_path):
    """An environment that names no engine anywhere, whatever this machine has set.

    An empty config folder (XDG's, and %LOCALAPPDATA%'s on Windows), no
    BGBG_ENGINE, a profile that isn't there, and a gpu.env that sets nothing.
    """
    env = {k: v for k, v in os.environ.items() if k != "BGBG_ENGINE"}
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    env["LOCALAPPDATA"] = str(tmp_path / "local")
    env["BGBG_PROFILE"] = str(tmp_path / "no-profile.json")
    env["BGBG_GPU_ENV"] = str(TESTDATA / "gpu-env" / "empty.env")
    return env


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
                                 "seg_models", "engines", "engine", "settings"}, \
        "the frontend reads each of these on start-up"
    assert worker.ready["engines"][0] == "ort", "ONNX Runtime is always on offer"
    assert worker.ready["providers"][-1] == "CPUExecutionProvider", \
        "the CPU is always the last provider, so the status bar always has one to show"
    assert "u2netp" in worker.ready["models"], "rembg's models are listed"


def test_every_model_the_worker_lists_is_described_or_left_out_on_purpose(worker):
    unaccounted = (set(worker.ready["models"]) - set(modelinfo.BG_MODELS)
                   - set(modelinfo.NOT_OFFERED))
    # The worker runs on this interpreter, so this is the rembg it lists.
    rembg = importlib.metadata.version("rembg")
    assert not unaccounted, \
        "rembg %s lists models with no modelinfo entry or NOT_OFFERED reason: %s" % (
            rembg, sorted(unaccounted))


def test_every_model_the_picker_offers_is_one_the_worker_runs(worker):
    unknown = set(modelinfo.BG_MODELS) - set(worker.ready["models"])
    assert not unknown, "a picker entry the worker can't load: %s" % sorted(unknown)


def test_every_segment_model_the_worker_lists_is_described(worker):
    rungs = {m["rung"] for m in worker.ready["seg_models"]}
    assert rungs, "the CPU worker has segmentation"
    assert rungs == set(modelinfo.SEG_MODELS), \
        "Segment's picker and the worker must offer the same models"


def test_with_no_engine_set_start_up_says_nothing_but_ready(start_worker, engine_env):
    """Every install today has no engine setting, and must start as it always did."""
    w = start_worker(engine_env)
    # A second "ready" marks the end of whatever start-up sent after the first.
    _, seen = w.ask({"op": "models"}, until={"ready"})
    assert [r["type"] for r in seen] == ["ready"], "no notice or other message at start-up"


def test_asking_for_pytorch_keeps_onnx_runtime_and_says_so(start_worker, engine_env, tmp_path):
    """Without PyTorch installed, a torch request must not break removal.

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


@FORMAT_FIXTURES
def test_convert_decodes_every_readable_format_into_a_png_to_preview(
        worker, tmp_path, fixture):
    """The frontend's fallback when GdkPixbuf has no loader for the format."""
    out = tmp_path / "preview.png"
    reply, _ = worker.ask(
        {"op": "convert", "id": 1, "input": str(TESTDATA / fixture), "output": str(out)},
        until={"converted", "error"})
    assert reply["type"] == "converted", "the worker reads it: %s" % reply.get("message")
    assert reply["frames"] == 1, "a still image has one frame"
    preview = Image.open(out)
    assert preview.format == "PNG", "the preview is a PNG, whatever the source"
    assert preview.mode in ("RGB", "RGBA"), "GdkPixbuf reads 8-bit RGB(A)"
    assert preview.size == (256, 256), "the preview is the source's full size"
    grey = preview.convert("L")
    assert grey.getpixel(SUBJECT_PIXEL) < 100, "the dark subject stays dark"
    assert grey.getpixel(BACKGROUND_PIXEL) < 250, "the light grey background isn't clipped to white"


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


def test_batch_skips_an_animated_webp_and_says_where_it_goes(worker, tmp_path):
    """Batch would keep only its first frame, as with a GIF."""
    inp, outdir = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    shutil.copyfile(TESTDATA / "subject.webp", inp / "still.webp")
    shutil.copyfile(TESTDATA / "animated.webp", inp / "moving.webp")
    reply, seen = worker.ask(
        {"op": "batch", "model": "u2netp", "id": 1,
         "input_dir": str(inp), "output_dir": str(outdir)},
        until={"done_batch", "error"})
    assert reply["type"] == "done_batch", "the still WebP is still processed: %s" \
        % reply.get("message")
    assert [p.name for p in outdir.iterdir()] == ["still_nobg.png"], \
        "the animation is skipped, not cut down to one frame"
    notices = [r["message"] for r in seen if r.get("type") == "notice"]
    assert any("moving.webp" in n and "Single" in n for n in notices), \
        "the user is told which file was skipped and where to open it: %s" % notices


@pytest.mark.parametrize("fixture, frames", [
    ("animated.webp", 3),
    # The odd one out: a still WebP is one frame.
    ("subject.webp", 1),
], ids=["animated", "still"])
def test_frames_counts_a_webp_from_the_file_itself(worker, fixture, frames):
    """The frontend asks this, not GdkPixbuf, whose WebP loader may decode stills only."""
    reply, _ = worker.ask({"op": "frames", "id": 1, "input": str(TESTDATA / fixture)},
                          until={"frames", "error"})
    assert reply.get("frames") == frames, reply


@pytest.fixture
def cutout(worker, tmp_path):
    """A transparent cut-out of subject.png, as the frontend would save it."""
    out = tmp_path / "cutout.png"
    reply, _ = worker.ask(
        {"op": "single", "model": "u2netp", "id": 1,
         "input": str(TESTDATA / "subject.png"), "output": str(out)},
        until={"done_single", "error"})
    assert reply["type"] == "done_single", reply.get("message")
    return out


def test_a_cutout_saved_as_webp_keeps_its_transparency(worker, tmp_path, cutout):
    saved = tmp_path / "saved.webp"
    reply, _ = worker.ask({"op": "convert", "id": 2, "input": str(cutout), "output": str(saved)},
                          until={"converted", "error"})
    assert reply["type"] == "converted", reply.get("message")
    assert Image.open(saved).format == "WEBP", "the suffix picks the format"
    assert_cut_out(saved)


def test_a_result_with_an_alpha_channel_still_saves_as_jpeg(worker, tmp_path, cutout):
    """JPEG has no alpha channel; the worker drops it rather than fail."""
    saved = tmp_path / "saved.jpg"
    reply, _ = worker.ask({"op": "convert", "id": 2, "input": str(cutout), "output": str(saved)},
                          until={"converted", "error"})
    assert reply["type"] == "converted", reply.get("message")
    assert Image.open(saved).format == "JPEG", "the suffix picks the format"


def test_an_unreadable_image_is_named_in_the_error_and_the_log(worker, tmp_path):
    bad = TESTDATA / "not_an_image.webp"
    reply, _ = worker.ask(
        {"op": "single", "model": "u2netp", "id": 1,
         "input": str(bad), "output": str(tmp_path / "x.png")},
        until={"done_single", "error"})
    assert reply["type"] == "error", "a text file can't be decoded"
    assert reply["message"] == "not_an_image.webp isn't an image bg-be-gone can read.", \
        "the error, which the frontend shows, names the file in plain words"
    assert str(bad) in (tmp_path / "worker.log").read_text(), \
        "the worker log has the file's full path"


def test_an_animated_webp_comes_back_animated_with_its_timing(worker, tmp_path):
    """Animated WebP goes through the GIF pipeline but stays a WebP, with full alpha."""
    out = tmp_path / "result.webp"
    reply, _ = worker.ask(
        {"op": "gif", "model": "u2netp", "id": 1,
         "input": str(TESTDATA / "animated.webp"), "output": str(out)},
        until={"gif_done", "error"})
    assert reply["type"] == "gif_done", "the animation must succeed: %s" % reply.get("message")
    result = Image.open(out)
    assert result.format == "WEBP", "the result keeps the source's format"
    assert result.n_frames == 3, "every frame is kept"
    durations = []
    for i in range(result.n_frames):
        result.seek(i)
        alpha = result.convert("RGBA").getchannel("A")
        durations.append(result.info["duration"])
        assert alpha.getpixel(SUBJECT_PIXEL) > 224, "every frame keeps the subject"
        assert alpha.getpixel(BACKGROUND_PIXEL) < 32, "every frame drops the background"
    assert durations == [80, 120, 160], "each frame keeps its own duration"
    assert Image.open(reply["preview"]).format == "PNG", \
        "a PNG first frame stands in where GdkPixbuf has no WebP loader"
