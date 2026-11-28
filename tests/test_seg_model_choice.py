"""Auto's SAM model: picked from the VRAM free right now, every time.

A failure on the device steps down for that load only; nothing is
remembered, and a download failure is an error, never a step down.
Imports segmentation in-process: it needs only onnxruntime, numpy and Pillow.
"""
import json
import shutil
import subprocess
import sys
import urllib.error

import pytest

import segmentation as seg
from workerpipe import REPLY_TIMEOUT_S, TESTDATA, WORKER

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class LoadedModel:
    """What SamSession.load returns, for a model that loads and encodes."""

    family = "sam2"

    def __init__(self, rung):
        self.rung = rung

    def encode(self, image):
        return self


@pytest.fixture
def sam(tmp_path, monkeypatch):
    """segmentation with its own folders and a 12 GB GPU with 10 GB free.

    The model loader is the boundary replaced here: `sam.fails` maps a rung to
    the exception its load raises; any other rung loads.
    """
    monkeypatch.setattr(seg, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(seg, "CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(seg, "CONFIG_PATH", str(tmp_path / "config" / "seg.json"))
    monkeypatch.setattr(seg, "probe_vram",
                        lambda: {"total": 12288, "free": 10000, "gpu": True})

    class Sam:
        fails = {}
        tried = []
        config = tmp_path / "config" / "seg.json"

    def load(rung, providers, progress=None):
        Sam.tried.append(rung)
        if rung in Sam.fails:
            raise Sam.fails[rung]
        return LoadedModel(rung)
    monkeypatch.setattr(seg.SamSession, "load", staticmethod(load))
    return Sam


def test_auto_ignores_a_remembered_rung_and_drops_it(sam):
    """A seg.json holding "small" on a 12 GB card made Auto use it on every launch."""
    sam.config.parent.mkdir()
    shutil.copyfile(TESTDATA / "seg-legacy-small.json", sam.config)
    pick = seg.auto_select_session(PROVIDERS, gpu=True)
    assert pick.rung == "large", "Auto picks from the VRAM free now"
    assert pick.mode == "auto", "the first pick worked"
    assert "rung" not in json.loads(sam.config.read_text()), "the stale rung is dropped"


def test_running_out_of_vram_steps_down_for_this_load_only(sam):
    sam.fails = {"large": RuntimeError("Failed to allocate memory: out of memory")}
    pick = seg.auto_select_session(PROVIDERS, gpu=True)
    assert (pick.rung, pick.mode) == ("base_plus", "fallback"), \
        "the next smaller model runs this time"
    assert not sam.config.exists(), "nothing is remembered for the next load"


def test_a_failed_download_is_an_error_not_a_step_down(sam):
    """Offline, a smaller model that happened to be cached ran instead, with no
    word about why."""
    sam.fails = {"large": urllib.error.URLError("no route to host")}
    with pytest.raises(seg.SegDownloadError) as raised:
        seg.auto_select_session(PROVIDERS, gpu=True)
    assert isinstance(raised.value.__cause__, urllib.error.URLError), "the cause is kept"
    assert sam.tried == ["large"], "no smaller model is tried for a download failure"


def test_a_damaged_cached_archive_is_removed_and_reported(tmp_path, monkeypatch):
    """A truncated zip stayed in the models folder, and every later load failed
    on it the same way."""
    models = tmp_path / "models"
    monkeypatch.setattr(seg, "MODELS_DIR", str(models))
    models.mkdir()
    archive = models / seg.MODELS["base_plus"]["zip"]
    shutil.copyfile(TESTDATA / "not_an_image.webp", archive)
    with pytest.raises(seg.SegModelArchiveError):
        seg.ensure_model("base_plus")
    assert not archive.exists(), "the damaged archive is removed, so the next try downloads"


def test_a_damaged_model_is_not_a_reason_to_step_down(sam):
    """Getting the model failed, not running it: a smaller one must not stand in."""
    sam.fails = {"large": seg.SegModelArchiveError("/models/x.zip", "not a zip file")}
    with pytest.raises(seg.SegModelArchiveError):
        seg.auto_select_session(PROVIDERS, gpu=True)
    assert sam.tried == ["large"], "no smaller model is tried"


@pytest.mark.parametrize("free_mib, rung", [
    # Large's measured peak (3041 MiB) plus its headroom: about 4.3 GB.
    (4313, "large"),
    # One MiB short of Large's gate: the next model down.
    (4312, "base_plus"),
    # Below Tiny's gate (3446 MiB): MobileSAM, which Auto falls back to.
    (3445, "mobile"),
], ids=["large-fits", "just-short-of-large", "below-tiny"])
def test_auto_picks_the_best_model_whose_measured_need_fits(free_mib, rung):
    """The old gates (8 GB for Large) kept Auto off models that fit in half that."""
    assert seg.rung_for_vram(free_mib, True) == rung, "the gate is the measured peak plus headroom"


def test_the_pick_says_how_much_vram_it_was_based_on(sam):
    """The picker shows it, so the user can see why Auto chose a model."""
    assert seg.auto_select_session(PROVIDERS, gpu=True).vram_free == 10000, \
        "Auto reports the free VRAM it probed"
    assert seg.auto_select_session(PROVIDERS, gpu=True, want="tiny").vram_free is None, \
        "a model the user chose probes nothing"


def test_a_good_archive_is_unpacked_and_removed(tmp_path, monkeypatch):
    """sam-archive-ok.zip holds a stand-in encoder and decoder for Base+."""
    models = tmp_path / "models"
    monkeypatch.setattr(seg, "MODELS_DIR", str(models))
    models.mkdir()
    shutil.copyfile(TESTDATA / "sam-archive-ok.zip", models / seg.MODELS["base_plus"]["zip"])
    encoder, decoder = seg.ensure_model("base_plus")
    with open(encoder, "rb") as f:
        assert f.read() == b"encoder stand-in", "the encoder is unpacked whole"
    with open(decoder, "rb") as f:
        assert f.read() == b"decoder stand-in", "the decoder is unpacked whole"
    assert seg.is_cached("base_plus"), "the next load finds it"
    assert sorted(p.name for p in models.iterdir()) == ["sam2.1_hiera_base_plus"], \
        "the archive is removed, and no part file is left"


def test_a_new_image_lets_go_of_the_old_model_before_the_pick():
    """The old session held its VRAM while free VRAM was probed for the next
    image, so the pick could come out smaller, or need room for two models.

    Runs the worker in its own interpreter: importing it can re-exec Python.
    """
    done = subprocess.run(
        [sys.executable, str(TESTDATA / "seg_reload_probe.py"), str(WORKER.parent),
         str(TESTDATA / "subject.png")],
        capture_output=True, text=True, timeout=REPLY_TIMEOUT_S)
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    assert seen["old_session_held"] is False, "the old model is released before the pick"
    assert seen["replies"] == ["seg_ready"], "the new model still loads"
