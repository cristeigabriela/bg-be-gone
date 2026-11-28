"""The PyTorch engine's checks that need no PyTorch: checkpoint integrity, and
how the worker falls back and reports when PyTorch can't run.

The worker tests put a stand-in torch package on PYTHONPATH
(testdata/fake_torch), which looks installed but has no GPU.
"""
import hashlib
import os
import sys

import pytest

from engines import torch_birefnet
from workerpipe import TESTDATA, WORKER, Worker

FAKE_TORCH = TESTDATA / "fake_torch"
GOOD = b"checkpoint bytes"


@pytest.fixture
def release(tmp_path, monkeypatch):
    """A one-model "release" served from a folder, with its checkpoint's digest."""
    served = tmp_path / "release"
    served.mkdir()
    (served / "model.pth").write_bytes(GOOD)
    monkeypatch.setattr(torch_birefnet, "_RELEASE", served.as_uri() + "/")
    monkeypatch.setattr(torch_birefnet, "MODELS", {
        "birefnet-test": ("model.pth", "swin_v1_t", hashlib.sha256(GOOD).hexdigest())})
    return served


def test_a_corrupt_cached_checkpoint_is_downloaded_again(release, tmp_path):
    cache = tmp_path / "models"
    cache.mkdir()
    (cache / "model.pth").write_bytes(b"cut short")
    path = torch_birefnet.ensure_checkpoint("birefnet-test", str(cache))
    with open(path, "rb") as f:
        assert f.read() == GOOD, "a cached file that fails its SHA-256 must be replaced"


def test_a_download_with_the_wrong_digest_is_refused(release, tmp_path):
    (release / "model.pth").write_bytes(b"tampered")
    with pytest.raises(torch_birefnet.ChecksumMismatchError):
        torch_birefnet.ensure_checkpoint("birefnet-test", str(tmp_path / "models"))
    assert not os.listdir(tmp_path / "models"), "nothing half-downloaded is left behind"


@pytest.fixture
def torch_env(tmp_path):
    """No engine named anywhere, and the stand-in torch importable."""
    env = {k: v for k, v in os.environ.items()
           if k not in {"BGBG_ENGINE", "BGBG_NO_TORCH", "BGBG_NO_ROCM"}}
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    env["LOCALAPPDATA"] = str(tmp_path / "local")
    env["BGBG_PROFILE"] = str(tmp_path / "no-profile.json")
    env["BGBG_GPU_ENV"] = str(TESTDATA / "gpu-env" / "empty.env")
    env["PYTHONPATH"] = os.pathsep.join([str(FAKE_TORCH), env.get("PYTHONPATH", "")])
    env["BGBG_TEST_TORCH_IMPORTS"] = str(tmp_path / "torch-imports.txt")
    return env


def torch_imported(env):
    """Whether the worker imported torch: the stand-in records each import."""
    return os.path.exists(env["BGBG_TEST_TORCH_IMPORTS"])


@pytest.fixture
def start_worker(tmp_path):
    started = []

    def start(env, command=None):
        w = Worker(tmp_path / f"worker-{len(started)}.log", env=env, command=command)
        started.append(w)
        return w

    yield start
    for w in started:
        w.close()


def test_pytorch_is_on_offer_when_installed(start_worker, torch_env):
    w = start_worker(torch_env)
    assert w.ready["engines"] == ["ort", "torch"], "the frontend shows the Engine setting"


@pytest.mark.parametrize("switch", ["BGBG_NO_TORCH", "BGBG_NO_ROCM"])
def test_a_kill_switch_takes_pytorch_off_the_table(start_worker, torch_env, switch):
    env = dict(torch_env, BGBG_ENGINE="torch", **{switch: "1"})
    w = start_worker(env)
    assert w.ready["engines"] == ["ort"], f"{switch} hides the PyTorch engine"
    reply, seen = w.ask({"op": "engine_status", "id": 1}, until={"engine_status"})
    notices = [r["message"] for r in seen if r.get("type") == "notice"]
    assert any(switch in m for m in notices), "the user learns which switch is set"
    assert reply["engine"] == "ort", "BiRefNet runs on ONNX Runtime"
    assert not torch_imported(env), f"{switch} stops PyTorch before it is imported"


def test_pytorch_without_a_gpu_reports_that_it_failed_and_why(start_worker, torch_env):
    """What Enable-AmdRocm.ps1 checks before it keeps a ROCm worker."""
    env = dict(torch_env, BGBG_ENGINE="torch")
    w = start_worker(env)
    reply, _ = w.ask({"op": "engine_status", "id": 1}, until={"engine_status"})
    assert reply["engine"] == "torch", "PyTorch was asked for"
    assert reply["state"] == "failed", "a PyTorch that can't reach a GPU can't run BiRefNet"
    assert "sees no ROCm or CUDA GPU" in reply["reason"], "it failed for having no GPU"
    assert torch_imported(env), "the start-up really ran"


def test_a_birefnet_job_without_a_gpu_runs_on_onnx_runtime_and_says_why(
        start_worker, torch_env, tmp_path):
    """u2netp stands in for BiRefNet (testdata/torch_fallback_worker.py)."""
    command = [sys.executable, str(TESTDATA / "torch_fallback_worker.py"), str(WORKER.parent)]
    w = start_worker(dict(torch_env, BGBG_ENGINE="torch"), command=command)
    out = tmp_path / "cutout.png"
    reply, seen = w.ask(
        {"op": "single", "model": "u2netp", "id": 3,
         "input": str(TESTDATA / "subject.png"), "output": str(out)},
        until={"done_single", "error"})
    assert reply["type"] == "done_single" and out.exists(), "the job still completes"
    notices = [r["message"] for r in seen if r.get("type") == "notice"]
    assert any("sees no ROCm or CUDA GPU" in m and "ONNX Runtime" in m for m in notices), \
        "the user hears why PyTorch wasn't used"


def test_a_loader_thread_that_cannot_start_leaves_onnx_runtime_in_charge(monkeypatch):
    """Thread.start() can raise (out of threads); requests waiting on the
    engine must still get an answer, so they fall back to ONNX Runtime."""
    def refuse(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(torch_birefnet.threading.Thread, "start", refuse)
    engine = torch_birefnet.TorchEngine()
    engine.start()
    assert engine.wait(timeout=5) is torch_birefnet.EngineState.FAILED, "nothing waits forever"
    assert "can't start new thread" in engine.reason, "the reason reaches the user's notice"
