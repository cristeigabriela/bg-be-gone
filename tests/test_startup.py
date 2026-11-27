"""What the worker settles before onnxruntime loads.

gpu.env, the ROCm device-code check and the CUDA loader re-exec all have to
happen first: the GPU stack reads its settings only at start-up, and a
provider with no code for the card can reset the GPU (d34d4e8). Each test
starts its own interpreter, so the import order is the worker's own.
"""
import json
import os
import site
import subprocess
import sys
from pathlib import Path

import pytest

TESTDATA = Path(__file__).parent / "testdata"
SRC = Path(__file__).resolve().parent.parent / "src" / "bgbg"
PROBE = TESTDATA / "startup_probe.py"
GPU_ENV = TESTDATA / "gpu-env"
ROCM = TESTDATA / "rocm"
TIMEOUT_S = 180

ROCM_DROPPED = ["MIGraphXExecutionProvider", "ROCMExecutionProvider"]


def probe(extra_env, gpu_env, *args, python=sys.executable):
    """Run the start-up probe; return what it saw when onnxruntime loaded."""
    env = {k: v for k, v in os.environ.items()
           if k not in {"BGBG_NO_ROCM", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"}}
    env.update(extra_env, BGBG_GPU_ENV=str(GPU_ENV / gpu_env))
    done = subprocess.run([str(python), str(PROBE), str(SRC), *args],
                          capture_output=True, text=True, env=env, timeout=TIMEOUT_S)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "extra_env, gpu_env, gfx, provider, expected",
    [
        # No ROCm provider installed: nothing to rule out.
        ({}, "empty.env", "gfx1031", "none", []),
        # No AMD card: an installed provider can't be checked, so it stays.
        ({}, "empty.env", "none", "provider-gfx1031.txt", []),
        # The provider was built for this card.
        ({}, "empty.env", "gfx1031", "provider-gfx1031.txt", []),
        # The provider has code for a sibling card only: running it could reset the GPU.
        ({}, "empty.env", "gfx1031", "provider-gfx1030.txt", ROCM_DROPPED),
        # The kill switch, set for one launch, beats a provider that would work.
        ({"BGBG_NO_ROCM": "1"}, "empty.env", "gfx1031", "provider-gfx1031.txt", ROCM_DROPPED),
        # The kill switch in gpu.env: gpu.env must be read before the check.
        ({}, "no-rocm.env", "gfx1031", "provider-gfx1031.txt", ROCM_DROPPED),
    ],
    ids=["no provider", "no amd card", "device code present", "device code missing",
         "kill switch in the environment", "kill switch in gpu.env"],
)
def test_rocm_is_ruled_out_before_onnxruntime_is_imported(
        extra_env, gpu_env, gfx, provider, expected):
    provider_arg = provider if provider == "none" else str(ROCM / provider)
    seen = probe(extra_env, gpu_env, "--gfx", gfx, "--rocm-provider", provider_arg)
    assert seen["unusable"] == expected, \
        "the ROCm decision must already be made when onnxruntime loads"


def test_gpu_env_can_move_the_cache_and_config_folders():
    """gpu.env is read before the folders are, as it was before the seam."""
    seen = probe({}, "moved-folders.env")
    assert seen["models_dir"] == os.path.join("bgbg-test-cache", "bg-be-gone", "models"), \
        "XDG_CACHE_HOME from gpu.env decides where models are downloaded"
    assert seen["config_path"] == os.path.join("bgbg-test-config", "bg-be-gone", "seg.json"), \
        "XDG_CONFIG_HOME from gpu.env decides where the config lives"


@pytest.mark.skipif(sys.platform != "linux", reason="LD_LIBRARY_PATH and the re-exec are Linux's")
def test_bundled_cuda_libraries_are_on_the_loader_path_before_onnxruntime_loads(tmp_path):
    """onnxruntime-gpu links CUDA libraries that ship inside pip wheels. The
    loader reads LD_LIBRARY_PATH only when a process starts, so the worker
    re-execs itself with the wheels' folders on it.

    A venv of our own holds a stand-in for those folders; a .pth file lends it
    this interpreter's packages.
    """
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)],
                   check=True, timeout=TIMEOUT_S)
    site_packages = next(venv.glob("lib/python*/site-packages"))
    cuda_libs = site_packages / "nvidia" / "cuda_runtime" / "lib"
    cuda_libs.mkdir(parents=True)
    (site_packages / "lend-packages.pth").write_text(
        "\n".join(site.getsitepackages() + [site.getusersitepackages()]) + "\n")

    seen = probe({"LD_LIBRARY_PATH": ""}, "empty.env", python=venv / "bin" / "python")
    on_path = (seen["exec_ld_library_path"] or "").split(os.pathsep)
    assert str(cuda_libs) in on_path, \
        "the process that imports onnxruntime must have started with the CUDA libraries on its path"
