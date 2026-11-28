"""Import the worker, and report what was settled when onnxruntime first loaded.

Usage: python startup_probe.py <src/bgbg> [--gfx ARCH] [--rocm-provider FILE]

``--gfx`` stands in for the card the driver reports, and ``--rocm-provider``
for the installed ROCm provider, so the device-code check decides the same on
any machine. ``none`` means no AMD card, or no provider. Without them, the
machine's own are used.

Prints one JSON object:

- ``unusable``: the providers the platform had ruled out when onnxruntime loaded.
- ``exec_ld_library_path``: LD_LIBRARY_PATH as this process image started with
  it, which shows the loader re-exec happened (Linux only; null elsewhere).
- ``u2net_home``: U2NET_HOME when onnxruntime loaded, which rembg reads next.
- ``models_dir`` and ``config_path``: where segmentation keeps its models and
  config.
"""
import argparse
import importlib.abc
import json
import os
import sys


def _exec_environ():
    """The environment this process image was started with, not os.environ."""
    try:
        with open("/proc/self/environ", "rb") as f:
            raw = f.read()
    except OSError:
        return None
    pairs = (item.split(b"=", 1) for item in raw.split(b"\0") if b"=" in item)
    return {os.fsdecode(k): os.fsdecode(v) for k, v in pairs}


seen = {}


class _WatchOnnxruntime(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "onnxruntime" and not seen:
            seen["unusable"] = sorted(sys.modules["plat"].unusable_providers())
            seen["exec_ld_library_path"] = (_exec_environ() or {}).get("LD_LIBRARY_PATH")
            seen["u2net_home"] = os.environ.get("U2NET_HOME")
        return None    # the normal finders do the import


parser = argparse.ArgumentParser()
parser.add_argument("src")
parser.add_argument("--gfx")
parser.add_argument("--rocm-provider")
args = parser.parse_args()

sys.meta_path.insert(0, _WatchOnnxruntime())
sys.path.insert(0, args.src)
from plat import linux  # noqa: E402  (imports no onnxruntime)

# The two readers of the hardware: amdkfd's sysfs nodes and the provider on disk.
if args.gfx is not None:
    gfx = None if args.gfx == "none" else args.gfx

    def _fake_gfx_arch():
        return gfx

    linux._gfx_arch = _fake_gfx_arch
if args.rocm_provider is not None:
    provider = None if args.rocm_provider == "none" else args.rocm_provider

    def _fake_rocm_ep_path():
        return provider

    linux._rocm_ep_path = _fake_rocm_ep_path

import worker  # noqa: E402

seen["models_dir"] = worker.seg.MODELS_DIR if worker._HAVE_SEG else None
seen["config_path"] = worker.seg.CONFIG_PATH if worker._HAVE_SEG else None
print(json.dumps(seen))
