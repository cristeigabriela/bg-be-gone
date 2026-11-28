"""Run the worker with u2netp standing in for a BiRefNet model, on the CPU.

Usage: python torch_fallback_worker.py <src/bgbg>

A BiRefNet request is what the PyTorch engine takes, and the smallest
BiRefNet download is hundreds of MB. Listing rembg's 4 MB u2netp among the
engine's models sends a u2netp request down the same path: PyTorch is
asked, fails, and the job falls back to ONNX Runtime. onnxruntime sees only
the CPU, so the provider that runs is known.
"""
import sys

sys.path.insert(0, sys.argv[1])
import worker  # noqa: E402  (prepares the process first, as the worker does)
import onnxruntime  # noqa: E402
from engines import torch_birefnet  # noqa: E402

torch_birefnet.MODELS["u2netp"] = torch_birefnet.MODELS["birefnet-general-lite"]
onnxruntime.get_available_providers = lambda: ["CPUExecutionProvider"]
worker.main()
