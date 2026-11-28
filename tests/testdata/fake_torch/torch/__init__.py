"""A stand-in for PyTorch with no GPU, for the worker tests.

Put on PYTHONPATH, it makes torch look installed to the worker
(importlib.util.find_spec finds it) without the 4 GB of a real build. It
reports no CUDA or ROCm device, so the PyTorch engine fails its start-up and
the worker has to fall back to ONNX Runtime.

Each import appends a line to the file BGBG_TEST_TORCH_IMPORTS names, so a
test can prove torch was never imported.
"""
import os

__version__ = "0.0+stand-in"

if os.environ.get("BGBG_TEST_TORCH_IMPORTS"):
    with open(os.environ["BGBG_TEST_TORCH_IMPORTS"], "a", encoding="utf-8") as _f:
        _f.write("imported\n")


class version:  # noqa: N801  (torch.version is a module in the real package)
    hip = None
    cuda = None


class cuda:  # noqa: N801
    @staticmethod
    def is_available():
        return False
