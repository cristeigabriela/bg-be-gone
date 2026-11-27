"""Run the worker as if onnxruntime found no GPU.

Usage: python cpu_only_worker.py <src/bgbg>

Stands in for a machine without a GPU, so a test can tell which provider
must run. Only onnxruntime's report of the hardware is replaced.
"""
import sys

sys.path.insert(0, sys.argv[1])
import worker  # noqa: E402  (prepares the process first, as the worker does)
import onnxruntime  # noqa: E402


def _cpu_only():
    return ["CPUExecutionProvider"]


onnxruntime.get_available_providers = _cpu_only
worker.main()
