"""Build a rembg session the way the worker does, and print its options.

Usage: python session_options_probe.py <src/bgbg>

A script, not ``python -c``: the worker's loader re-exec reruns sys.argv,
which must name a file. The worker is imported first, as it starts itself.
Prints one JSON object.
"""
import json
import sys

sys.path.insert(0, sys.argv[1])
import worker  # noqa: E402
import onnxruntime as ort  # noqa: E402

session = worker.ort_engine.new_session("u2netp", ["CPUExecutionProvider"])
options = session.inner_session.get_session_options()
print(json.dumps({
    "optimisation_off":
        options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "intra_op_threads": options.intra_op_num_threads,
}))
