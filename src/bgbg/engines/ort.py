"""The ONNX Runtime engine: rembg sessions, provider order and session options.

Used on every platform. The worker owns the jobs, the session cache and the
CPU retry; this module only says which providers to try, how to label them,
and how to build a session with the options these models need.

Importing it imports onnxruntime, so the worker imports it only after
``plat.prepare_process()``.
"""
import os

import onnxruntime as ort

import plat
from engines import EngineError

# Background removal is optional (a segmentation-only install ships without it).
try:
    from rembg import remove  # noqa: F401  (the worker runs jobs through it)
    from rembg.sessions import sessions_class, sessions_names
    HAVE_REMBG = True
except Exception:  # noqa: BLE001
    HAVE_REMBG = False
    sessions_names = []

# Preference order; the first one onnxruntime can actually initialise wins.
PROVIDER_ORDER = [
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]
PROVIDER_LABELS = {
    "CUDAExecutionProvider": "NVIDIA (CUDA)",
    "ROCMExecutionProvider": "AMD (ROCm)",
    "MIGraphXExecutionProvider": "AMD (MIGraphX)",
    "DmlExecutionProvider": "DirectML",
    "CPUExecutionProvider": "CPU",
}

# HEURISTIC conv-algo search skips onnxruntime's default EXHAUSTIVE benchmark.
# EXHAUSTIVE re-runs on every process's first inference (no on-disk cache), so it
# would slow the first generation of *every* launch; HEURISTIC keeps it fast.
PROVIDER_OPTIONS = {
    "CUDAExecutionProvider": {"cudnn_conv_algo_search": "HEURISTIC"},
}


def provider_name(p):
    return p[0] if isinstance(p, tuple) else p


def choose_providers(available, unusable):
    """The providers to hand onnxruntime, best first, with their options.

    Args:
        available: provider names onnxruntime reports.
        unusable: provider names this machine must not use (see
            ``plat.unusable_providers``).

    Returns:
        A list of names, or ``(name, options)`` pairs for a provider with
        options. The CPU is always last, so a session can always be built.
    """
    avail = set(available) - set(unusable)
    chosen = []
    for p in PROVIDER_ORDER:
        if p not in avail:
            continue
        opts = PROVIDER_OPTIONS.get(p)
        chosen.append((p, opts) if opts else p)
    if not any(provider_name(c) == "CPUExecutionProvider" for c in chosen):
        chosen.append("CPUExecutionProvider")
    return chosen


def preferred_providers():
    return choose_providers(ort.get_available_providers(), plat.unusable_providers())


def provider_names():
    return [provider_name(p) for p in preferred_providers()]


# The first inference of a session still warms up (memory alloc, kernel load),
# so the first generation is a little slower; the UI mentions this.
GPU_PROVIDERS = {"CUDAExecutionProvider", "ROCMExecutionProvider",
                 "MIGraphXExecutionProvider"}


def session_options():
    """Session options for the rembg models.

    Graph optimisation is off for the same reason segmentation.py turns it off
    on the SAM graphs: onnxruntime's optimiser miscompiles a fused op and emits
    NaN. On a GPU provider that is worse than a bad cutout — the miscompiled
    kernel can fault the device hard enough to force a reset — so the models
    that go through rembg get the same treatment.
    """
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # rembg.new_session honours OMP_NUM_THREADS; we build sessions without it.
    if "OMP_NUM_THREADS" in os.environ:
        threads = int(os.environ["OMP_NUM_THREADS"])
        so.inter_op_num_threads = threads
        so.intra_op_num_threads = threads
    return so


class UnknownModelError(EngineError, ValueError):
    def __init__(self, model):
        super().__init__(f"unknown background-removal model: {model}")
        self.model = model


def new_session(model, providers):
    """A rembg session with our session options.

    Built from the session class, not ``rembg.new_session``: that builds its
    own SessionOptions and passes them positionally, so a ``sess_opts=`` from
    us raises TypeError (every rembg from 2.0.61 to at least 2.0.76).
    """
    cls = next((c for c in sessions_class if c.name() == model), None)
    if cls is None:
        raise UnknownModelError(model)
    return cls(model, session_options(), providers=providers)


def is_runtime_error(e):
    """True for an onnxruntime run failure — most importantly a GPU out-of-memory,
    which BiRefNet surfaces as a failed Mul node deep in the ASPP decoder."""
    if type(e).__module__.startswith("onnxruntime"):
        return True
    if isinstance(e, RuntimeWarning):      # NaN mask, promoted by the worker
        return True
    m = str(e).lower()
    return ("onnxruntimeerror" in m or "non-zero status" in m
            or "out of memory" in m or "cuda" in m or "cudnn" in m
            or "failed to allocate" in m or "hiperroroutofmemory" in m)
