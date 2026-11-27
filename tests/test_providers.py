"""Which onnxruntime providers the worker asks for, best first."""
import pytest

from engines import ort as ort_engine

CUDA = "CUDAExecutionProvider"
ROCM = "ROCMExecutionProvider"
MIGRAPHX = "MIGraphXExecutionProvider"
DML = "DmlExecutionProvider"
CPU = "CPUExecutionProvider"
# CUDA always goes with its options: HEURISTIC keeps the first run of each launch fast.
CUDA_WITH_OPTIONS = (CUDA, {"cudnn_conv_algo_search": "HEURISTIC"})
ROCM_DEVICE_CODE_MISSING = {ROCM, MIGRAPHX}


@pytest.mark.parametrize(
    "available, unusable, expected",
    [
        # Every provider there: the full preference order.
        ([CPU, DML, MIGRAPHX, ROCM, CUDA], set(),
         [CUDA_WITH_OPTIONS, ROCM, MIGRAPHX, DML, CPU]),
        # The ROCm provider has no code for this card: MIGraphX goes with it.
        ([CPU, DML, MIGRAPHX, ROCM, CUDA], ROCM_DEVICE_CODE_MISSING,
         [CUDA_WITH_OPTIONS, DML, CPU]),
        # A CPU-only install.
        ([CPU], set(), [CPU]),
        # The odd one out: onnxruntime didn't list the CPU, which is added anyway.
        ([CUDA], set(), [CUDA_WITH_OPTIONS, CPU]),
        # Providers the worker doesn't know are never asked for.
        (["TensorrtExecutionProvider", "OpenVINOExecutionProvider", CPU], set(), [CPU]),
    ],
    ids=["every provider", "rocm without device code", "cpu only",
         "cpu missing from the report", "unknown providers"],
)
def test_providers_follow_the_preference_order_with_the_cpu_last(available, unusable, expected):
    chosen = ort_engine.choose_providers(available, unusable)
    assert chosen == expected, \
        "the best usable provider comes first, and the CPU is always there to fall back to"
