"""What the model rows say: the subtitle every model row renders through."""
import pytest

from engines.selection import Engine
from modelinfo import Device, Family, Measured, ModelInfo, Pick, row_subtitle, size_text

ONNX_BYTES = 972_666_916     # 928 MB, BiRefNet's ONNX export
PTH_BYTES = 885_082_437      # 844 MB, its PyTorch checkpoint

# Runs on both engines, with a different VRAM peak on each.
BOTH_ENGINES = ModelInfo("both", "Both Model", "Both", Family.BIREFNET, "test",
                         download={Engine.ORT: ONNX_BYTES, Engine.TORCH: PTH_BYTES},
                         vram={Engine.ORT: 3174, Engine.TORCH: 4608})
# ONNX Runtime only, measured.
ORT_ONLY = ModelInfo("ort", "Ort Model", "Ort", Family.ISNET, "test",
                     download={Engine.ORT: ONNX_BYTES}, vram={Engine.ORT: 512})
# The GPU computed NaN, so it runs on the CPU.
CPU_HERE = ModelInfo("cpu", "Cpu Model", "Cpu", Family.MOBILESAM, "test",
                     download={Engine.ORT: ONNX_BYTES}, vram={Engine.ORT: Measured.RUNS_ON_CPU})
# Never measured.
UNMEASURED = ModelInfo("new", "New Model", "New", Family.SAM2, "test",
                       download={Engine.ORT: ONNX_BYTES})


@pytest.mark.parametrize(
    "pick, info, engine, why, name, device, expected",
    [
        # Auto names its pick and the room it had: the row the user reads on Auto.
        (Pick.AUTO, BOTH_ENGINES, Engine.ORT, "11.2 GB free", None, Device.GPU,
         "Auto · Both Model · ~3.1 GB VRAM (11.2 GB free)"),
        # Auto without a measurement still names the pick.
        (Pick.AUTO, UNMEASURED, Engine.ORT, None, None, Device.GPU,
         "Auto · New Model · VRAM unknown"),
        # A model this table doesn't know: the worker's name for it.
        (Pick.AUTO, None, Engine.ORT, None, "SAM 3 Huge", Device.GPU,
         "Auto · SAM 3 Huge · VRAM unknown"),
        # A fallback says so, not "Auto"; on the CPU, VRAM doesn't apply.
        (Pick.FALLBACK, UNMEASURED, Engine.ORT, None, None, Device.CPU,
         "Fallback · New Model · on the CPU"),
        # The free VRAM Auto picked by still shows after the model fell back to the CPU.
        (Pick.FALLBACK, UNMEASURED, Engine.ORT, "2.0 GB free", None, Device.CPU,
         "Fallback · New Model · on the CPU (2.0 GB free)"),
        # Picked by name: the row's value already names it, so only the cost.
        (Pick.MANUAL, BOTH_ENGINES, Engine.ORT, None, None, Device.GPU,
         "928 MB download · ~3.1 GB VRAM"),
        # A preset names the model it maps to.
        (Pick.PRESET, BOTH_ENGINES, Engine.ORT, None, None, Device.GPU,
         "Both Model · 928 MB · ~3.1 GB VRAM"),
        # PyTorch's own download and VRAM, when the model runs on it.
        (Pick.MANUAL, BOTH_ENGINES, Engine.TORCH, None, None, Device.GPU,
         "844 MB download · ~4.5 GB VRAM"),
        # Measured, but the GPU couldn't run it: says where it runs instead of a size.
        (Pick.AUTO, CPU_HERE, Engine.ORT, None, None, Device.GPU,
         "Auto · Cpu Model · runs on the CPU here"),
        # The odd one out: PyTorch asked for, but this model runs on ONNX Runtime.
        (Pick.MANUAL, ORT_ONLY, Engine.TORCH, None, None, Device.GPU,
         "928 MB download · ~512 MB VRAM"),
    ],
    ids=["auto measured", "auto unmeasured", "auto unknown model", "fallback on cpu",
         "fallback on cpu, free vram known", "manual", "preset", "torch", "runs on cpu",
         "torch asked, ort runs it"],
)
def test_a_model_row_names_the_pick_and_its_cost(pick, info, engine, why, name, device,
                                                 expected):
    got = row_subtitle(pick, info, engine, why=why, name=name, device=device)
    assert got == expected, "the row must say what runs and what it costs on that engine"


@pytest.mark.parametrize(
    "n_bytes, expected",
    [
        (4_574_861, "4.4 MB"),        # under 10 MB keeps a decimal
        (44_173_029, "42 MB"),
        (972_666_916, "928 MB"),
        (1_024_331_469, "977 MB"),    # 1.02 GB on the wire, but under 1024 MiB
        (1_200_000_000, "1.1 GB"),
    ],
    ids=["tiny", "small", "large", "just under a gibibyte", "over a gibibyte"],
)
def test_download_sizes_read_like_the_file_manager(n_bytes, expected):
    assert size_text(n_bytes) == expected, "sizes are in binary units, as Windows and GNOME show"
