"""What each model is and what it costs: name, family, purpose, download, VRAM.

The model pickers show these, so a model's cost is clear before it's picked,
and a row on Auto can say what Auto chose. The frontend reads this module,
and segmentation.py sets Segment's Auto VRAM gates from the same SAM peaks.

Download sizes are the files' real sizes: rembg's release assets, SAM 2.1's
zips, MobileSAM's two ONNX files, and BiRefNet's PyTorch checkpoints.

VRAM is measured, never estimated, as the peak GPU memory of the worker's own
process (Windows' "GPU Process Memory" counters, every 50 ms) through the real
worker: load, then a cutout or a Segment-everything pass on a 1600x1598 photo.
Measured on an RX 6750 XT (12 GB): ONNX Runtime on DirectML, PyTorch on ROCm.
A model with no measurement shows "VRAM unknown".

Standard library only, so the frontend and the worker can both import it.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field

from engines.selection import Engine

MIB = 1 << 20


class Family(enum.Enum):
    """The section a model is listed under in its picker."""

    BIREFNET = "BiRefNet"
    ISNET = "ISNet"
    U2NET = "U²-Net"
    SAM2 = "SAM 2.1"
    MOBILESAM = "MobileSAM"


class Measured(enum.Enum):
    """A VRAM measurement that came out as something other than a peak."""

    # The GPU computed NaN, so the worker runs the model on the CPU instead.
    RUNS_ON_CPU = "runs on the CPU here"


@dataclass(frozen=True)
class ModelInfo:
    """One model, as the pickers describe it."""

    key: str              # rembg's session name, or Segment's rung
    name: str             # full name, for the open picker
    short: str            # the row's value, short enough for the sidebar
    family: Family
    purpose: str          # a few words on when to pick it
    download: dict = field(default_factory=dict)   # Engine -> bytes to download
    vram: dict = field(default_factory=dict)       # Engine -> peak MiB or Measured

    def engine_for(self, engine: Engine) -> Engine:
        """The engine that runs this model when `engine` is asked for.

        PyTorch runs BiRefNet only; every other model stays on ONNX Runtime.
        """
        return engine if engine in self.download else Engine.ORT

    def download_bytes(self, engine: Engine) -> int | None:
        return self.download.get(self.engine_for(engine))

    def vram_mib(self, engine: Engine) -> int | Measured | None:
        return self.vram.get(self.engine_for(engine))


def _birefnet(key, name, short, purpose, onnx, pth, vram_ort, vram_torch):
    return ModelInfo(key, name, short, Family.BIREFNET, purpose,
                     download={Engine.ORT: onnx, Engine.TORCH: pth},
                     vram={Engine.ORT: vram_ort, Engine.TORCH: vram_torch})


def _ort(key, name, short, family, purpose, onnx, vram):
    return ModelInfo(key, name, short, family, purpose,
                     download={Engine.ORT: onnx}, vram={Engine.ORT: vram})


# The two sizes every full-size BiRefNet shares: rembg's ONNX export, and the
# PyTorch checkpoint (ZhengPeng7/BiRefNet release v1) the PyTorch engine loads.
_BIREFNET_ONNX = 972_666_916
_BIREFNET_PTH = 885_082_437
# Their VRAM, in MiB. On DirectML a full BiRefNet filled the free VRAM (about
# 10 GB) and spilled 6.2 GB more into shared system memory, so its figure is
# the two together: what it asks for, more than a 12 GB card has.
_BIREFNET_ORT_MIB = 10_275 + 6_354
_BIREFNET_TORCH_MIB = 5_326

# Background removal, in picker order: rembg session name -> what it is.
BG_MODELS = {m.key: m for m in [
    _birefnet("birefnet-general", "BiRefNet General", "BiRefNet",
              "best all-round", _BIREFNET_ONNX, _BIREFNET_PTH,
              _BIREFNET_ORT_MIB, _BIREFNET_TORCH_MIB),
    _birefnet("birefnet-general-lite", "BiRefNet General Lite", "BiRefNet Lite",
              "faster and lighter than General", 224_005_088, 177_791_685,
              10_158 + 5_369, 3_642),       # spills on DirectML too
    _birefnet("birefnet-massive", "BiRefNet Massive", "BiRefNet Massive",
              "General, trained on more data", _BIREFNET_ONNX, _BIREFNET_PTH,
              _BIREFNET_ORT_MIB, _BIREFNET_TORCH_MIB),
    _birefnet("birefnet-portrait", "BiRefNet Portrait", "BiRefNet Portrait",
              "people and portraits", _BIREFNET_ONNX, _BIREFNET_PTH,
              _BIREFNET_ORT_MIB, _BIREFNET_TORCH_MIB),
    _birefnet("birefnet-hrsod", "BiRefNet HRSOD", "BiRefNet HRSOD",
              "high-resolution photos", _BIREFNET_ONNX, _BIREFNET_PTH,
              _BIREFNET_ORT_MIB, _BIREFNET_TORCH_MIB),
    _birefnet("birefnet-dis", "BiRefNet DIS", "BiRefNet DIS",
              "fine detail, like hair and wires", _BIREFNET_ONNX, _BIREFNET_PTH,
              _BIREFNET_ORT_MIB, _BIREFNET_TORCH_MIB),
    _ort("isnet-general-use", "ISNet General", "ISNet", Family.ISNET,
         "older all-round model", 178_648_008, 747),
    _ort("isnet-anime", "ISNet Anime", "ISNet Anime", Family.ISNET,
         "anime and illustrations", 176_069_933, 709),
    _ort("u2net", "U²-Net", "U²-Net", Family.U2NET,
         "fast, rough edges", 175_997_641, 454),
    _ort("u2net_human_seg", "U²-Net Human", "U²-Net Human", Family.U2NET,
         "fast, people only", 175_997_641, 454),
    _ort("silueta", "Silueta", "Silueta", Family.U2NET,
         "U²-Net shrunk to a quarter of the size", 44_173_029, 332),
]}

# rembg models the picker leaves out, each with why. The worker lists every
# rembg model, so a new rembg release that adds one fails the tests until it
# lands here or in BG_MODELS.
NOT_OFFERED = {
    "u2netp": "a 4 MB test-size U²-Net; Silueta is the small pick",
    "u2net_cloth_seg": "splits clothing into parts, not a background cutout",
    "birefnet-cod": "finds camouflaged objects, not a background cutout",
    "bria-rmbg": "licensed for non-commercial use only",
    "sam": "needs clicked prompts; Segment runs SAM 2.1 instead",
    "u2net_custom": "needs a model file of your own",
    "dis_custom": "needs a model file of your own",
    "ben_custom": "needs a model file of your own",
    # From rembg 2.0.88.
    "atlascloud": "sends the photo to Atlas Cloud's API, with an API key; the app stays local",
    "withoutbg": "sends the photo to withoutBG's API, with an API key; the app stays local",
    "birefnet-general-fp16": "BiRefNet General converted to fp16 on first use, not measured",
    "birefnet-dynamic": "BiRefNet for any input size, not measured",
}

# Segment's models, best first: rung -> what it is. Download is the zip
# (SAM 2.1) or the two ONNX files (MobileSAM). All run on ONNX Runtime.
SEG_MODELS = {m.key: m for m in [
    _ort("large", "SAM 2.1 Large", "SAM 2.1 Large", Family.SAM2,
         "most precise, slowest", 805_244_997, 3_041),
    _ort("base_plus", "SAM 2.1 Base+", "SAM 2.1 Base+", Family.SAM2,
         "precise, a little lighter than Large", 271_998_747, 2_591),
    _ort("small", "SAM 2.1 Small", "SAM 2.1 Small", Family.SAM2,
         "balanced", 142_896_844, 2_392),
    _ort("tiny", "SAM 2.1 Tiny", "SAM 2.1 Tiny", Family.SAM2,
         "fast, for smaller GPUs", 116_507_723, 2_347),
    _ort("mobile", "MobileSAM", "MobileSAM", Family.MOBILESAM,
         "fastest, fine on the CPU", 28_157_093 + 16_501_323,
         Measured.RUNS_ON_CPU),     # its encoder computes NaN on DirectML
]}


def by_family(models: dict) -> list:
    """``[(Family, [key, ...]), ...]``: runs of one family, in the table's order."""
    sections = []
    for info in models.values():
        if not sections or sections[-1][0] is not info.family:
            sections.append((info.family, []))
        sections[-1][1].append(info.key)
    return sections


class Pick(enum.Enum):
    """How a row's model came to be the one in use.

    The first three values are the worker's ``seg_ready`` modes.
    """

    AUTO = "auto"          # Auto chose it
    MANUAL = "manual"      # picked by name in the row itself
    FALLBACK = "fallback"  # what ran after the first choice failed
    PRESET = "preset"      # a friendly preset (Subject) that maps to a model


class Device(enum.Enum):
    """Where a loaded model runs, as the worker reports it."""

    GPU = "gpu"
    CPU = "cpu"


def size_text(n_bytes: int) -> str:
    """A download size the way Windows and GNOME Files show it: 928 MB, 1.2 GB."""
    mib = n_bytes / MIB
    if mib >= 1024:
        return "%.1f GB" % (mib / 1024)
    if mib >= 10:
        return "%d MB" % round(mib)
    return "%.1f MB" % mib


def vram_text(mib: int | Measured | None) -> str:
    if mib is None:
        return "VRAM unknown"
    if isinstance(mib, Measured):
        return mib.value
    if mib >= 1024:
        return "~%.1f GB VRAM" % (mib / 1024)
    return "~%d MB VRAM" % mib


def free_text(free_mib: int) -> str:
    """Free VRAM, as Auto's reason: "11.2 GB free"."""
    return "%.1f GB free" % (free_mib / 1024)


def entry_subtitle(info: ModelInfo, engine: Engine) -> str:
    """The second line of a model in an open picker: cost, then purpose."""
    parts = []
    size = info.download_bytes(engine)
    if size:
        parts.append(size_text(size) + " download")
    parts.append(vram_text(info.vram_mib(engine)))
    parts.append(info.purpose)
    return " · ".join(parts)


def row_subtitle(pick: Pick, info: ModelInfo | None, engine: Engine,
                 why: str | None = None, name: str | None = None,
                 device: Device = Device.GPU) -> str:
    """A model row's subtitle: what is in use and what it costs.

    Every model row renders through this, so a row on Auto always names its
    pick. The Engine row and Segment's live VRAM probe feed it too.

    Args:
        pick: how the model was chosen.
        info: the model, or None for one this module doesn't know.
        engine: the engine the app asked for; ``info`` says which one runs it.
        why: a few words on Auto's or the fallback's reason, e.g. "11.2 GB free".
        name: the model's name when ``info`` is None.
        device: where the model runs; on the CPU its VRAM doesn't apply.

    Returns:
        e.g. "Auto · SAM 2.1 Base+ · ~6.1 GB VRAM (11.2 GB free)" or
        "928 MB download · ~3.1 GB VRAM".
    """
    label = info.name if info else (name or "unknown model")
    vram = _vram_part(info, engine, device)
    size = info.download_bytes(engine) if info else None
    if pick is Pick.MANUAL:
        parts = [size_text(size) + " download"] if size else []
        parts.append(vram)
    elif pick is Pick.PRESET:
        parts = [label] + ([size_text(size)] if size else []) + [vram]
    else:
        lead = "Auto" if pick is Pick.AUTO else "Fallback"
        parts = [lead, label, vram]
    text = " · ".join(parts)
    return "%s (%s)" % (text, why) if why else text


def model_summary(label: str, info: ModelInfo | None, engine: Engine,
                  device: Device = Device.GPU) -> str:
    """A model and its VRAM on one line, for a folded section's subtitle.

    Args:
        label: how to name the model, e.g. "SAM 2.1 Large (Auto)".
        info: the model, or None for one this module doesn't know.
        engine: the engine the app asked for; ``info`` says which one runs it.
        device: where the model runs; on the CPU its VRAM doesn't apply.

    Returns:
        e.g. "SAM 2.1 Large (Auto) · ~3.0 GB VRAM", or "MobileSAM · on the CPU".
    """
    return "%s · %s" % (label, _vram_part(info, engine, device))


def _vram_part(info: ModelInfo | None, engine: Engine, device: Device) -> str:
    """The VRAM a row names, the same in every row: "on the CPU" when it runs there."""
    if device is Device.CPU:
        return "on the CPU"
    return vram_text(info.vram_mib(engine) if info else None)
