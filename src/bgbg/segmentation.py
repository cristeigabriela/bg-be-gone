"""Segment Anything backend for bg-be-gone.

Torch-free ONNX Runtime segmentation, kept deliberately in the same stack as the
background-removal worker (onnxruntime + numpy + pillow, no torch). Provides:

- A VRAM-aware model ladder (SAM 2.1 large/base_plus/small/tiny + MobileSAM) with
  auto-selection and step-down on OOM/failure.
- ``SamSession`` — encode-once, then interactive point decoding or a numpy
  "segment everything" automatic mask generator (grid sampling + NMS).
- Weight download-on-demand into ``~/.cache/bg-be-gone/models``. The model is
  picked anew on every load; nothing about it is remembered.

Runs inside the worker's venv. The GTK frontend never imports this (it has no
numpy); masks are exchanged as PNG files on disk.
"""
import os
import zlib
import json
import colorsys
import zipfile
import tempfile
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter
import onnxruntime as ort

from plat import config_dir, models_dir, probe_vram

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = models_dir()
CONFIG_DIR = config_dir()
CONFIG_PATH = os.path.join(CONFIG_DIR, "seg.json")

_HF = "https://huggingface.co"

# ImageNet normalisation (SAM 2.1 preprocessing).
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

# ---------------------------------------------------------------------------
# Model registry / ladder
# ---------------------------------------------------------------------------
# Peak dedicated VRAM (MiB) each SAM 2.1 model used to segment everything on a
# 1600x1598 photo, measured on DirectML on 2026-11-27.
_MEASURED_PEAK_MIB = {"large": 3041, "base_plus": 2591, "small": 2392, "tiny": 2347}
# Headroom over the measured peak: a bigger photo, another provider, the driver.
_GATE_FACTOR = 1.25
_GATE_SLACK_MIB = 512


def _gate(rung):
    """The free VRAM (MiB) Auto wants before it picks `rung`."""
    return round(_MEASURED_PEAK_MIB[rung] * _GATE_FACTOR) + _GATE_SLACK_MIB


# ``vram_gate`` is the minimum *free* VRAM (MiB) for auto-selection to pick this
# rung: about 4.3 GB for Large, down to about 3.4 GB for Tiny.
MODELS = {
    "large": dict(
        family="sam2", label="SAM 2.1 Large", vram_gate=_gate("large"),
        repo="vietanhdev/segment-anything-2.1-onnx-models",
        zip="sam2.1_hiera_large_20260221.zip", stem="sam2.1_hiera_large"),
    "base_plus": dict(
        family="sam2", label="SAM 2.1 Base+", vram_gate=_gate("base_plus"),
        repo="vietanhdev/segment-anything-2.1-onnx-models",
        zip="sam2.1_hiera_base_plus_20260221.zip", stem="sam2.1_hiera_base_plus"),
    "small": dict(
        family="sam2", label="SAM 2.1 Small", vram_gate=_gate("small"),
        repo="vietanhdev/segment-anything-2.1-onnx-models",
        zip="sam2.1_hiera_small_20260221.zip", stem="sam2.1_hiera_small"),
    "tiny": dict(
        family="sam2", label="SAM 2.1 Tiny", vram_gate=_gate("tiny"),
        repo="vietanhdev/segment-anything-2.1-onnx-models",
        zip="sam2.1_hiera_tiny_20260221.zip", stem="sam2.1_hiera_tiny"),
    "mobile": dict(
        family="sam1", label="MobileSAM", vram_gate=0,
        repo="Acly/MobileSAM",
        files=["mobile_sam_image_encoder.onnx", "sam_mask_decoder_single.onnx"]),
}

# Order used when stepping DOWN after a failure (best → cheapest).
LADDER = ["large", "base_plus", "small", "tiny", "mobile"]
# Rungs auto-selection may choose, best first; MobileSAM is the fallback.
_AUTO_ORDER = ["large", "base_plus", "small", "tiny"]


class SegError(Exception):
    pass


class ComputeError(SegError):
    """A model ran but produced garbage (e.g. NaN from a broken GPU kernel).
    Retry the same model on CPU rather than dropping to a smaller one."""


class SegCanceled(SegError):
    """The caller's cancel event was set while the work was running."""

    def __init__(self, stage):
        super().__init__(f"canceled while {stage}")
        self.stage = stage


class SegAcquireError(SegError):
    """Getting a model's files failed, as opposed to loading or running them.

    It says nothing about whether the model fits, so it is never a reason to
    step down to a smaller one. `user_message` is the text the window shows.
    """


class SegDownloadError(SegAcquireError):
    """No SAM model could be loaded because its download failed (usually offline)."""

    def __init__(self, rung):
        super().__init__(f"could not download the {rung} model")
        self.rung = rung
        self.user_message = ("No internet connection. The Segment model is downloaded "
                             "once, on its first use: connect and try again.")


class SegDownloadIncompleteError(SegAcquireError):
    """A download ended before the size the server announced."""

    def __init__(self, url, done, total):
        super().__init__(f"download of {url} ended at {done} of {total} bytes")
        self.url = url
        self.done = done
        self.total = total
        self.user_message = ("The Segment model's download stopped early (%d of %d MB). "
                             "Try again." % (done >> 20, total >> 20))


class SegModelArchiveError(SegAcquireError):
    """A downloaded model archive can't be read; it has been removed."""

    def __init__(self, path, reason):
        super().__init__(f"bad model archive {path}: {reason}")
        self.path = path
        self.reason = reason
        self.user_message = ("The downloaded Segment model was damaged, so it was removed. "
                             "Try again to download it afresh.")


def _is_download_error(exc):
    """True when `exc` is a download that couldn't reach the server."""
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError))


def _provider_name(p):
    return p[0] if isinstance(p, tuple) else p


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------
def load_config():
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(**kw):
    """Merge `kw` into the config; a value of None removes that key."""
    cfg = load_config()
    cfg.update(kw)
    cfg = {k: v for k, v in cfg.items() if v is not None}
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f)
        os.replace(tmp, CONFIG_PATH)
    except Exception:
        pass


def rung_for_vram(free_mb, gpu):
    """Auto-select rung from free VRAM: the best one whose gate it clears."""
    if not gpu:
        return "mobile"
    for name in _AUTO_ORDER:
        if free_mb >= MODELS[name]["vram_gate"]:
            return name
    return "mobile"


# ---------------------------------------------------------------------------
# Weight download
# ---------------------------------------------------------------------------
def _download(url, dst, progress=None):
    """Fetch `url` into `dst`, through a partial file of its own.

    ``progress(done, total)`` runs after every chunk, with total 0 when the
    server sends no size; it may raise to stop the download (the worker's
    cancel does). Whatever stops it, the partial file is removed.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst),
                               prefix=os.path.basename(dst) + ".", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(url, timeout=60) as r:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
        # A connection that drops can still end the read cleanly.
        if total and done != total:
            raise SegDownloadIncompleteError(url, done, total)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def ensure_model(rung, progress=None):
    """Return ``(encoder_path, decoder_path)``, downloading on first use."""
    spec = MODELS[rung]
    if spec["family"] == "sam1":
        enc, dec = spec["files"]
        enc_p = os.path.join(MODELS_DIR, enc)
        dec_p = os.path.join(MODELS_DIR, dec)
        for name, path in ((enc, enc_p), (dec, dec_p)):
            if not os.path.exists(path):
                _download(f"{_HF}/{spec['repo']}/resolve/main/{name}", path, progress)
        return enc_p, dec_p
    # sam2: a zip holding <stem>.encoder.onnx and <stem>.decoder.onnx
    stem = spec["stem"]
    out_dir = os.path.join(MODELS_DIR, stem)
    enc_p = os.path.join(out_dir, f"{stem}.encoder.onnx")
    dec_p = os.path.join(out_dir, f"{stem}.decoder.onnx")
    if os.path.exists(enc_p) and os.path.exists(dec_p):
        return enc_p, dec_p
    zip_p = os.path.join(MODELS_DIR, spec["zip"])
    if not os.path.exists(zip_p):
        _download(f"{_HF}/{spec['repo']}/resolve/main/{spec['zip']}", zip_p, progress)
    os.makedirs(out_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_p) as z:
            bad = z.testzip()          # every member's CRC, before anything is kept
            if bad is not None:
                raise zipfile.BadZipFile(f"{bad} fails its CRC check")
            for member in z.namelist():
                if member.endswith(".onnx"):
                    target = enc_p if "encoder" in member else dec_p
                    # Through a part file: is_cached trusts any file at target.
                    with open(target + ".part", "wb") as f:
                        f.write(z.read(member))
                    os.replace(target + ".part", target)
    except zipfile.BadZipFile as e:
        # Kept, it would fail every later load the same way.
        os.remove(zip_p)
        raise SegModelArchiveError(zip_p, str(e)) from e
    try:
        os.remove(zip_p)
    except OSError:
        pass
    if not (os.path.exists(enc_p) and os.path.exists(dec_p)):
        raise SegError(f"{rung}: expected encoder/decoder onnx not found in zip")
    return enc_p, dec_p


def is_cached(rung):
    try:
        spec = MODELS[rung]
    except KeyError:
        return False
    if spec["family"] == "sam1":
        return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in spec["files"])
    d = os.path.join(MODELS_DIR, spec["stem"])
    return (os.path.exists(os.path.join(d, f"{spec['stem']}.encoder.onnx"))
            and os.path.exists(os.path.join(d, f"{spec['stem']}.decoder.onnx")))


# ---------------------------------------------------------------------------
# Mask helpers (numpy)
# ---------------------------------------------------------------------------
def _stability(logit, offset=1.0):
    """SAM stability score: area(>+t) / area(>-t)."""
    high = np.count_nonzero(logit > offset)
    low = np.count_nonzero(logit > -offset)
    return (high / low) if low else 0.0


def _bbox(mask):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _iou(a, b):
    inter = np.count_nonzero(a & b)
    if inter == 0:
        return 0.0
    union = np.count_nonzero(a) + np.count_nonzero(b) - inter
    return inter / union if union else 0.0


def _bbox_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def _nms(records, iou_thresh=0.7, max_objects=256, cancel=None):
    """Greedy IoU-NMS. ``records`` = list of ``(score, mask_bool, ...)``; any
    further items ride along. Bbox pre-filter then mask IoU; returns the
    surviving records sorted by score desc, or nothing once ``cancel`` is set."""
    records = sorted(records, key=lambda r: r[0], reverse=True)
    kept, kept_bb = [], []
    for rec in records:
        if cancel is not None and cancel.is_set():
            return []
        mask = rec[1]
        bb = _bbox(mask)
        if bb[2] == 0:
            continue
        drop = False
        for km, kbb in zip(kept, kept_bb):
            if _bbox_overlap(bb, kbb) and _iou(mask, km[1]) > iou_thresh:
                drop = True
                break
        if not drop:
            kept.append(rec)
            kept_bb.append(bb)
            if len(kept) >= max_objects:
                break
    return kept


# ---------------------------------------------------------------------------
# SAM session
# ---------------------------------------------------------------------------
class SamSession:
    """Loaded SAM ONNX pair (encoder + decoder) for one rung.

    Call :meth:`encode` once per image, then :meth:`decode_points` (interactive)
    or :meth:`everything` (automatic). Masks returned as ``bool`` HxW arrays in
    original-image pixels.
    """

    def __init__(self, rung, family, encoder, decoder):
        self.rung = rung
        self.family = family
        self.encoder = encoder
        self.decoder = decoder
        self._enc_input = encoder.get_inputs()[0].name
        self._emb = None            # tuple of embedding tensors
        self._emb_dev = None        # the same tensors resident on the GPU (DirectML)
        self._dml_device = 0        # the DirectML adapter, as the session options name it
        self._size = None           # (W, H) original
        self._scale = None          # 1024 / max(W, H)
        self._nw = self._nh = None  # resized-to-1024 content dims

    @classmethod
    def load(cls, rung, providers, progress=None):
        spec = MODELS[rung]
        enc_p, dec_p = ensure_model(rung, progress)
        dml = _provider_name(providers[0]) == "DmlExecutionProvider"

        def options(level):
            so = ort.SessionOptions()
            so.log_severity_level = 3
            so.graph_optimization_level = level
            if dml:
                # Required by the DirectML provider (onnxruntime docs).
                so.enable_mem_pattern = False
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            return so

        # onnxruntime 1.27's graph optimiser miscompiles a fused op in these
        # SAM ONNX graphs and emits NaN (nondeterministically, on both CPU and
        # CUDA). Disabling optimisation avoids the buggy pass and is measurably
        # faster here too. See the NaN guard in encode() as a backstop.
        off = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # DirectML is the exception for the encoder: with optimisation off it
        # dispatches every one of the encoder's small ops on its own, which
        # made SAM 2.1 Large encode in 2.41 s on an RX 6750 XT. The basic level
        # only folds constants and drops redundant nodes (no fusion), gave
        # bit-identical embeddings and no NaN over 16 runs, and encodes in
        # 0.58 s (Tiny: 1.82 s -> 0.21 s).
        enc_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC if dml else off
        enc = ort.InferenceSession(enc_p, options(enc_level), providers=providers)
        dec = ort.InferenceSession(dec_p, options(off), providers=providers)
        session = cls(rung, spec["family"], enc, dec)
        if dml and isinstance(providers[0], tuple):
            session._dml_device = int(providers[0][1].get("device_id", 0))
        return session

    def provider(self):
        act = self.encoder.get_providers()
        return act[0] if act else "CPUExecutionProvider"

    # ---- encode ----
    def encode(self, pil):
        pil = pil.convert("RGB")
        W, H = pil.size
        self._size = (W, H)
        self._scale = 1024.0 / max(W, H)
        self._nw = round(W * self._scale)
        self._nh = round(H * self._scale)
        if self.family == "sam2":
            arr = np.asarray(pil.resize((self._nw, self._nh), Image.BILINEAR),
                             np.float32) / 255.0
            arr = (arr - _MEAN) / _STD
            buf = np.zeros((1024, 1024, 3), np.float32)
            buf[:self._nh, :self._nw] = arr
            inp = np.ascontiguousarray(buf.transpose(2, 0, 1)[None])
            he0, he1, emb = self.encoder.run(None, {"image": inp})
            self._emb = (emb, he0, he1)
            # Every decode feeds the same ~16 MB of embeddings. On DirectML,
            # upload them once per image instead of once per click: 35.8 ->
            # 21.3 ms per decode on SAM 2.1 Large, identical logits.
            self._emb_dev = None
            if self.decoder.get_providers()[0] == "DmlExecutionProvider":
                self._emb_dev = tuple(
                    ort.OrtValue.ortvalue_from_numpy(t, "dml", self._dml_device)
                    for t in self._emb)
        else:  # sam1 / mobile — encoder takes an HWC 0-255 image and does its own
            # normalise + pad-to-1024, but it does NOT resize: the longest side
            # must already be 1024 or the content lands in the corner of the
            # padded canvas at the wrong scale (and >1024 images get cropped,
            # since the pad amount goes negative).
            small = pil.resize((self._nw, self._nh), Image.BILINEAR)
            emb = self.encoder.run(
                None, {self._enc_input: np.asarray(small, np.float32)})[0]
            self._emb = (emb,)
        # A broken/unsupported GPU kernel (e.g. very new arch) silently emits
        # NaN instead of erroring. Treat that as a compute failure so the ladder
        # can step down / fall back to CPU.
        if any(np.isnan(t).any() for t in self._emb):
            raise ComputeError(f"{self.rung}: encoder produced NaN on "
                               f"{self.provider()}")
        return self

    # ---- low-level decode ----
    def _decode_sam2(self, coords, labels, mask_input=None):
        """coords/labels already in 1024-space, shape [G, P, *]. A padding point
        is appended per prompt. Returns (logits[G,3,256,256], iou[G,3])."""
        emb, he0, he1 = self._emb
        G = coords.shape[0]
        pad_c = np.zeros((G, 1, 2), np.float32)
        pad_l = np.full((G, 1), -1.0, np.float32)
        pc = np.concatenate([coords, pad_c], axis=1)
        pl = np.concatenate([labels, pad_l], axis=1)
        if mask_input is None:
            mi = np.zeros((G, 1, 256, 256), np.float32)
            hmi = np.zeros((G,), np.float32)
        else:
            mi = mask_input
            hmi = np.ones((G,), np.float32)
        prompts = {"point_coords": pc, "point_labels": pl,
                   "mask_input": mi, "has_mask_input": hmi}
        if self._emb_dev is not None:
            b = self.decoder.io_binding()
            for name, v in zip(("image_embed", "high_res_feats_0", "high_res_feats_1"),
                               self._emb_dev):
                b.bind_ortvalue_input(name, v)
            for name, v in prompts.items():
                b.bind_cpu_input(name, v)
            for o in self.decoder.get_outputs():
                b.bind_output(o.name, "cpu")
            self.decoder.run_with_iobinding(b)
            out = b.copy_outputs_to_cpu()
            return out[0], out[1]
        out = self.decoder.run(None, {
            "image_embed": emb, "high_res_feats_0": he0, "high_res_feats_1": he1,
            **prompts})
        return out[0], out[1]

    def _crop(self, low):
        """The part of 256x256 logits over the padded 1024 input that covers the image."""
        vh = max(1, round(self._nh / 4.0))
        vw = max(1, round(self._nw / 4.0))
        return np.ascontiguousarray(low[:vh, :vw])

    def _upscale(self, crop):
        """Cropped low-res logits (see _crop) -> original-size bool mask."""
        im = Image.fromarray(crop, mode="F").resize(self._size, Image.BILINEAR)
        return np.asarray(im) > 0.0

    def _crop_upscale(self, low):
        """256x256 logits over the padded 1024 input -> original-size bool mask."""
        return self._upscale(self._crop(low))

    # ---- interactive point prompt ----
    def decode_points(self, points, labels, prev_low=None):
        """points: list of (x, y) in ORIGINAL px. labels: 1 fg / 0 bg.
        Returns (mask_bool HxW, score, low_res[1,256,256] for refinement)."""
        s = self._scale
        coords = np.array([[[x * s, y * s] for x, y in points]], np.float32)
        labs = np.array([labels], np.float32)
        if self.family == "sam2":
            logits, iou = self._decode_sam2(coords, labs, prev_low)
            best = int(np.argmax(iou[0]))
            low = logits[0, best]
            return self._crop_upscale(low), float(iou[0, best]), low[None, None]
        W, H = self._size
        emb, = self._emb
        mi = prev_low if prev_low is not None else np.zeros((1, 1, 256, 256), np.float32)
        hmi = np.array([1.0 if prev_low is not None else 0.0], np.float32)
        out = self.decoder.run(None, {
            "image_embeddings": emb, "point_coords": coords, "point_labels": labs,
            "mask_input": mi, "has_mask_input": hmi,
            "orig_im_size": np.array([H, W], np.float32)})
        mask = out[0][0, 0] > 0.0
        return mask, float(out[1][0, 0]), out[2]

    def _grid_decode(self, x, y):
        """Decode a single grid point (1024-space). Returns (low256 logit, score).

        Decodes one prompt at a time: these ONNX decoder exports don't support a
        ``num_labels>1`` batch (the mask-gate Mul can't broadcast), and the
        decoder is light enough that per-point calls are fine, especially on GPU.
        """
        if self.family == "sam2":
            pc = np.array([[[x, y], [0, 0]]], np.float32)
            pl = np.array([[1, -1]], np.float32)
            logits, iou = self._decode_sam2(pc, pl)
            best = int(np.argmax(iou[0]))
            return logits[0, best], float(iou[0, best])
        emb, = self._emb
        W, H = self._size
        out = self.decoder.run(None, {
            "image_embeddings": emb,
            "point_coords": np.array([[[x, y]]], np.float32),
            "point_labels": np.array([[1]], np.float32),
            "mask_input": np.zeros((1, 1, 256, 256), np.float32),
            "has_mask_input": np.zeros((1,), np.float32),
            "orig_im_size": np.array([H, W], np.float32)})
        return out[2][0, 0], float(out[1][0, 0])

    # ---- automatic "everything" ----
    def everything(self, points_per_side=32, pred_iou_thresh=0.86,
                   stability_thresh=0.90, min_area_frac=0.0008,
                   cancel=None, progress=None):
        """Grid-sampled automatic mask generation. Returns a list of bool masks
        (HxW) sorted largest-first. Cooperative-cancellable via ``cancel`` (a
        threading.Event); ``progress(done, total)`` is called after every grid
        point, whether or not it found an object.

        Candidates are filtered and de-duplicated on their low-resolution masks
        (at most 256x256), and only the survivors are scaled to full size: at
        4K a full-size candidate took 8.3 MB and one overlap test 11 ms.
        """
        W, H = self._size
        xs = np.linspace(0, self._nw, points_per_side + 2)[1:-1]
        ys = np.linspace(0, self._nh, points_per_side + 2)[1:-1]
        grid = [(float(x), float(y)) for y in ys for x in xs]
        total = len(grid)
        min_area = max(64, int(min_area_frac * W * H))
        records = []          # (score, low-res bool mask, low-res logits)
        for i, (x, y) in enumerate(grid):
            if cancel is not None and cancel.is_set():
                break
            lg, score = self._grid_decode(x, y)
            if score >= pred_iou_thresh and _stability(lg) >= stability_thresh:
                crop = self._crop(lg)
                low = crop > 0.0
                # The low-res area, in full-size pixels.
                if np.count_nonzero(low) * (W * H) / low.size >= min_area:
                    records.append((score, low, crop))
            if progress:
                progress(i + 1, total)

        def canceled():
            return cancel is not None and cancel.is_set()

        # A cancel after the grid still stops here: de-duplicating and scaling
        # up hundreds of candidates takes seconds of its own.
        kept = [] if canceled() else _nms(records, iou_thresh=0.7, cancel=cancel)
        masks = []
        for _, _, crop in kept:
            if canceled():
                return []
            masks.append(self._upscale(crop))
        masks.sort(key=np.count_nonzero, reverse=True)
        return masks


# ---------------------------------------------------------------------------
# Auto-selection with step-down
# ---------------------------------------------------------------------------
# Keys older versions kept in seg.json to remember a step-down. Nothing reads
# them: a bad moment (another app holding VRAM, a driver hiccup) must not
# lower the quality of later loads.
_STALE_KEYS = ("rung", "stepped_down_at")


def _looks_like_oom(exc):
    s = str(exc).lower()
    return any(k in s for k in ("out of memory", "oom", "cuda", "cublas",
                                "cudnn", "failed to allocate", "hipmalloc",
                                "rocm", "miopen"))


@dataclass(frozen=True)
class SamPick:
    """The SAM session auto_select_session loaded, and how it was picked."""

    session: "SamSession"
    rung: str
    mode: str                     # auto, manual or fallback
    vram_free: Optional[int]      # MiB free when Auto picked; None when not probed


def auto_select_session(providers, gpu, want="auto", probe_image=None,
                        on_step=None, on_progress=None):
    """Pick and load a SAM session, stepping down the ladder on failure.

    ``providers``    onnxruntime provider list (from the worker).
    ``gpu``          whether a GPU execution provider is actually active.
    ``want``         "auto" or an explicit rung name.
    ``probe_image``  a PIL image used to force the first encode (where the GPU
                     actually reserves workspace and OOM/NaN surface).
    ``on_step``      called ``(rung, exc)`` each time an attempt is abandoned.

    Auto picks from the VRAM free right now, on every load. Failure handling
    differs by cause, and lasts for this load only; nothing is remembered:

    - out of memory: step down to a smaller model, keeping the GPU.
    - a compute failure (NaN from a broken GPU kernel, or any other error):
      retry the *same* model on CPU before stepping down, so a too-new GPU
      degrades to CPU without needless downloads.
    - a failed download, or a model file that arrived damaged: raise a
      SegAcquireError. It says nothing about the model, so a smaller one that
      happens to be cached must not stand in.

    Returns a SamPick.
    """
    cpu_only = ["CPUExecutionProvider"]
    have_gpu = gpu and any(_provider_name(p) != "CPUExecutionProvider"
                           for p in providers)

    if want == "auto":
        vram = probe_vram()
        vram_free = vram["free"] if vram else None
        start = rung_for_vram(vram_free or 0, have_gpu and bool(vram))
        base_mode = "auto"
        if any(k in load_config() for k in _STALE_KEYS):
            save_config(**{k: None for k in _STALE_KEYS})
    else:
        start = want if want in LADDER else "base_plus"
        base_mode = "manual"
        vram_free = None               # nothing was probed for a model the user chose

    def load(name, prov):
        try:
            sess = SamSession.load(name, prov, progress=on_progress)
        except SegAcquireError:
            raise
        except Exception as e:
            if _is_download_error(e):
                raise SegDownloadError(name) from e
            raise
        if probe_image is not None:
            sess.encode(probe_image)
        return sess

    for name in LADDER[LADDER.index(start):]:
        attempts = ([providers, cpu_only] if have_gpu else [cpu_only])
        for prov in attempts:
            try:
                sess = load(name, prov)
                first = name == start and prov is attempts[0]
                return SamPick(sess, name, base_mode if first else "fallback", vram_free)
            except SegAcquireError:
                raise
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "_Canceled":
                    raise      # a cancel isn't a load failure — don't step down
                if on_step:
                    on_step(name, e)
                if _looks_like_oom(e):
                    break  # a smaller model helps -> next rung on the GPU
                # compute/NaN error: try the next provider (CPU) for this model
                continue

    # Ultimate fallback: MobileSAM forced onto CPU.
    return SamPick(load("mobile", cpu_only), "mobile", "fallback", vram_free)


# ---------------------------------------------------------------------------
# Object packaging: colors, label map, per-object PNGs, extract compositing
# ---------------------------------------------------------------------------
def _obj_color(cx, cy):
    """Deterministic, position-hashed colour so an object keeps the same colour
    across selections and re-runs (not tied to detection order)."""
    key = f"{int(cx) // 8}:{int(cy) // 8}".encode()
    h = (zlib.crc32(key) % 3600) / 3600.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.68, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def _save_soft(mask, path, feather):
    """Write a mask as a soft-edged (anti-aliased) L PNG."""
    im = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L")
    if feather > 0:
        im = im.filter(ImageFilter.GaussianBlur(feather))
    im.save(path)


# Clockwise 8-neighbour offsets (row, col) starting at West — for Moore tracing.
_CW = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]


def _trace_boundary(mask):
    """Moore-neighbour boundary tracing. Returns the ordered outer contour
    (list of (x, y)) of the component containing the topmost-leftmost pixel."""
    H, W = mask.shape
    m = np.zeros((H + 2, W + 2), bool)
    m[1:-1, 1:-1] = mask
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return []
    k = np.lexsort((xs, ys))[0]              # topmost, then leftmost
    start = (int(ys[k]), int(xs[k]))
    cur = start
    back = (start[0], start[1] - 1)          # approached from the West (bg)
    out = [cur]
    limit = 8 * int(m.sum()) + 16
    for _ in range(limit):
        bi = _CW.index((back[0] - cur[0], back[1] - cur[1]))
        nxt = None
        for j in range(1, 9):
            dy, dx = _CW[(bi + j) % 8]
            p = (cur[0] + dy, cur[1] + dx)
            if m[p[0], p[1]]:
                pdy, pdx = _CW[(bi + j - 1) % 8]
                back = (cur[0] + pdy, cur[1] + pdx)
                nxt = p
                break
        if nxt is None or nxt == start:
            break
        cur = nxt
        out.append(cur)
    return [(c[1] - 1, c[0] - 1) for c in out]   # -> (x, y), unpad


def _rdp(pts, eps):
    """Ramer-Douglas-Peucker polyline simplification (keeps endpoints)."""
    n = len(pts)
    if n < 3:
        return list(pts)
    P = np.asarray(pts, float)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = P[b] - P[a]
        L2 = float(seg.dot(seg))
        rel = P[a + 1:b] - P[a]
        if L2 == 0.0:
            d = np.hypot(rel[:, 0], rel[:, 1])
        else:
            t = np.clip((rel @ seg) / L2, 0.0, 1.0)
            proj = P[a] + t[:, None] * seg
            d = np.hypot(*(P[a + 1:b] - proj).T)
        if d.size == 0:
            continue
        mi = int(np.argmax(d))
        if d[mi] > eps:
            idx = a + 1 + mi
            keep[idx] = True
            stack.append((a, idx))
            stack.append((idx, b))
    return [pts[i] for i in range(n) if keep[i]]


def _label(mask, connectivity=8):
    """Label connected components (iterative BFS). Returns (labels, count)."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    if connectivity == 8:
        nbr = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    else:
        nbr = ((-1, 0), (1, 0), (0, -1), (0, 1))
    cur = 0
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if lab[sy, sx]:
            continue
        cur += 1
        dq = deque(((sy, sx),))
        lab[sy, sx] = cur
        while dq:
            y, x = dq.popleft()
            for dy, dx in nbr:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = cur
                    dq.append((ny, nx))
    return lab, cur


def contour(mask, epsilon=1.8, min_area=80, max_pts=500, max_dim=480):
    """All boundary polylines of ``mask`` (each a list of [x, y] in original-image
    px): the outer contour of every connected component PLUS its holes, so
    complex or disjoint objects (branchy trees, rings) outline correctly. The
    mask is traced at reduced resolution for speed on large images, then scaled
    back. Empty for tiny masks."""
    if np.count_nonzero(mask) < min_area:
        return []
    H, W = mask.shape
    if max(H, W) > max_dim:
        sc = max_dim / max(H, W)
        sw, sh = max(1, round(W * sc)), max(1, round(H * sc))
        small = np.asarray(Image.fromarray(
            np.where(mask, 255, 0).astype(np.uint8)).resize(
            (sw, sh), Image.NEAREST)) > 127
    else:
        small, sw, sh = mask, W, H
    up = W / sw                                  # small px -> original px
    small_min = max(8, int(min_area * (sw * sh) / (W * H)))

    regions = []
    lab, n = _label(small, 8)
    for k in range(1, n + 1):
        comp = lab == k
        if np.count_nonzero(comp) >= small_min:
            regions.append(comp)
    # holes = background components that don't touch the border
    blab, bn = _label(~small, 4)
    border = (set(blab[0, :].tolist()) | set(blab[-1, :].tolist())
              | set(blab[:, 0].tolist()) | set(blab[:, -1].tolist()))
    for k in range(1, bn + 1):
        if k in border:
            continue
        hole = blab == k
        if np.count_nonzero(hole) >= max(6, small_min // 4):
            regions.append(hole)

    polys = []
    for reg in regions:
        poly = _trace_boundary(reg)
        if len(poly) < 3:
            continue
        poly = [(x * up, y * up) for x, y in poly]
        poly = _rdp(poly, epsilon)
        if len(poly) > max_pts:
            step = int(np.ceil(len(poly) / max_pts))
            poly = poly[::step]
        polys.append([[int(round(x)), int(round(y))] for x, y in poly])
    return polys


def _save_idmap(arr, path):
    """Encode a uint16 id-per-pixel array as ``R + G*256`` in an RGB PNG."""
    H, W = arr.shape
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[..., 0] = (arr & 0xFF).astype(np.uint8)
    rgb[..., 1] = (arr >> 8).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(path)
    return path


def save_objects(masks, size, out_dir, prefix, feather=0.8, cancel=None, progress=None):
    """Write soft per-object mask PNGs + three lookup maps for hit-testing
    (``masks`` is largest-first). Returns ``(maps, objects)`` where maps has:
      - ``label``   : the most *specific* object per pixel (smallest on top)
      - ``general`` : the most *general* object per pixel (largest on top)
      - ``depth``   : how many objects overlap each pixel (an ``L`` PNG)
    ids are encoded ``R + G*256`` (0 = background).

    Each object takes ~0.4 s at 4K (its PNG and outline), so ``progress(done,
    total)`` is called after each one and ``cancel`` (a threading.Event) is
    checked before each; a set cancel raises SegCanceled."""
    W, H = size
    os.makedirs(out_dir, exist_ok=True)
    label = np.zeros((H, W), np.uint16)      # specific: smallest painted last
    general = np.zeros((H, W), np.uint16)    # general: largest painted last
    depth = np.zeros((H, W), np.uint16)
    objs = []
    for idx, mask in enumerate(masks):
        if cancel is not None and cancel.is_set():
            raise SegCanceled("building layers")
        if progress and idx:
            progress(idx, len(masks))
        oid = idx + 1
        label[mask] = oid
        depth += mask
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        mp = os.path.join(out_dir, f"{prefix}_obj{oid}.png")
        _save_soft(mask, mp, feather)
        x, y, w, h = _bbox(mask)
        objs.append({"id": oid, "color": "#%02x%02x%02x" % _obj_color(cx, cy),
                     "bbox": [x, y, w, h], "area": int(xs.size), "mask": mp,
                     "contour": contour(mask)})
    # The lookup maps take a while too: check once more, and count the run as
    # done only once they are written.
    if cancel is not None and cancel.is_set():
        raise SegCanceled("building layers")
    for idx in range(len(masks) - 1, -1, -1):   # largest painted last -> on top
        general[masks[idx]] = idx + 1
    maps = {
        "label": _save_idmap(label, os.path.join(out_dir, f"{prefix}_labelmap.png")),
        "general": _save_idmap(general, os.path.join(out_dir, f"{prefix}_general.png")),
        "depth": os.path.join(out_dir, f"{prefix}_depth.png"),
    }
    Image.fromarray(np.minimum(depth, 255).astype(np.uint8), "L").save(maps["depth"])
    if progress and masks:
        progress(len(masks), len(masks))
    return maps, objs


def save_mask(mask, path, feather=0.8):
    """Write a single soft-edged mask PNG; return its (x, y, w, h) bbox."""
    _save_soft(mask, path, feather)
    return list(_bbox(mask))


def load_union(paths):
    """Combine soft mask PNGs into one uint8 alpha (max), keeping soft edges."""
    out = None
    for p in paths:
        a = np.asarray(Image.open(p).convert("L"))
        out = a if out is None else np.maximum(out, a)
    return out


def _hex_to_rgb(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def composite_extract(src_pil, alpha, bg, dst, blur=20):
    """Composite the ``alpha`` (uint8 HxW) cutout of ``src_pil`` over ``bg`` and
    save to ``dst``. ``bg`` is "transparent" / "blur" / "#rrggbb"."""
    src = src_pil.convert("RGB")
    cutout = src.convert("RGBA")
    cutout.putalpha(Image.fromarray(alpha.astype(np.uint8), "L"))
    if bg == "transparent":
        cutout.save(dst)
    elif bg == "blur":
        base = src.filter(ImageFilter.GaussianBlur(max(1, blur))).convert("RGBA")
        base.alpha_composite(cutout)
        base.convert("RGB").save(dst)
    else:
        flat = Image.new("RGBA", cutout.size, _hex_to_rgb(bg) + (255,))
        flat.alpha_composite(cutout)
        flat.convert("RGB").save(dst)
    return dst
