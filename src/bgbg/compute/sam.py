"""Segment Anything: the model ladder, the session, the automatic mask generator.

Torch-free ONNX Runtime segmentation, deliberately in the same stack as the
background-removal worker (onnxruntime + numpy + pillow, no torch):

- a VRAM-aware model ladder (SAM 2.1 large/base_plus/small/tiny + MobileSAM) with
  auto-selection and step-down on OOM or garbage output;
- ``SamSession`` -- encode once, then interactive point decoding, or a numpy
  "segment everything" grid pass;
- weight download-on-demand into ``~/.cache/bg-be-gone/models``, with the
  last-good rung remembered in ``~/.config/bg-be-gone/seg.json``.

What happens to a mask *after* the network -- contours, labelmaps, NMS, the id
maps, compositing -- is ``compute.maskops``, which needs no model and is the half
that must be mirrored in TypeScript.

Runs inside the worker's venv. The GTK frontend never imports this (it has no
numpy); masks are exchanged as PNG files on disk.
"""
import os
import json
import zipfile
import subprocess
import urllib.request

import numpy as np
from PIL import Image
import onnxruntime as ort

from compute.maskops import _stability, _nms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_CACHE_HOME = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
_CONFIG_HOME = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
MODELS_DIR = os.path.join(_CACHE_HOME, "bg-be-gone", "models")
CONFIG_DIR = os.path.join(_CONFIG_HOME, "bg-be-gone")
CONFIG_PATH = os.path.join(CONFIG_DIR, "seg.json")

_HF = "https://huggingface.co"

# ImageNet normalisation (SAM 2.1 preprocessing).
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

# ---------------------------------------------------------------------------
# Model registry / ladder
# ---------------------------------------------------------------------------
# ``vram_gate`` is the minimum *free* VRAM (MiB) for auto-selection to pick this
# rung. ``large`` is intentionally never auto-selected (manual override only) —
# the auto default aims "high but not the highest", the SAM analogue of the
# app's BiRefNet-General default.
MODELS = {
    "large": dict(
        family="sam2", label="SAM 2.1 Large", vram_gate=8000,
        repo="vietanhdev/segment-anything-2.1-onnx-models",
        zip="sam2.1_hiera_large_20260221.zip", stem="sam2.1_hiera_large"),
    "base_plus": dict(
        family="sam2", label="SAM 2.1 Base+", vram_gate=6000,
        repo="vietanhdev/segment-anything-2.1-onnx-models",
        zip="sam2.1_hiera_base_plus_20260221.zip", stem="sam2.1_hiera_base_plus"),
    "small": dict(
        family="sam2", label="SAM 2.1 Small", vram_gate=4000,
        repo="vietanhdev/segment-anything-2.1-onnx-models",
        zip="sam2.1_hiera_small_20260221.zip", stem="sam2.1_hiera_small"),
    "tiny": dict(
        family="sam2", label="SAM 2.1 Tiny", vram_gate=2500,
        repo="vietanhdev/segment-anything-2.1-onnx-models",
        zip="sam2.1_hiera_tiny_20260221.zip", stem="sam2.1_hiera_tiny"),
    "mobile": dict(
        family="sam1", label="MobileSAM", vram_gate=0,
        repo="Acly/MobileSAM",
        files=["mobile_sam_image_encoder.onnx", "sam_mask_decoder_single.onnx"]),
}

# Order used when stepping DOWN after a failure (best → cheapest).
LADDER = ["large", "base_plus", "small", "tiny", "mobile"]
# Rungs auto-selection is allowed to choose (excludes ``large``).
_AUTO_ORDER = ["base_plus", "small", "tiny"]


class SegError(Exception):
    pass


class ComputeError(SegError):
    """A model ran but produced garbage (e.g. NaN from a broken GPU kernel).
    Retry the same model on CPU rather than dropping to a smaller one."""


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
    cfg = load_config()
    cfg.update(kw)
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f)
        os.replace(tmp, CONFIG_PATH)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# VRAM probing
# ---------------------------------------------------------------------------
def probe_vram():
    """Return ``{"total":MiB, "free":MiB, "gpu":True}`` or ``None``.

    NVIDIA via ``nvidia-smi``; AMD via ``rocm-smi`` or the DRM sysfs nodes. Best
    effort — any failure returns ``None`` and the caller treats the host as CPU.
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6)
        if r.returncode == 0 and r.stdout.strip():
            total, free = (int(x) for x in r.stdout.strip().splitlines()[0].split(","))
            return {"total": total, "free": free, "gpu": True}
    except Exception:
        pass
    try:
        r = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--csv"],
                           capture_output=True, text=True, timeout=6)
        if r.returncode == 0:
            total = used = None
            for line in r.stdout.splitlines():
                cells = line.split(",")
                for i, c in enumerate(cells):
                    if "vram total memory" in c.lower():
                        total = int(cells[i + 1]) // (1024 * 1024)
                    if "vram total used memory" in c.lower():
                        used = int(cells[i + 1]) // (1024 * 1024)
            if total:
                free = total - (used or 0)
                return {"total": total, "free": free, "gpu": True}
    except Exception:
        pass
    # DRM sysfs (AMD)
    try:
        import glob as _glob
        for base in _glob.glob("/sys/class/drm/card*/device"):
            tot = os.path.join(base, "mem_info_vram_total")
            usd = os.path.join(base, "mem_info_vram_used")
            if os.path.exists(tot):
                with open(tot) as f:
                    total = int(f.read()) // (1024 * 1024)
                used = 0
                if os.path.exists(usd):
                    with open(usd) as f:
                        used = int(f.read()) // (1024 * 1024)
                return {"total": total, "free": total - used, "gpu": True}
    except Exception:
        pass
    return None


def rung_for_vram(free_mb, gpu):
    """Auto-select rung from free VRAM, capped at ``base_plus``."""
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
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    with urllib.request.urlopen(url, timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(done, total)
    os.replace(tmp, dst)


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
    with zipfile.ZipFile(zip_p) as z:
        for member in z.namelist():
            if member.endswith(".onnx"):
                data = z.read(member)
                target = enc_p if "encoder" in member else dec_p
                with open(target, "wb") as f:
                    f.write(data)
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

# SAM session
# ---------------------------------------------------------------------------
def preprocess(pil, family):
    """Image -> the encoder's input tensor. Pure: no model, no GPU.

    Split out of `encode` because this is the part a second implementation gets
    silently wrong — a mis-scaled or mis-normalised tensor does not raise, it
    just segments the wrong thing. It is pinned by spec/goldens/model_io.json so
    the TypeScript core can check itself against the same numbers.

    Returns (tensor, meta). `meta` carries the letterbox geometry the decoder
    needs to map points and masks back to original-image pixels.

    The two families want *different things*, and this is the landmine:

      sam2   NCHW float, ImageNet-normalised, letterboxed top-left into a
             1024x1024 canvas (the rest stays zero).
      sam1   HWC float 0..255. The encoder normalises and pads to 1024 itself —
             but it does NOT resize. So the longest side must already be 1024,
             or the content lands in the corner of the padded canvas at the
             wrong scale, and anything over 1024 gets cropped (the pad amount
             goes negative). We resize; we must not normalise or pad.
    """
    W, H = pil.size
    scale = 1024.0 / max(W, H)
    nw, nh = round(W * scale), round(H * scale)
    meta = {"size": (W, H), "scale": scale, "nw": nw, "nh": nh}

    small = pil.resize((nw, nh), Image.BILINEAR)
    if family != "sam2":
        return np.asarray(small, np.float32), meta

    arr = np.asarray(small, np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    buf = np.zeros((1024, 1024, 3), np.float32)     # letterbox: top-left
    buf[:nh, :nw] = arr
    return np.ascontiguousarray(buf.transpose(2, 0, 1)[None]), meta


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
        self._size = None           # (W, H) original
        self._scale = None          # 1024 / max(W, H)
        self._nw = self._nh = None  # resized-to-1024 content dims

    @classmethod
    def load(cls, rung, providers, progress=None):
        spec = MODELS[rung]
        enc_p, dec_p = ensure_model(rung, progress)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        # onnxruntime 1.27's graph optimiser miscompiles a fused op in these
        # SAM ONNX graphs and emits NaN (nondeterministically, on both CPU and
        # CUDA). Disabling optimisation avoids the buggy pass and is measurably
        # faster here too. See the NaN guard in encode() as a backstop.
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        enc = ort.InferenceSession(enc_p, so, providers=providers)
        dec = ort.InferenceSession(dec_p, so, providers=providers)
        return cls(rung, spec["family"], enc, dec)

    def provider(self):
        act = self.encoder.get_providers()
        return act[0] if act else "CPUExecutionProvider"

    # ---- encode ----
    def encode(self, pil):
        pil = pil.convert("RGB")
        inp, meta = preprocess(pil, self.family)
        self._size = meta["size"]
        self._scale = meta["scale"]
        self._nw, self._nh = meta["nw"], meta["nh"]
        if self.family == "sam2":
            he0, he1, emb = self.encoder.run(None, {"image": inp})
            self._emb = (emb, he0, he1)
        else:
            emb = self.encoder.run(None, {self._enc_input: inp})[0]
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
        out = self.decoder.run(None, {
            "image_embed": emb, "high_res_feats_0": he0, "high_res_feats_1": he1,
            "point_coords": pc, "point_labels": pl,
            "mask_input": mi, "has_mask_input": hmi})
        return out[0], out[1]

    def _crop_upscale(self, low):
        """256x256 logits over the padded 1024 input -> original-size bool mask."""
        vh = max(1, round(self._nh / 4.0))
        vw = max(1, round(self._nw / 4.0))
        crop = np.ascontiguousarray(low[:vh, :vw])
        im = Image.fromarray(crop, mode="F").resize(self._size, Image.BILINEAR)
        return np.asarray(im) > 0.0

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
        threading.Event); ``progress(done, total)`` is called periodically."""
        W, H = self._size
        xs = np.linspace(0, self._nw, points_per_side + 2)[1:-1]
        ys = np.linspace(0, self._nh, points_per_side + 2)[1:-1]
        grid = [(float(x), float(y)) for y in ys for x in xs]
        total = len(grid)
        min_area = max(64, int(min_area_frac * W * H))
        records = []
        for i, (x, y) in enumerate(grid):
            if cancel is not None and cancel.is_set():
                break
            lg, score = self._grid_decode(x, y)
            if score < pred_iou_thresh:
                continue
            if _stability(lg) < stability_thresh:
                continue
            mask = self._crop_upscale(lg)
            if np.count_nonzero(mask) < min_area:
                continue
            records.append((score, mask))
            if progress and (i % 8 == 0 or i == total - 1):
                progress(i + 1, total)

        kept = _nms(records, iou_thresh=0.7)
        kept.sort(key=lambda r: np.count_nonzero(r[1]), reverse=True)
        return [m for _, m in kept]


# ---------------------------------------------------------------------------
# Auto-selection with step-down
# ---------------------------------------------------------------------------
def _looks_like_oom(exc):
    s = str(exc).lower()
    return any(k in s for k in ("out of memory", "oom", "cuda", "cublas",
                                "cudnn", "failed to allocate", "hipmalloc",
                                "rocm", "miopen"))


def auto_select_session(providers, gpu, want="auto", probe_image=None,
                        on_step=None, on_progress=None):
    """Pick and load a SAM session, stepping down the ladder on failure.

    ``providers``    onnxruntime provider list (from the worker).
    ``gpu``          whether a GPU execution provider is actually active.
    ``want``         "auto" or an explicit rung name.
    ``probe_image``  a PIL image used to force the first encode (where the GPU
                     actually reserves workspace and OOM/NaN surface).
    ``on_step``      called ``(rung, exc)`` each time an attempt is abandoned.

    Failure handling differs by cause: an out-of-memory error steps down to a
    smaller model (keeping the GPU), while a compute failure (NaN from a broken
    GPU kernel, or any other error) retries the *same* model on CPU before
    stepping down — so a too-new GPU degrades to CPU without needless downloads.

    Returns ``(session, rung, mode)`` where mode is auto|manual|fallback.
    """
    cpu_only = ["CPUExecutionProvider"]
    have_gpu = gpu and any(_provider_name(p) != "CPUExecutionProvider"
                           for p in providers)

    cfg = load_config()
    if want == "auto":
        vram = probe_vram()
        derived = rung_for_vram(vram["free"] if vram else 0, have_gpu and bool(vram))
        # Only honour a *saved* rung when it is cheaper than what VRAM suggests
        # — i.e. a remembered step-down from a prior failure — so we don't
        # re-attempt a known-too-big model every launch. Never let it cap us
        # below the VRAM pick (a manual choice is not persisted, see below).
        saved = cfg.get("rung")
        if saved in LADDER and LADDER.index(saved) > LADDER.index(derived):
            start = saved
        else:
            start = derived
        base_mode, persist = "auto", True
    else:
        start = want if want in LADDER else "base_plus"
        base_mode, persist = "manual", False

    for name in LADDER[LADDER.index(start):]:
        attempts = ([providers, cpu_only] if have_gpu else [cpu_only])
        for prov in attempts:
            try:
                sess = SamSession.load(name, prov, progress=on_progress)
                if probe_image is not None:
                    sess.encode(probe_image)
                if persist:
                    save_config(rung=name)
                first = (name == start and prov is attempts[0])
                return sess, name, (base_mode if first else "fallback")
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
    sess = SamSession.load("mobile", cpu_only, progress=on_progress)
    if probe_image is not None:
        sess.encode(probe_image)
    if persist:
        save_config(rung="mobile")
    return sess, "mobile", "fallback"


# ---------------------------------------------------------------------------
# Object packaging: colors, label map, per-object PNGs, extract compositing
# ---------------------------------------------------------------------------
