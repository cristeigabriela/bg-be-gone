"""rembg's BiRefNet models, run natively in PyTorch (ROCm or CUDA).

rembg ships BiRefNet as ONNX files exported from the checkpoints BiRefNet
publishes. This package loads those same checkpoints into the model code they
were exported from (vendored here, inference only), so the masks match
rembg's: on the CPU, BiRefNet-general-lite reproduced rembg's ONNX masks
exactly (mean abs diff 0.0, IoU 1.0).

Why not the ONNX files on the GPU: their deformable convolutions were
exported as gather ops that materialise every 7x7 tap at once (784 MB
tensors), which overflows a 12 GB card on DirectML. PyTorch runs the real op.

``TorchSession`` mimics a rembg session: ``rembg.remove(img, session=...)``
calls ``predict(img)`` and gets a list with one 'L' mask.
"""
import hashlib
import os
import shutil
import tempfile
import urllib.request

import numpy as np
import torch
from PIL import Image

from .birefnet import BiRefNet

_RELEASE = "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/"

# rembg model name -> (checkpoint in BiRefNet's v1 release, backbone, sha256).
# The checkpoint names are the stems of rembg 2.0.76's ONNX file names.
MODELS = {
    "birefnet-general": ("BiRefNet-general-epoch_244.pth", "swin_v1_l",
                         "11341a6a1c12646627e8d28da025bfec8aad027929d377cbe8fd4759636cc77c"),
    "birefnet-general-lite": ("BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth", "swin_v1_t",
                              "6a1e050c6ec2697e5ed268455df544782b023acf8643ab771250979094875ab1"),
    "birefnet-portrait": ("BiRefNet-portrait-epoch_150.pth", "swin_v1_l",
                          "5f942b1d2196fd4c3dfd8e9335d4ea8e7d7541346450c04c0cf4405ed66adfd7"),
    "birefnet-dis": ("BiRefNet-DIS-epoch_590.pth", "swin_v1_l",
                     "31602ac94f0bc994b88e4226077db345a35b32890b1e0c1b9f7cfa2802c5ee13"),
    "birefnet-hrsod": ("BiRefNet-HRSOD_DHU-epoch_115.pth", "swin_v1_l",
                       "8e5141259a566b94ebe83e6306dd2e9ebcaa1e0c2d217311bc676b4d6ed1701d"),
    "birefnet-cod": ("BiRefNet-COD-epoch_125.pth", "swin_v1_l",
                     "f6feaae33dc7d8f32bcb77226547ce3ff7099b772917acebf5955b3299614bb8"),
    "birefnet-massive": ("BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.pth", "swin_v1_l",
                         "36e932fb35a7b96c9dc1c5f9522b7156cdad241fb40e3a82fe8ea6b0adfbd673"),
}

# rembg's BiRefNet preprocessing: ImageNet mean/std at 1024x1024.
_SIZE = (1024, 1024)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class BiRefNetError(Exception):
    """Base for every failure this package raises."""


class ChecksumMismatchError(BiRefNetError, OSError):
    def __init__(self, fname, expected, actual):
        super().__init__(f"{fname}: checksum mismatch after download "
                         f"(expected {expected}, got {actual})")
        self.fname, self.expected, self.actual = fname, expected, actual


class NonFiniteMaskError(BiRefNetError, RuntimeError):
    """A broken GPU kernel shows up as NaN, not an error. Subclasses
    RuntimeError so the worker's GPU-failure fallback catches it."""

    def __init__(self, model, device):
        super().__init__(f"{model}: non-finite mask on {device}")
        self.model, self.device = model, device


def gpu_available():
    """True when PyTorch has a ROCm or CUDA device."""
    return torch.cuda.is_available()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_checkpoint(model, models_dir, progress=None):
    """Path to the verified checkpoint for `model`, downloading it if needed."""
    fname, _, digest = MODELS[model]
    path = os.path.join(models_dir, fname)
    if os.path.exists(path):
        return path
    os.makedirs(models_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=models_dir, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(_RELEASE + fname) as r:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
                done += len(block)
                if progress:
                    progress(done, total)
        actual = _sha256(tmp)
        if actual != digest:
            raise ChecksumMismatchError(fname, digest, actual)
        shutil.move(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


class TorchSession:
    """A rembg-compatible session running one BiRefNet model in PyTorch."""

    def __init__(self, model, models_dir, device="cuda", dtype=torch.float16, progress=None):
        fname, bb, _ = MODELS[model]
        self.model_name = model
        self.device = torch.device(device)
        self.dtype = dtype
        net = BiRefNet(bb=bb)
        sd = torch.load(ensure_checkpoint(model, models_dir, progress),
                        map_location="cpu", weights_only=True)
        # Checkpoints saved from torch.compile carry this prefix (BiRefNet's
        # utils.check_state_dict strips it the same way).
        sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
        net.load_state_dict(sd, strict=True)
        self.net = net.eval().to(self.device, dtype)
        self._mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)

    def providers(self):
        return ["TorchROCm" if torch.version.hip else "TorchCUDA"]

    @torch.inference_mode()
    def predict(self, img, *args, **kwargs):
        # Same steps as rembg's BaseSession.normalize + BiRefNetSessionGeneral
        # .predict, in float32 on the device instead of float64 on the CPU.
        small = img.convert("RGB").resize(_SIZE, Image.Resampling.LANCZOS)
        x = torch.from_numpy(np.asarray(small)).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).float()
        x = x / torch.clamp(x.max(), min=1e-6)
        x = ((x - self._mean) / self._std).to(self.dtype)
        pred = torch.sigmoid(self.net(x)[-1].float())[0, 0]
        if not torch.isfinite(pred).all():
            # Fail loudly, so the worker falls back instead of returning garbage.
            raise NonFiniteMaskError(self.model_name, self.device)
        lo, hi = pred.min(), pred.max()
        pred = ((pred - lo) / (hi - lo)).cpu().numpy()
        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        return [mask.resize(img.size, Image.Resampling.LANCZOS)]
