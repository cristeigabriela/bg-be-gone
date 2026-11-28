"""The parts of the PyTorch engine that need torch (see engines/torch_birefnet.py).

``TorchSession`` mimics a rembg session: ``rembg.remove(img, session=...)``
calls ``predict(img)`` and gets a list with one 'L' mask.
"""
import numpy as np
import torch
from PIL import Image
from torchvision.ops import deform_conv2d

from birefnet import BiRefNet
from engines.torch_birefnet import TorchEngineError

# rembg's BiRefNet preprocessing: ImageNet mean/std at 1024x1024.
_SIZE = (1024, 1024)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class NonFiniteMaskError(TorchEngineError, RuntimeError):
    """A broken GPU kernel shows up as NaN, not an error. Subclasses
    RuntimeError so the worker's GPU-failure fallback catches it."""

    def __init__(self, model, device):
        super().__init__(f"{model}: non-finite mask on {device}")
        self.model, self.device = model, device


def check_device():
    """Run torchvision's deformable convolution once on the GPU.

    BiRefNet's decoder depends on it, and a torchvision built without the
    device's kernels only fails when the op runs, not when it is imported.
    """
    x = torch.zeros(1, 1, 3, 3, device="cuda")
    weight = torch.zeros(1, 1, 3, 3, device="cuda")
    offset = torch.zeros(1, 18, 1, 1, device="cuda")
    deform_conv2d(x, offset, weight)
    torch.cuda.synchronize()


class TorchSession:
    """A rembg-compatible session running one BiRefNet model in PyTorch."""

    def __init__(self, model, checkpoint, backbone, device="cuda", dtype=torch.float32):
        self.model_name = model
        self.device = torch.device(device)
        self.dtype = dtype
        net = BiRefNet(bb=backbone)
        sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
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
        # A perfectly flat prediction has no range to stretch; keep its level.
        pred = (pred - lo) / (hi - lo) if hi > lo else torch.clamp(pred, 0, 1)
        mask = Image.fromarray((pred.cpu().numpy() * 255).astype("uint8"), mode="L")
        return [mask.resize(img.size, Image.Resampling.LANCZOS)]
