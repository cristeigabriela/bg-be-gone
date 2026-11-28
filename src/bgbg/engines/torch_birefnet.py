"""The PyTorch engine: rembg's BiRefNet models, run natively from the
checkpoints rembg's ONNX files were exported from.

rembg ships BiRefNet as ONNX exports whose deformable convolutions became
gather ops that materialise every 7x7 tap at once (784 MB tensors), which
overflows a 12 GB card on DirectML. PyTorch runs the real op. On the CPU,
BiRefNet-general-lite reproduced rembg's ONNX masks exactly (mean abs diff
0.0, IoU 1.0).

This module imports no torch: :class:`TorchEngine` imports it on a thread of
its own, so the worker answers at once while PyTorch loads (about 3 s warm on
Windows, up to 40 s right after install while antivirus scans its DLLs). The
code that needs torch is in engines/torch_session.py. The worker decides
when to use this engine (engines/selection.py) and falls back to ONNX
Runtime when it can't start or load.
"""
import enum
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import threading
import urllib.request

import plat
from engines import EngineError

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

# Variables that stop this engine before torch is imported. BGBG_NO_ROCM
# counts too: on Windows the only GPU build of torch we install is ROCm's.
KILL_SWITCHES = ("BGBG_NO_TORCH", "BGBG_NO_ROCM")


class TorchEngineError(EngineError):
    """Base for every failure of the PyTorch engine."""


class ChecksumMismatchError(TorchEngineError, OSError):
    def __init__(self, fname, expected, actual):
        super().__init__(f"{fname}: checksum mismatch (expected {expected}, got {actual})")
        self.fname, self.expected, self.actual = fname, expected, actual


class EngineState(enum.Enum):
    OFF = "off"             # not asked for, switched off, or not installed
    STARTING = "starting"   # importing torch on its thread
    READY = "ready"
    FAILED = "failed"       # torch is there but can't run BiRefNet on a GPU


def installed():
    """True when torch is in this interpreter, judged without importing it."""
    return importlib.util.find_spec("torch") is not None


def switched_off(environ=os.environ):
    """The kill switch that is set, or None."""
    return next((name for name in KILL_SWITCHES if environ.get(name)), None)


def checkpoint_dir():
    return os.path.join(plat.models_dir(), "birefnet")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _verified(path, digest):
    """True when `path` has `digest`, re-hashing only when the file changed.

    Hashing a 900 MB checkpoint takes seconds, so a sidecar records the size
    and modification time it was last checked at, with the digest that matched.
    """
    sidecar = path + ".verified"
    st = os.stat(path)
    stamp = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
    try:
        with open(sidecar, encoding="utf-8") as f:
            # NOTE: on purpose, an unchanged size and time skip the hash. A file
            # damaged in place with both kept still fails torch.load, which
            # falls back to ONNX Runtime.
            if json.load(f) == stamp:
                return True
    except (OSError, ValueError):
        pass
    if _sha256(path) != digest:
        return False
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(stamp, f)
    return True


def ensure_checkpoint(model, models_dir=None, progress=None):
    """Path to the verified checkpoint for `model`, downloading it if needed.

    A cached file is checked too: one that no longer matches its SHA-256
    (cut short, or changed on disk) is downloaded again.
    """
    fname, _, digest = MODELS[model]
    models_dir = models_dir or checkpoint_dir()
    path = os.path.join(models_dir, fname)
    if os.path.exists(path) and _verified(path, digest):
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
        _verified(path, digest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


class TorchEngine:
    """PyTorch, loaded on a thread of its own, and the sessions it builds.

    ``start()`` returns at once; ``wait()`` blocks until the start-up is over.
    A failure is kept in ``reason`` for the worker to tell the user.
    """

    def __init__(self):
        self.state = EngineState.OFF
        self.reason = None
        self.device = None          # e.g. "AMD (ROCm)" once ready
        self._done = threading.Event()
        self._done.set()
        self._session_module = None
        self._torch = None

    def start(self):
        if self.state in (EngineState.STARTING, EngineState.READY):
            return
        # MIOpen (ROCm's convolution library) compiles kernels on first use;
        # one folder keeps them between launches. Before torch is imported.
        cache = plat.kernel_cache_dir()
        os.environ.setdefault("MIOPEN_USER_DB_PATH", os.path.join(cache, "db"))
        os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", os.path.join(cache, "cache"))
        self.state, self.reason = EngineState.STARTING, None
        self._done.clear()
        try:
            threading.Thread(target=self._start_up, name="torch-init", daemon=True).start()
        except RuntimeError as err:
            # Waiters must still get an answer: ONNX Runtime, and why.
            self.state, self.reason = EngineState.FAILED, f"the PyTorch loader didn't start ({err})"
            self._done.set()

    def _start_up(self):
        try:
            import torch
            # The GPU first: without one, nothing else here can run.
            if not torch.cuda.is_available():
                raise TorchEngineError(f"PyTorch {torch.__version__} sees no ROCm or CUDA GPU")
            from engines import torch_session
            # BiRefNet's decoder needs torchvision's compiled deformable
            # convolution on the device; run it once on a tiny input.
            torch_session.check_device()
            # No per-shape kernel search: it re-runs on every launch, and on
            # RDNA2 MIOpen's exhaustive search is known to hang.
            torch.backends.cudnn.benchmark = False
            self._torch, self._session_module = torch, torch_session
            self.device = "AMD (ROCm)" if torch.version.hip else "NVIDIA (CUDA)"
            self.state = EngineState.READY
        except Exception as err:  # noqa: BLE001  (any failure means: use ONNX Runtime)
            self.state, self.reason = EngineState.FAILED, f"{type(err).__name__}: {err}"
        finally:
            self._done.set()

    def wait(self, timeout=None):
        self._done.wait(timeout)
        return self.state

    def status(self):
        return {"state": self.state.value, "reason": self.reason, "device": self.device}

    def new_session(self, model, progress=None):
        """A rembg-compatible session for `model`; raises when it can't load."""
        if self.state is not EngineState.READY:
            raise TorchEngineError(f"the PyTorch engine is {self.state.value}")
        dtype = "float16" if os.environ.get("BGBG_TORCH_DTYPE") == "fp16" else "float32"
        fname, backbone, _ = MODELS[model]
        return self._session_module.TorchSession(
            model, ensure_checkpoint(model, progress=progress), backbone,
            dtype=getattr(self._torch, dtype))

    def release_memory(self):
        """Hand PyTorch's cached VRAM back, e.g. before a CPU retry."""
        if self._torch is not None:
            self._torch.cuda.empty_cache()
