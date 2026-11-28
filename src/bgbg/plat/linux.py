"""Linux: XDG paths, the GPU stack's start-up settings, and VRAM probes.

Owns what the worker must do before onnxruntime is imported (``gpu.env``, the
ROCm device-code check, the loader re-exec), where models and config live, and
how much VRAM the card has. Importing this module does none of that work;
:func:`prepare_process` does.
"""
import os
import sys
import glob
import subprocess


_rocm_unsafe = False       # provider has no code for this card — skip ROCm


def _gfx_arch():
    """The AMD GPU's LLVM target (``"gfx1031"``) via amdkfd, or None.

    The driver packs it as major*10000 + minor*100 + step, with minor and step
    as nibbles in the name, so 100301 is gfx1031 and 90010 is gfx90a.
    """
    for props in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")):
        try:
            with open(props) as f:
                for line in f:
                    if line.startswith("gfx_target_version"):
                        v = int(line.split()[1])
                        if v:
                            return "gfx%d%x%x" % (v // 10000, v // 100 % 100, v % 100)
        except (OSError, ValueError, IndexError):
            continue
    return None


def _rocm_ep_path():
    """Path to the installed ROCm execution provider, or None.

    Located on disk rather than through ``ort.get_available_providers()``,
    because this has to run before onnxruntime is imported.
    """
    venv = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
    found = glob.glob(os.path.join(
        venv, "lib", "python*", "site-packages", "onnxruntime", "capi",
        "libonnxruntime_providers_rocm.so"))
    return found[0] if found else None


def _have_rocm_ep():
    return _rocm_ep_path() is not None


def _rocm_ep_has_arch(arch):
    """True if the ROCm provider carries device code for `arch`.

    hipcc tags each embedded offload bundle with its target triple, so the
    binary is asked directly rather than matched against a list of what some
    wheel is assumed to ship — which is what lets a provider built locally for
    this exact card be recognised as usable.
    """
    path = _rocm_ep_path()
    if not path:
        return False
    needle = ("amdgcn-amd-amdhsa--" + arch).encode()
    try:
        with open(path, "rb") as f:
            overlap = b""
            while True:
                block = f.read(8 << 20)
                if not block:
                    return False
                if needle in overlap + block:
                    return True
                overlap = block[-len(needle):]
    except OSError:
        return False


def _check_rocm_arch():
    """Drop the ROCm provider unless it has device code for this card.

    install.sh settles this at install time; this is the backstop for a venv
    that predates it, a card swapped since, or the AppImage.

    Deliberately no HSA_OVERRIDE_GFX_VERSION fallback. Making a card run a
    sibling's kernels is the usual advice for GPUs ROCm does not support, and it
    is not safe: mismatched kernels decode as garbage on the shader core, page
    fault, time out the graphics ring and force a GPU reset that takes the
    compositor with it. The CPU is slower but correct.
    """
    global _rocm_unsafe
    # Kill switch, for taking ROCm out of the picture without rebuilding.
    if os.environ.get("BGBG_NO_ROCM"):
        _rocm_unsafe = True
        return
    if not _have_rocm_ep():
        return
    arch = _gfx_arch()
    if arch is not None and not _rocm_ep_has_arch(arch):
        _rocm_unsafe = True


def _load_gpu_env():
    """Apply the GPU settings install.sh recorded in ``gpu.env``.

    Tuning the GPU stack reads only at process start, so it has to land before
    onnxruntime is imported. Kept out of the user's global environment on
    purpose — it would affect every other ROCm app too. Anything already set
    wins, so it stays overridable per-launch.
    """
    venv = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
    for path in (os.environ.get("BGBG_GPU_ENV"),
                 os.path.join(os.path.dirname(venv), "gpu.env"),
                 os.path.join(os.environ.get("XDG_DATA_HOME",
                                             os.path.expanduser("~/.local/share")),
                              "bg-be-gone", "gpu.env")):
        if not path:
            continue
        try:
            with open(path) as f:
                lines = f.readlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())
        return


def _ensure_gpu_libs():
    """Put any venv-bundled NVIDIA CUDA/cuDNN libs on LD_LIBRARY_PATH.

    onnxruntime-gpu links libcudart.so.* which ship inside the pip wheels; the
    dynamic loader needs them on the path before import, so we set the env and
    re-exec once if necessary. AMD/ROCm libs come from the system loader path,
    so nothing to do there.
    """
    venv = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
    libdirs = glob.glob(os.path.join(
        venv, "lib", "python*", "site-packages", "nvidia", "*", "lib"))
    if not libdirs:
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = cur.split(os.pathsep) if cur else []
    if all(d in parts for d in libdirs):
        return
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(libdirs + parts)
    os.execv(sys.executable, [sys.executable] + sys.argv)


def prepare_process():
    """Get the GPU stack ready; must run before onnxruntime is imported.

    The order matters. ``gpu.env`` can set ``BGBG_NO_ROCM``, so it is read
    before the device-code check. The loader re-exec comes last, because it
    replaces the process and the new one runs all three again.
    """
    _load_gpu_env()
    _check_rocm_arch()
    _ensure_gpu_libs()


def unusable_providers():
    """The onnxruntime providers that must not run here, as a set of names.

    MIGraphX goes with ROCm: it runs on the same device code.
    """
    if _rocm_unsafe:
        return {"ROCMExecutionProvider", "MIGraphXExecutionProvider"}
    return set()


def provider_options():
    """Per-provider options for this machine: none on Linux."""
    return {}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Functions, not constants: gpu.env can set the XDG variables, and it is only
# read by prepare_process(), after this module is imported.
def models_dir():
    """Where downloaded models are kept."""
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(cache_home, "bg-be-gone", "models")


def config_dir():
    """Where the app's settings are kept."""
    config_home = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(config_home, "bg-be-gone")


def kernel_cache_dir():
    """Where the PyTorch engine's GPU stack keeps compiled kernels (MIOpen's)."""
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(cache_home, "bg-be-gone", "miopen")


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


# ---------------------------------------------------------------------------
# The frontend's side
# ---------------------------------------------------------------------------
def default_worker_python():
    """The worker interpreter install.sh builds; the launcher normally names it."""
    return os.path.expanduser("~/.local/share/bg-be-gone/venv/bin/python")


def worker_log_path():
    """Where the worker's stderr goes: GLib's user cache folder."""
    from gi.repository import GLib   # the frontend's; the worker never asks
    return os.path.join(GLib.get_user_cache_dir(), "bg-be-gone-worker.log")


def subprocess_flags():
    """Popen creation flags: none on Linux."""
    return 0


def worker_env(environ=None):
    """The worker inherits the frontend's environment as it is: None, for Popen."""
    return None


def frontend_started(app_id):
    """Called once GTK is up. The .desktop file names the window icon on Linux."""
