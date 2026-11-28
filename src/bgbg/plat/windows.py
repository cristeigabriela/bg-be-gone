"""Windows: %LOCALAPPDATA% paths, the worker's DLL folders, the VRAM probe, and
the frontend's side of starting the worker.

The frontend runs on MSYS2's mingw Python (PyGObject's ABI) and the worker on
a standalone CPython (onnxruntime ships only MSVC wheels). The launcher sets
up the environment for the first one; :func:`worker_env` takes that setup
back out for the second, so the worker starts the same from a source
install, a portable zip or the installer.

Importing this module does no work and needs nothing outside the standard
library, so the worker, the frontend and the install scripts can all use it.
Run as a script, it starts another script with :func:`worker_env` (see
:func:`main`), which is how the install scripts probe a worker.
"""
import glob
import logging
import os
import subprocess
import sys
import sysconfig

# Variables that point an interpreter at another Python's stdlib or packages.
# The bundle's launcher sets PYTHONHOME to the mingw tree, which a CPython
# worker reads as its own and then fails to find `encodings`.
_INTERPRETER_VARS = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONUSERBASE",
    "PYTHONPLATLIBDIR",
    "PYTHONSTARTUP",
    "PYTHONEXECUTABLE",
)

# A PATH folder holding one of these runs on MSYS2's or Cygwin's runtime, or
# on mingw's: the frontend's own bin\, a shared C:\msys64, Git for Windows'
# tools. Their zlib, libstdc++ and friends must not shadow the DLLs the
# worker's MSVC packages load by name.
_POSIX_RUNTIME_DLLS = (
    "msys-2.0.dll",          # MSYS2's usr\bin
    "cygwin1.dll",           # Cygwin
    "libwinpthread-1.dll",   # mingw-w64 builds: ucrt64\bin, mingw64\bin
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def data_dir():
    """Per-user folder for models, settings, logs and caches."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "bg-be-gone")


def models_dir():
    """Where downloaded models are kept."""
    return os.path.join(data_dir(), "models")


def config_dir():
    """Where the app's settings are kept."""
    return os.path.join(data_dir(), "config")


def kernel_cache_dir():
    """Where the GPU stack keeps compiled kernels (MIOpen's, for ROCm PyTorch)."""
    return os.path.join(data_dir(), "miopen")


# ---------------------------------------------------------------------------
# Worker start-up
# ---------------------------------------------------------------------------
def _add_nvidia_dll_dirs():
    """Make the CUDA and cuDNN DLLs of the nvidia-*-cu12 wheels loadable.

    They sit in ``site-packages\\nvidia\\<component>\\bin``. Windows has no
    LD_LIBRARY_PATH to re-exec with, so each folder joins the DLL search path
    instead. site-packages is asked of the interpreter, so this holds for a
    venv and for the release's standalone CPython alike.
    """
    site = sysconfig.get_paths()["purelib"]
    for d in glob.glob(os.path.join(site, "nvidia", "*", "bin")):
        try:
            os.add_dll_directory(d)
        except OSError:
            pass


def prepare_process():
    """Get the worker ready; must run before onnxruntime and rembg are imported."""
    _add_nvidia_dll_dirs()
    # rembg reads U2NET_HOME when it is imported; its default would be
    # %USERPROFILE%\.u2net, outside the app's own folder.
    os.environ.setdefault("U2NET_HOME", os.path.join(data_dir(), "u2net"))


def unusable_providers():
    """The onnxruntime providers that must not run here, as a set of names.

    Windows builds of onnxruntime have no ROCm provider. BGBG_NO_ROCM is still
    honoured, so the kill switch means the same on both systems.
    """
    if os.environ.get("BGBG_NO_ROCM"):
        return {"ROCMExecutionProvider", "MIGraphXExecutionProvider"}
    return set()


def provider_options():
    """Per-provider options for this machine: DirectML runs on the adapter
    the VRAM probe measures (see winvram.main_adapter).

    NOTE: DirectML's device_id is taken to count adapters in DXGI's own
    order. A one-GPU PC can't tell; a laptop with two GPUs is untested.
    """
    try:
        from . import winvram
        adapter = winvram.main_adapter()
    except Exception:  # noqa: BLE001
        adapter = None
    if adapter is None:
        return {}
    return {"DmlExecutionProvider": {"device_id": str(adapter.index)}}


def probe_vram():
    """Return ``{"total":MiB, "free":MiB, "gpu":True}`` or ``None``.

    Any vendor through DXGI and the GPU memory counters, then ``nvidia-smi``,
    both for the adapter DirectML runs on (``winvram.main_adapter()``).
    Best effort: any failure returns ``None`` and the caller treats the host
    as CPU.
    """
    try:
        from . import winvram
        info = winvram.probe()
        if info:
            return {k: info[k] for k in ("total", "free", "gpu")}
    except Exception:  # noqa: BLE001
        pass
    # nvidia-smi's numbers count only for the adapter DirectML runs on, found
    # by name; an AMD main adapter, or two NVIDIA cards of one name, is unknown.
    try:
        from . import winvram
        adapter = winvram.main_adapter()
        if adapter is None:
            return None
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6, creationflags=subprocess_flags())
        if r.returncode != 0:
            return None
        rows = [line.rsplit(",", 2) for line in r.stdout.strip().splitlines()]
        matches = [row for row in rows if len(row) == 3 and row[0].strip() == adapter.name.strip()]
        if len(matches) == 1:
            _, total, free = matches[0]
            return {"total": int(total), "free": int(free), "gpu": True}
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# The frontend's side
# ---------------------------------------------------------------------------
def default_worker_python():
    """The worker interpreter of a default source install.

    The launchers set BGBG_VENV_PYTHON, so this only matters for a bare
    `python app.py`. A worker built as a venv keeps its interpreter under
    Scripts\\, so that one is tried when the standalone one is missing.
    """
    worker = os.path.join(data_dir(), "worker")
    standalone = os.path.join(worker, "python.exe")
    legacy_venv = os.path.join(worker, "Scripts", "python.exe")
    if os.path.exists(standalone) or not os.path.exists(legacy_venv):
        return standalone
    return legacy_venv


def worker_log_path():
    """Where the worker's stderr goes.

    Not GLib's user cache folder: on Windows that is the Internet Explorer
    cache (...\\Microsoft\\Windows\\INetCache).
    """
    return os.path.join(data_dir(), "bg-be-gone-worker.log")


def subprocess_flags():
    """Popen creation flags: no console window, since the frontend runs windowless."""
    return subprocess.CREATE_NO_WINDOW


def _same_dir(a, b):
    # abspath also normalises: a trailing separator, "..", or a missing drive.
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _runs_on_posix_runtime(folder):
    return any(os.path.isfile(os.path.join(folder, dll)) for dll in _POSIX_RUNTIME_DLLS)


def worker_env(environ=None):
    """A copy of `environ` that is safe to start the CPython worker with.

    Drops the variables in _INTERPRETER_VARS, keeps the user's own
    site-packages out, and takes off PATH the frontend's interpreter folder
    and every folder that runs on an MSYS2, Cygwin or mingw runtime.

    Args:
        environ: the environment to start from; os.environ when None.

    Returns:
        A new dict; `environ` is not changed.
    """
    env = dict(os.environ if environ is None else environ)
    for name in _INTERPRETER_VARS:
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"

    frontend_bin = os.path.dirname(os.path.abspath(sys.executable))
    kept = [p for p in env.get("PATH", "").split(os.pathsep)
            if p and not _same_dir(p, frontend_bin) and not _runs_on_posix_runtime(p)]
    env["PATH"] = os.pathsep.join(kept)
    return env


_render_choice = None   # wincompose.Choice, from frontend_starting()


def _start_marker():
    from . import wincompose
    return os.path.join(data_dir(), wincompose.START_MARKER)


def frontend_starting():
    """Called before GTK initialises: render on the GPU through Direct
    Composition, unless the last such start never showed its window."""
    global _render_choice
    from . import wincompose
    _render_choice = wincompose.choose_renderer(_start_marker())


def frontend_started(app_id):
    """Called once GTK is up. Windows has no .desktop file to name the window
    icon, so the taskbar would show Python's.

    The launcher names the install's hicolor theme folder in BGBG_ICON_DIR.
    It is added through GTK's API rather than XDG_DATA_DIRS, which GLib
    would read with %VAR% expanded.
    """
    from gi.repository import Gdk, Gtk
    icons = os.environ.get("BGBG_ICON_DIR")
    if icons and os.path.isdir(icons):
        Gtk.IconTheme.get_for_display(Gdk.Display.get_default()).add_search_path(icons)
    Gtk.Window.set_default_icon_name(app_id)
    if _render_choice is None:
        return
    from . import wincompose
    if _render_choice.renderer is wincompose.Renderer.DCOMP:
        wincompose.watch_gl_windows()   # transparent corners on every surface
    else:
        logging.warning("rendering with Cairo: %s", _render_choice.reason)


def frontend_handed_off():
    """Called when this start passed its files to a running bg-be-gone: it
    ends without a window, and that's a start that went fine."""
    from . import wincompose
    if _render_choice is not None and _render_choice.renderer is wincompose.Renderer.DCOMP:
        wincompose.start_confirmed(_start_marker())


def frontend_window(window):
    """Called with the main window once presented: its first frame shows the
    Direct Composition start went fine."""
    from . import wincompose
    if _render_choice is not None and _render_choice.renderer is wincompose.Renderer.DCOMP:
        wincompose.confirm_on_first_frame(window, _start_marker())


# ---------------------------------------------------------------------------
# As a script
# ---------------------------------------------------------------------------
def main(argv):
    """``python -I windows.py SCRIPT [ARGS...]``: run SCRIPT on this
    interpreter with :func:`worker_env`, and exit with its status.

    The install scripts start their worker probes this way, so a probe gets
    exactly the environment the frontend gives the worker. ``-I`` keeps this
    first interpreter itself clear of PYTHONHOME and friends.
    """
    if len(argv) < 2:
        raise SystemExit(main.__doc__)
    done = subprocess.run([sys.executable] + argv[1:], env=worker_env())
    return done.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv))
