"""The environment the Windows frontend and the install scripts start the worker with.

The bundle's launcher sets PYTHONHOME to the mingw tree and puts that tree's
bin folder first on PATH, for the GTK frontend. The worker is a different
CPython, which would read both as its own; a shared MSYS2 or Git's tools on
PATH would lend it the wrong zlib or libstdc++. Pure Python, so these run on
every system.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from plat import windows

WINDOWS_PY = Path(windows.__file__)
TIMEOUT_S = 60


@pytest.fixture
def dirs(tmp_path):
    """Folders as a Windows PATH holds them; the runtime ones carry their DLL."""
    made = {}
    for name in ("frontend_bin", "system32", "tools", "other_msys2"):
        (tmp_path / name).mkdir()
        made[name] = str(tmp_path / name)
    (tmp_path / "other_msys2" / "msys-2.0.dll").write_bytes(b"")
    return made


@pytest.fixture
def frontend(monkeypatch, dirs):
    """Pretend this process is the bundle's pythonw.exe."""
    monkeypatch.setattr(sys, "executable", os.path.join(dirs["frontend_bin"], "pythonw.exe"))


def launcher_env(dirs):
    return {
        "PYTHONHOME": dirs["frontend_bin"],
        "PYTHONPATH": os.path.join(dirs["frontend_bin"], "lib"),
        # As a launcher writes it: the frontend's folder first, with a trailing separator.
        "PATH": os.pathsep.join([dirs["frontend_bin"] + os.sep, dirs["system32"],
                                 dirs["other_msys2"], dirs["tools"]]),
        "LOCALAPPDATA": os.path.join(os.sep, "users", "jose", "appdata", "local"),
    }


def test_the_worker_never_inherits_the_frontends_pythonhome(frontend, dirs):
    env = windows.worker_env(launcher_env(dirs))
    assert "PYTHONHOME" not in env, "a CPython worker can't start on the mingw stdlib"
    assert "PYTHONPATH" not in env, "the mingw site-packages must not shadow the worker's"


def test_only_the_frontend_and_posix_runtime_folders_leave_path(frontend, dirs):
    env = windows.worker_env(launcher_env(dirs))
    assert env["PATH"].split(os.pathsep) == [dirs["system32"], dirs["tools"]], \
        "the frontend's folder and a second MSYS2 go; everything else stays, in order"


@pytest.mark.parametrize("dll", ["msys-2.0.dll", "cygwin1.dll", "libwinpthread-1.dll"],
                         ids=["msys2 usr-bin", "cygwin", "mingw bin"])
def test_any_folder_on_a_posix_or_mingw_runtime_leaves_path(frontend, dirs, tmp_path, dll):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / dll).write_bytes(b"")
    env = windows.worker_env({"PATH": os.pathsep.join([str(runtime), dirs["tools"]])})
    assert env["PATH"].split(os.pathsep) == [dirs["tools"]], \
        f"a folder holding {dll} lends the worker DLLs built for another runtime"


def test_the_users_own_site_packages_stay_out(frontend, dirs):
    env = windows.worker_env(launcher_env(dirs))
    assert env["PYTHONNOUSERSITE"] == "1", "a pip --user install must not leak into the worker"
    assert env["LOCALAPPDATA"] == launcher_env(dirs)["LOCALAPPDATA"], \
        "unrelated variables pass through"


def test_the_callers_environment_is_left_alone(frontend, dirs):
    original = launcher_env(dirs)
    windows.worker_env(original)
    assert original == launcher_env(dirs), "worker_env returns a copy; the caller's is unchanged"


def test_the_install_scripts_probe_gets_the_same_environment(dirs):
    """install.ps1 and Enable-AmdRocm.ps1 start their worker probe through
    ``python -I windows.py``, so the probe and the frontend share one policy."""
    env = dict(os.environ, PYTHONHOME=dirs["frontend_bin"],
               PATH=os.pathsep.join([dirs["other_msys2"], os.environ.get("PATH", "")]))
    report = ("import os; print(os.environ.get('PYTHONHOME'));"
              " print(os.environ['PATH'].split(os.pathsep)[0])")
    done = subprocess.run([sys.executable, "-I", str(WINDOWS_PY), "-c", report],
                          capture_output=True, text=True, env=env, timeout=TIMEOUT_S)
    assert done.returncode == 0, done.stderr
    pythonhome, first_on_path = done.stdout.splitlines()
    assert pythonhome == "None", "the probed script must not see the launcher's PYTHONHOME"
    assert first_on_path != dirs["other_msys2"], "a second MSYS2 on PATH must not reach it"


NVIDIA_SMI = Path(__file__).parent / "testdata" / "nvidia-smi"


@pytest.mark.skipif(sys.platform != "win32", reason="winvram reads DXGI")
@pytest.mark.parametrize(
    "adapter, smi, expected",
    [
        # DirectML runs on the AMD card; nvidia-smi only knows the NVIDIA one.
        ("AMD Radeon RX 6800", "one-rtx4060.csv", None),
        # The adapter DirectML uses is the one nvidia-smi reports.
        ("NVIDIA GeForce RTX 4060", "one-rtx4060.csv", {"total": 8188, "free": 7900, "gpu": True}),
        # Two cards by one name: which row is DirectML's can't be told.
        ("NVIDIA GeForce RTX 3080", "two-rtx3080.csv", None),
    ],
    ids=["other-vendor", "same-card", "ambiguous"],
)
def test_the_nvidia_smi_fallback_measures_only_the_adapter_directml_uses(
        monkeypatch, adapter, smi, expected):
    """When DXGI can't say how much is used, nvidia-smi's numbers count only
    for the card DirectML runs on; otherwise the VRAM is unknown."""
    from plat import winvram
    monkeypatch.setattr(winvram, "probe", lambda: None)
    monkeypatch.setattr(winvram, "main_adapter",
                        lambda: winvram.Adapter(0, adapter, 8 << 30, 1, 0))
    output = (NVIDIA_SMI / smi).read_text(encoding="utf-8")
    monkeypatch.setattr(windows.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=output, stderr=""))
    assert windows.probe_vram() == expected, "SAM picks its model by these numbers"
