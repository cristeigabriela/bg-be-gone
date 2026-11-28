"""Where the Windows launcher finds GTK, the app and the worker.

Pure path logic, so these run on every system; the real start through a
shortcut is checked on Windows by packaging/test-windows-launch.ps1.
"""
import json

import pytest

from plat import winlaunch

# The folder names a user can give that a shell would rewrite: an accent, a
# space, and a literal %TEMP% that cmd.exe and WScript.Shell.Run would expand.
AWKWARD = ["Programmes de José", "%TEMP%"]


@pytest.fixture
def bundle(tmp_path):
    """An unzipped release bundle under awkward folder names."""
    root = tmp_path.joinpath(*AWKWARD, "BgBeGone")
    (root / "bin").mkdir(parents=True)
    (root / "app" / "src" / "bgbg" / "plat").mkdir(parents=True)
    (root / "versions.json").write_text(json.dumps({"app": "1.2.0"}), encoding="utf-8")
    return root


def paths_for(script, interpreter, *args):
    return winlaunch.layout(["winlaunch.py", *args], script=str(script),
                            interpreter=str(interpreter))


def test_a_bundle_finds_everything_next_to_itself(bundle):
    script = bundle / "app" / "src" / "bgbg" / "plat" / "winlaunch.py"
    paths, _ = paths_for(script, bundle / "bin" / "pythonw.exe")
    assert paths["gtk"] == str(bundle), "the bundle's bin\\.. holds GTK's lib and share"
    assert paths["app"] == str(bundle / "app" / "src" / "bgbg"), \
        "the app is the script's parent folder"
    assert paths["worker_python"] == str(bundle / "worker" / "python.exe"), \
        "the worker ships inside the bundle"
    assert paths["version"] == "1.2.0", "the version comes from the bundle's versions.json"


def test_awkward_folder_names_reach_the_environment_unchanged(bundle):
    script = bundle / "app" / "src" / "bgbg" / "plat" / "winlaunch.py"
    paths, _ = paths_for(script, bundle / "bin" / "pythonw.exe")
    env = winlaunch.environment(paths, {"PATH": "C:\\Windows"})
    for name in ("BGBG_VENV_PYTHON", "BGBG_WORKER", "BGBG_ICON_DIR"):
        assert all(part in env[name] for part in AWKWARD), \
            f"{name} must keep 'José' and the literal '%TEMP%' as they are"


# GLib reads these with g_getenv(), which on Windows expands %VAR% in the value.
GLIB_PATH_VARIABLES = ["GI_TYPELIB_PATH", "GSETTINGS_SCHEMA_DIR", "GDK_PIXBUF_MODULE_FILE",
                       "GDK_PIXBUF_MODULEDIR", "XDG_DATA_DIRS"]


def test_no_path_reaches_glib_through_a_variable_it_would_expand(bundle):
    """Under a folder named %TEMP%, GLib would turn GSETTINGS_SCHEMA_DIR into
    a path inside the real temp folder; MSYS2's GLib finds its own files."""
    script = bundle / "app" / "src" / "bgbg" / "plat" / "winlaunch.py"
    paths, _ = paths_for(script, bundle / "bin" / "pythonw.exe")
    env = winlaunch.environment(paths, {})
    assert not [name for name in GLIB_PATH_VARIABLES if name in env], \
        "the launcher must leave GLib's own path variables unset"


def test_a_source_install_takes_its_worker_from_the_install_folder(tmp_path):
    checkout = tmp_path / "checkout" / "src" / "bgbg"
    gtk = tmp_path / "prefix" / "msys64" / "ucrt64"
    install = tmp_path.joinpath(*AWKWARD, "bg-be-gone")
    install.mkdir(parents=True)
    (install / "bg-be-gone-install.json").write_text(json.dumps({"app_version": "1.3.0"}),
                                                    encoding="utf-8")
    paths, app_argv = paths_for(checkout / "plat" / "winlaunch.py", gtk / "bin" / "pythonw.exe",
                                f"--install={install}", "C:\\photos\\%TEMP% é.png")
    assert paths["worker_python"] == str(install / "worker" / "python.exe"), \
        "a source install's worker is in its install folder, not the checkout"
    assert paths["icons"] == str(install / "share" / "icons"), "the app icon is the install's"
    assert paths["version"] == "1.3.0", "the version comes from the install manifest"
    assert app_argv == ["winlaunch.py", "C:\\photos\\%TEMP% é.png"], \
        "the flag is the launcher's; the file to open reaches the app unchanged"


def test_a_side_by_side_install_starts_as_its_own_instance(tmp_path):
    """install.ps1 -AppId: a test install must not open its window inside the
    usual one, which runs under the app's own single-instance id."""
    install = tmp_path / "bg-be-gone-test"
    install.mkdir()
    (install / "bg-be-gone-install.json").write_text(
        json.dumps({"app_id": "io.github.cristeigabriela.BgBeGone.Test"}), encoding="utf-8")
    paths, _ = paths_for(tmp_path / "src" / "bgbg" / "plat" / "winlaunch.py",
                         tmp_path / "ucrt64" / "bin" / "pythonw.exe", f"--install={install}")
    env = winlaunch.environment(paths, {})
    assert env["BGBG_APP_ID"] == "io.github.cristeigabriela.BgBeGone.Test", \
        "the install's own app id reaches the app"


def test_a_launcher_setting_for_the_worker_wins(bundle):
    script = bundle / "app" / "src" / "bgbg" / "plat" / "winlaunch.py"
    paths, _ = paths_for(script, bundle / "bin" / "pythonw.exe")
    env = winlaunch.environment(paths, {"BGBG_VENV_PYTHON": "D:\\dev\\worker\\python.exe"})
    assert env["BGBG_VENV_PYTHON"] == "D:\\dev\\worker\\python.exe", \
        "a developer or a test can point the app at another worker"
