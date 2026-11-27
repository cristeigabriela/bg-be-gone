"""Start bg-be-gone on Windows.

Shortcuts, "Open with" and Setup all run MSYS2's windowless interpreter on
this file::

    <gtk>\\bin\\pythonw.exe <app>\\plat\\winlaunch.py [--install=<folder>] [FILE]

It tells the app where its worker and icon are, then runs the app in this
same process. Every path comes from where this file and the interpreter are,
so nothing is expanded or re-quoted on the way: a folder named "José" or
"%TEMP%" reaches the app as it is.

GTK itself gets no paths from here. MSYS2's GLib, GTK and gdk-pixbuf find
their typelibs, schemas, loaders and data next to their own DLLs, and on
Windows GLib's g_getenv() expands %VAR% inside a variable's value, so a
GSETTINGS_SCHEMA_DIR under a "%TEMP%" folder would point somewhere else.

- The release bundle needs no argument. The interpreter's bin\\ sits in the
  bundle, the app in app\\src\\bgbg, the worker in worker\\.
- A source install runs the checkout's copy and names its install folder,
  where the worker and the app icon live; install.ps1 records the rest in
  that folder's bg-be-gone-install.json.
"""
import json
import os
import sys

_INSTALL_FLAG = "--install="
_MANIFEST = "bg-be-gone-install.json"


class LaunchError(Exception):
    """The install this launcher was pointed at can't be started."""


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as err:
        raise LaunchError(f"can't read {path}: {err}") from err


def layout(argv, script=__file__, interpreter=sys.executable):
    """Where everything is, and the arguments left for the app.

    Args:
        argv: this process's arguments, including any ``--install=`` flag.
        script: this file; its folder's parent is the app's source folder.
        interpreter: the frontend's pythonw.exe; its folder's parent is
            GTK's prefix (MSYS2's ucrt64, or the bundle).

    Returns:
        ``(paths, app_argv)``: a dict of the places below, and argv without
        the flag.
    """
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(script)))
    gtk_root = os.path.dirname(os.path.dirname(os.path.abspath(interpreter)))
    install = None
    app_argv = [argv[0]]
    for arg in argv[1:]:
        if arg.startswith(_INSTALL_FLAG) and install is None:
            install = arg[len(_INSTALL_FLAG):]
        else:
            app_argv.append(arg)

    app_id = None
    if install is None:            # the release bundle: one folder holds it all
        root = gtk_root
        version = _read_json(os.path.join(root, "versions.json")).get("app")
    else:
        root = install
        manifest = _read_json(os.path.join(root, _MANIFEST))
        version = manifest.get("app_version")
        # install.ps1 -AppId: a second install that runs beside the usual one.
        app_id = manifest.get("app_id")
    paths = {
        "app": app_dir,
        "gtk": gtk_root,
        "worker_python": os.path.join(root, "worker", "python.exe"),
        # The hicolor theme folder that holds the app icon.
        "icons": os.path.join(root, "share", "icons"),
        "version": version,
        "app_id": app_id,
    }
    return paths, app_argv


def environment(paths, environ):
    """The variables to set before the app starts, as a dict.

    Only variables Python code reads, plus PATH for the helper programs GLib
    starts; GLib would expand a '%' in anything else (see the module doc).
    """
    env = {
        "PATH": os.pathsep.join([os.path.join(paths["gtk"], "bin"), environ.get("PATH", "")]),
        "BGBG_WORKER": os.path.join(paths["app"], "worker.py"),
        "BGBG_ICON_DIR": paths["icons"],
        # A launcher setting wins, so a test or a developer can point elsewhere.
        "BGBG_VENV_PYTHON": environ.get("BGBG_VENV_PYTHON") or paths["worker_python"],
    }
    if paths["version"] and not environ.get("BGBG_VERSION"):
        env["BGBG_VERSION"] = paths["version"]
    if paths["app_id"] and not environ.get("BGBG_APP_ID"):
        env["BGBG_APP_ID"] = paths["app_id"]
    return env


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    paths, app_argv = layout(argv)
    os.environ.update(environment(paths, os.environ))
    # The app's folder instead of this one, so `import plat` finds the package
    # and nothing imports windows.py by its bare name.
    sys.path[0] = paths["app"]
    sys.argv = app_argv
    import app
    return app.main()


if __name__ == "__main__":
    sys.exit(main())
