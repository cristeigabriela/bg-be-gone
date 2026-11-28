"""Start bg-be-gone: ``pythonw.exe -m bg_be_gone_launch [FILE]``.

Shortcuts and "Open with" name this module, not a path: the shell expands
%VAR% inside a shortcut's arguments, so a path through a folder named
"%TEMP%" would arrive changed. This file sits in the frontend interpreter's
site-packages and holds the paths itself, as Python strings no shell reads.

In the release bundle it works them out from its own location. A source
install writes its checkout and install folders into the two lines below.
"""
import os
import runpy
import sys

APP_DIR = None   # @APP_DIR@
INSTALL = None   # @INSTALL@

if APP_DIR is None:
    # The bundle: <root>\lib\python3.x\site-packages\this file.
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    APP_DIR = os.path.join(_root, "app", "src", "bgbg")

_script = os.path.join(APP_DIR, "plat", "winlaunch.py")
sys.argv = [_script] + ([f"--install={INSTALL}"] if INSTALL else []) + sys.argv[1:]
runpy.run_path(_script, run_name="__main__")
