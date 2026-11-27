@echo off
rem bg-be-gone portable launcher. All paths are bundle-relative (%~dp0), so the
rem unzipped folder can live anywhere. Runs the GTK frontend on the bundled
rem mingw pythonw.exe; the frontend spawns the ML worker from BGBG_VENV_PYTHON.
setlocal
set "APPROOT=%~dp0"
set "PATH=%APPROOT%bin;%PATH%"
set "PYTHONHOME=%APPROOT%"
set "GI_TYPELIB_PATH=%APPROOT%lib\girepository-1.0"
set "GDK_PIXBUF_MODULEDIR=%APPROOT%lib\gdk-pixbuf-2.0\2.10.0\loaders"
set "GDK_PIXBUF_MODULE_FILE=%APPROOT%lib\gdk-pixbuf-2.0\2.10.0\loaders.cache"
set "GSETTINGS_SCHEMA_DIR=%APPROOT%share\glib-2.0\schemas"
set "XDG_DATA_DIRS=%APPROOT%share"
set "BGBG_VENV_PYTHON=%APPROOT%worker\Scripts\python.exe"
set "BGBG_WORKER=%APPROOT%app\src\bgbg\worker.py"
set "BGBG_VERSION=__VERSION__"
start "" "%APPROOT%bin\pythonw.exe" "%APPROOT%app\src\bgbg\app.py" %*
