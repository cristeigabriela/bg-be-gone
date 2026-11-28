#!/usr/bin/env bash
# Collect a redistributable GTK4 + libadwaita + PyGObject bundle for bg-be-gone
# from an MSYS2/UCRT64 install into dist/BgBeGone (the frontend half of the
# Windows artifact). Run from an MSYS2 UCRT64 shell, on the pinned packages
# install-windows-gtk.ps1 -BuildTools installs. The worker is added
# separately by build-windows-worker.ps1, and the launch module by
# package-windows.ps1.
#
# Env: ROOT (repo root, default: parent of this script).
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
P=/ucrt64
OUT="$ROOT/dist/BgBeGone"
APP_ICON=io.github.cristeigabriela.BgBeGone.png

PYLIB="$(basename "$(ls -d "$P"/lib/python3.*/ | head -1)")"   # e.g. python3.12
echo "==> Bundling GTK4 + libadwaita ($PYLIB) -> $OUT"

rm -rf "$OUT"
mkdir -p "$OUT/bin" "$OUT/lib" "$OUT/share/glib-2.0" "$OUT/share/icons" \
         "$OUT/app"

# --- DLL closure ------------------------------------------------------------
# Seed with the interpreter plus every library that is dlopened through a
# typelib or a pixbuf loader: ntldd can't find those from the exe alone.
# Wildcards cover the soname suffixes (libadwaita-1-0.dll, libgtk-4-1.dll),
# so a package bump doesn't silently drop a dlopened library.
seeds=(
  "$P"/bin/pythonw.exe "$P"/bin/python.exe "$P"/bin/libpython3.*.dll
  "$P"/bin/libgtk-4*.dll "$P"/bin/libadwaita-1*.dll "$P"/bin/librsvg-2*.dll
  "$P"/bin/libgirepository-*.dll
  "$P"/lib/gdk-pixbuf-2.0/2.10.0/loaders/*.dll
  "$P"/lib/"$PYLIB"/site-packages/gi/*.pyd
  "$P"/lib/"$PYLIB"/site-packages/cairo/*.pyd
)

# Resolve recursive deps, keep only UCRT64 DLLs, copy into bin/.
{
  for s in "${seeds[@]}"; do
    [ -e "$s" ] || continue
    ntldd -R "$s" 2>/dev/null | sed -n 's/.*=> \(.*\) (0x[0-9a-fA-F]*).*/\1/p'
  done
} | grep -i 'ucrt64' | sort -u | while read -r dll; do
  u="$(cygpath -u "$dll" 2>/dev/null || echo "$dll")"
  if [ -f "$u" ]; then cp -u "$u" "$OUT/bin/"; fi
done

# The seeds themselves (exe + dlopened libs) and the GLib spawn/dbus helpers.
for s in "${seeds[@]}"; do
  if [ -e "$s" ]; then cp -u "$s" "$OUT/bin/"; fi
done
cp -u "$P"/bin/gspawn-win64-helper*.exe "$OUT/bin/" 2>/dev/null || true
cp -u "$P"/bin/gdbus.exe "$OUT/bin/" 2>/dev/null || true

# --- Python stdlib + gi/cairo (one copy gets stdlib and site-packages) ------
cp -r "$P/lib/$PYLIB" "$OUT/lib/$PYLIB"
find "$OUT/lib/$PYLIB" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$OUT/lib/$PYLIB"/{test,idlelib,turtledemo,tkinter} 2>/dev/null || true

# --- Typelibs (PyGObject resolves these at import), pixbuf loaders ----------
# MSYS2's loaders.cache names each loader relative to the GTK root, so it
# works unchanged wherever the bundle is unpacked.
cp -r "$P/lib/girepository-1.0" "$OUT/lib/girepository-1.0"
cp -r "$P/lib/gdk-pixbuf-2.0"   "$OUT/lib/gdk-pixbuf-2.0"

# --- Data: schemas, icon themes, mime ---------------------------------------
cp -r "$P/share/glib-2.0/schemas" "$OUT/share/glib-2.0/schemas"
for t in Adwaita AdwaitaLegacy hicolor; do
  [ -d "$P/share/icons/$t" ] && cp -r "$P/share/icons/$t" "$OUT/share/icons/$t"
done
[ -d "$P/share/mime" ] && cp -r "$P/share/mime" "$OUT/share/mime" || true

# The app's own icon, where the launcher points GTK's icon theme.
mkdir -p "$OUT/share/icons/hicolor/512x512/apps"
cp "$ROOT/data/$APP_ICON" "$OUT/share/icons/hicolor/512x512/apps/$APP_ICON"

# Compiled schemas are mandatory: libadwaita reads org.gtk.Settings.* at start.
"$P/bin/glib-compile-schemas.exe" "$OUT/share/glib-2.0/schemas"
for t in Adwaita AdwaitaLegacy hicolor; do
  [ -d "$OUT/share/icons/$t" ] && \
    "$P/bin/gtk4-update-icon-cache.exe" -qtf "$OUT/share/icons/$t" || true
done

# --- App source and the console launcher ------------------------------------
cp -r "$ROOT/src" "$OUT/app/src"
find "$OUT/app" -type d -name __pycache__ -prune -exec rm -rf {} +
# CRLF: cmd.exe reads LF-only batch files too, but mis-parses some lines.
sed 's/\r\?$/\r/' "$ROOT/packaging/BgBeGone.cmd" > "$OUT/BgBeGone.cmd"

# Bytecode now, so a first start in Program Files has nothing to write, and
# the installer's file list covers every file the app has.
"$OUT/bin/python.exe" -I -m compileall -q "$OUT/lib/$PYLIB" "$OUT/app/src"

# What the frontend runs on, for bug reports and the release notes.
pacman -Q "${MINGW_PACKAGE_PREFIX:-mingw-w64-ucrt-x86_64}"-{gtk4,libadwaita,python,python-gobject} \
  > "$OUT/gtk-versions.txt"

echo "==> GTK bundle ready: $OUT"
