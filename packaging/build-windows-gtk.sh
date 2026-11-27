#!/usr/bin/env bash
# Collect a redistributable GTK4 + libadwaita + PyGObject bundle for bg-be-gone
# from an MSYS2/UCRT64 install into dist/BgBeGone (the frontend half of the
# Windows artifact). Run from an MSYS2 UCRT64 shell. The worker venv is added
# separately by build-windows-worker.ps1.
#
# Env: ROOT (repo root, default: parent of this script), VERSION (default 0.0.0).
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VERSION="${VERSION:-0.0.0}"
P=/ucrt64
OUT="$ROOT/dist/BgBeGone"

PYLIB="$(basename "$(ls -d "$P"/lib/python3.*/ | head -1)")"   # e.g. python3.12
echo "==> Bundling GTK4 + libadwaita ($PYLIB) -> $OUT"

rm -rf "$OUT"
mkdir -p "$OUT/bin" "$OUT/lib" "$OUT/share/glib-2.0" "$OUT/share/icons" \
         "$OUT/app"

# --- DLL closure ------------------------------------------------------------
# Seed with the interpreter plus every library that is dlopened through a
# typelib / pixbuf loader (ntldd can't discover those from the exe alone —
# same reason the Linux build force-injects libadwaita).
# Use wildcards for the version/soname suffixes (e.g. libadwaita-1-0.dll,
# libgtk-4-1.dll) so a package bump doesn't silently drop a dlopened library.
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

# --- Typelibs (KEEP — PyGObject resolves these at import), loaders ----------
cp -r "$P/lib/girepository-1.0" "$OUT/lib/girepository-1.0"
cp -r "$P/lib/gdk-pixbuf-2.0"   "$OUT/lib/gdk-pixbuf-2.0"

# --- Data: schemas, icon themes, mime ---------------------------------------
cp -r "$P/share/glib-2.0/schemas" "$OUT/share/glib-2.0/schemas"
for t in Adwaita AdwaitaLegacy hicolor; do
  [ -d "$P/share/icons/$t" ] && cp -r "$P/share/icons/$t" "$OUT/share/icons/$t"
done
[ -d "$P/share/mime" ] && cp -r "$P/share/mime" "$OUT/share/mime" || true

# --- Regenerate caches with bundle-relative paths ---------------------------
loaders="$OUT/lib/gdk-pixbuf-2.0/2.10.0/loaders"
cache="$OUT/lib/gdk-pixbuf-2.0/2.10.0/loaders.cache"
GDK_PIXBUF_MODULEDIR="$loaders" "$P/bin/gdk-pixbuf-query-loaders.exe" > "$cache"
# query-loaders bakes in absolute build-machine paths; rewrite each to a bare
# filename so it resolves relative to GDK_PIXBUF_MODULEDIR (set by the launcher)
# on whatever machine unzips the bundle.
sed -i 's#"[^"]*/loaders/\([^"/]*\.dll\)"#"\1"#' "$cache"
# Compiled schemas are MANDATORY — libadwaita reads org.gtk.Settings.* at start.
"$P/bin/glib-compile-schemas.exe" "$OUT/share/glib-2.0/schemas"
for t in Adwaita AdwaitaLegacy hicolor; do
  [ -d "$OUT/share/icons/$t" ] && \
    "$P/bin/gtk4-update-icon-cache.exe" -qtf "$OUT/share/icons/$t" || true
done

# --- App source + launcher --------------------------------------------------
cp -r "$ROOT/src" "$OUT/app/src"
sed "s/__VERSION__/$VERSION/" "$ROOT/packaging/BgBeGone.cmd" > "$OUT/BgBeGone.cmd"
cp "$ROOT/packaging/BgBeGone.vbs" "$OUT/BgBeGone.vbs"

echo "==> GTK bundle ready: $OUT"
