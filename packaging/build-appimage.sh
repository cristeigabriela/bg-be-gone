#!/usr/bin/env bash
# Build an AppImage for bg-be-gone (variant: cpu | cuda | rocm).
#
# Built on Arch Linux (locally and in the CI arch container) so the bundled
# GTK4 + libadwaita match a normal Arch install (same look as running from
# source). One relocatable Python 3.12 (via uv) serves both the GTK frontend
# and the worker: PyGObject binds the system GTK/libadwaita, and onnxruntime
# (which has no 3.14 wheels) gets a version it supports.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
WORK="${WORK:-$ROOT/build}"
APPDIR="$WORK/AppDir"
ARCH="$(uname -m)"
VERSION="${VERSION:-1.1.0}"; export VERSION
VARIANT="${VARIANT:-cpu}"   # cpu | cuda | rocm | seg | seg-cuda | seg-rocm
ID=io.github.cristeigabriela.BgBeGone
export APPIMAGE_EXTRACT_AND_RUN=1   # so the tool AppImages run without FUSE

echo "==> bg-be-gone $VERSION ($VARIANT) on $ARCH"
rm -rf "$APPDIR" "$WORK/venv"
mkdir -p "$WORK" "$APPDIR/usr/bin" "$APPDIR/usr/lib/bg-be-gone"

# --- unified Python 3.12 venv: just PyGObject for now (binds system GTK) ---
# The ML stack is added AFTER linuxdeploy so linuxdeploy never has to resolve
# the self-contained scipy/CUDA/ROCm wheels.
uv venv --python 3.12 "$WORK/venv" >/dev/null
VPY="$WORK/venv/bin/python"
uv pip install --python "$VPY" --quiet pygobject pycairo

# --- bundle the relocatable interpreter + the venv's site-packages --------
PYROOT="$(dirname "$(dirname "$(uv python find 3.12)")")"
cp -a "$PYROOT/." "$APPDIR/usr/"
# Drop unused stdlib extension modules so linuxdeploy doesn't have to resolve
# their system libraries (Tcl/Tk, libcrypt, ncurses, readline, gdbm, ...), which
# a minimal build environment may not have. None are used by the app or worker.
_dyn="$APPDIR/usr/lib/python3.12/lib-dynload"
for m in _tkinter _crypt readline _curses _curses_panel nis ossaudiodev spwd \
         _dbm _gdbm audioop; do
  rm -f "$_dyn/$m".*.so
done
rm -rf "$APPDIR"/usr/lib/python3.12/{tkinter,idlelib,turtledemo,test,curses,dbm}
SITE_SRC="$WORK/venv/lib/python3.12/site-packages"
SITE_DST="$APPDIR/usr/lib/python3.12/site-packages"
mkdir -p "$SITE_DST"
cp -a "$SITE_SRC/." "$SITE_DST/"

# --- app source ----------------------------------------------------------
cp -r "$ROOT/src" "$APPDIR/usr/lib/bg-be-gone/src"

# --- desktop / icon / metadata -------------------------------------------
install -Dm644 "$ROOT/data/$ID.png" "$APPDIR/usr/share/icons/hicolor/512x512/apps/$ID.png"
install -Dm644 "$ROOT/data/$ID.desktop" "$APPDIR/usr/share/applications/$ID.desktop"
install -Dm644 "$ROOT/data/$ID.metainfo.xml" "$APPDIR/usr/share/metainfo/$ID.metainfo.xml"
cp "$ROOT/data/$ID.png" "$APPDIR/$ID.png"
cp "$ROOT/data/$ID.desktop" "$APPDIR/$ID.desktop"

# --- AppRun ---------------------------------------------------------------
cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"
export PYTHONHOME="$HERE/usr"
export BGBG_VENV_PYTHON="$HERE/usr/bin/python3"
export BGBG_WORKER="$HERE/usr/lib/bg-be-gone/src/bgbg/compute/service.py"
for hook in "$HERE"/apprun-hooks/*.sh; do [ -f "$hook" ] && . "$hook"; done
# The gtk hook forces GTK_THEME="" which flattens libadwaita styling; drop it so
# libadwaita applies its stylesheet and reads the host colour scheme/accent.
unset GTK_THEME
exec "$HERE/usr/bin/python3" "$HERE/usr/lib/bg-be-gone/src/bgbg/app.py" "$@"
EOF
chmod +x "$APPDIR/AppRun"
sed -i "2i export BGBG_VERSION=\"$VERSION\"" "$APPDIR/AppRun"

# --- bundle GTK4 + libadwaita (from Arch) and package --------------------
cd "$WORK"
fetch() { [ -f "$2" ] || wget -qO "$2" "$1"; chmod +x "$2"; }
fetch "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-$ARCH.AppImage" linuxdeploy
fetch "https://raw.githubusercontent.com/linuxdeploy/linuxdeploy-plugin-gtk/master/linuxdeploy-plugin-gtk.sh" linuxdeploy-plugin-gtk.sh
fetch "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$ARCH.AppImage" appimagetool
# linuxdeploy-plugin-gtk assumes a Debian layout; on Arch several module dirs
# (gtk-4.0 modules, gdk-pixbuf loaders) don't exist because they are built into
# the libraries. Make copy_lib_tree still create the mirror dir (so later cache
# writes have a parent) but skip copying a source dir that isn't there.
# shellcheck disable=SC2016  # these $vars must stay literal for the plugin
sed -i 's#\(mkdir -p "${dst::-1}${elem/$LD_GTK_LIBRARY_PATH//usr/lib}"\)#\1; [ -e "$elem" ] || continue#' \
  linuxdeploy-plugin-gtk.sh

export DEPLOY_GTK_VERSION=4
export NO_STRIP=1   # linuxdeploy's bundled strip can't parse Arch's .relr.dyn
# libadwaita is dlopened via the typelib, so linuxdeploy/the gtk plugin don't
# see it as a dependency — force it in (and let linuxdeploy pull its deps).
ADW_LIB="$(find /usr/lib -maxdepth 1 -name 'libadwaita-1.so.0' | head -1)"
./linuxdeploy --appdir "$APPDIR" --plugin gtk \
  --executable "$APPDIR/usr/bin/python3" \
  --desktop-file "$APPDIR/$ID.desktop" \
  --icon-file "$ROOT/data/$ID.png" \
  --library "$ADW_LIB" \
  --exclude-library 'libtcl*' --exclude-library 'libtk*'

# --- add rembg + onnxruntime (variant) after GTK is bundled --------------
# Install with the 3.12 venv python for wheel tags, into the bundled
# site-packages, so linuxdeploy never sees these self-contained wheels.
# seg* variants ship the lean Segment Anything stack (onnxruntime + numpy +
# pillow) with no rembg/BiRefNet — a much smaller AppImage. SAM weights are not
# bundled (downloaded on first use), keeping the image small and the Apache
# weights out of the MIT artifact.
case "$VARIANT" in
  cuda)
    uv pip install --python "$VPY" --target "$SITE_DST" --quiet "rembg[gpu]" \
      "numba>=0.60" "llvmlite>=0.43" nvidia-cuda-runtime nvidia-cublas \
      nvidia-cufft nvidia-curand nvidia-cudnn-cu13 ;;
  rocm)
    uv pip install --python "$VPY" --target "$SITE_DST" --quiet "rembg[cpu]" \
      "numba>=0.60" "llvmlite>=0.43"
    rm -rf "$SITE_DST/onnxruntime" "$SITE_DST"/onnxruntime-[0-9]*.dist-info
    uv pip install --python "$VPY" --target "$SITE_DST" --quiet onnxruntime-rocm ;;
  seg-cuda)
    uv pip install --python "$VPY" --target "$SITE_DST" --quiet \
      onnxruntime-gpu numpy pillow nvidia-cuda-runtime nvidia-cublas \
      nvidia-cufft nvidia-curand nvidia-cudnn-cu13 ;;
  seg-rocm)
    uv pip install --python "$VPY" --target "$SITE_DST" --quiet numpy pillow onnxruntime-rocm ;;
  seg)
    uv pip install --python "$VPY" --target "$SITE_DST" --quiet onnxruntime numpy pillow ;;
  *)
    uv pip install --python "$VPY" --target "$SITE_DST" --quiet "rembg[cpu]" \
      "numba>=0.60" "llvmlite>=0.43" ;;
esac

OUT="bg-be-gone-${VERSION}-${VARIANT}-${ARCH}.AppImage"
./appimagetool "$APPDIR" "$OUT"
mkdir -p "$ROOT/dist"
mv "$OUT" "$ROOT/dist/"
echo "==> $ROOT/dist/$OUT"
