#!/usr/bin/env bash
# Install bg-be-gone for the current user: build the worker virtualenv,
# register the desktop entry and icon. Re-runnable (idempotent).
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}/bg-be-gone"
VENV="$DATA/venv"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/512x512/apps"
BINDIR="$HOME/.local/bin"

# Worker needs Python 3.9-3.12 (onnxruntime); the system Python may be newer.
PYVER="3.12"

# --segment-only builds a lean venv with just the Segment Anything stack
# (onnxruntime + numpy + pillow) and no rembg/BiRefNet.
SEGMENT_ONLY="${BGBG_SEGMENT_ONLY:-0}"
for arg in "$@"; do
  case "$arg" in
    --segment-only) SEGMENT_ONLY=1 ;;
    -h|--help)
      echo "Usage: ./install.sh [--segment-only]"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

msg() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }

detect_vendor() {
  if [ -n "${BGBG_VENDOR:-}" ]; then echo "$BGBG_VENDOR"; return; fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo nvidia; return
  fi
  if command -v rocminfo >/dev/null 2>&1 \
     || lspci 2>/dev/null | grep -Eiq 'amd/ati|advanced micro devices.*\[.*(radeon|navi|vega)'; then
    echo amd; return
  fi
  echo cpu
}

deps_present() {
  # Look up the package without importing it (importing onnxruntime needs the
  # CUDA libs on LD_LIBRARY_PATH, which are only set at worker runtime). In
  # segment-only mode there is no rembg, so probe onnxruntime instead.
  local pkg="rembg"; [ "$SEGMENT_ONLY" = 1 ] && pkg="onnxruntime"
  "$VENV/bin/python" -c \
    "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$pkg') else 1)" \
    >/dev/null 2>&1
}

make_venv() {
  if [ -x "$VENV/bin/python" ] && deps_present; then
    msg "Reusing existing worker venv at $VENV"
    return
  fi
  mkdir -p "$DATA"
  if [ -x "$VENV/bin/python" ]; then
    msg "Found venv without dependencies; installing into it"
    if command -v uv >/dev/null 2>&1; then
      PIP=(uv pip install --python "$VENV/bin/python")
    else
      PIP=("$VENV/bin/python" -m pip install --upgrade)
    fi
  elif command -v uv >/dev/null 2>&1; then
    msg "Creating venv (Python $PYVER) with uv"
    uv venv --python "$PYVER" "$VENV"
    PIP=(uv pip install --python "$VENV/bin/python")
  else
    local py
    py="$(command -v "python$PYVER" || true)"
    [ -z "$py" ] && py="$(command -v python3.11 || true)"
    if [ -z "$py" ]; then
      warn "Need 'uv' or python$PYVER. Install uv: https://docs.astral.sh/uv/"
      exit 1
    fi
    msg "Creating venv with $py"
    "$py" -m venv "$VENV"
    PIP=("$VENV/bin/python" -m pip install --upgrade)
    "${PIP[@]}" pip >/dev/null
  fi

  local vendor; vendor="$(detect_vendor)"
  msg "Detected GPU vendor: $vendor"
  if [ "$SEGMENT_ONLY" = 1 ]; then
    msg "Segmentation-only venv (no rembg/BiRefNet)"
    case "$vendor" in
      nvidia)
        "${PIP[@]}" onnxruntime-gpu numpy pillow \
          nvidia-cuda-runtime nvidia-cublas nvidia-cufft nvidia-curand nvidia-cudnn-cu13
        ;;
      amd)
        "${PIP[@]}" numpy pillow
        if "${PIP[@]}" onnxruntime-rocm >/dev/null 2>&1; then
          msg "Installed onnxruntime-rocm (ROCm acceleration)"
        else
          warn "onnxruntime-rocm not available from PyPI; using CPU."
          "${PIP[@]}" onnxruntime
        fi
        ;;
      *)
        "${PIP[@]}" onnxruntime numpy pillow
        ;;
    esac
    return
  fi
  case "$vendor" in
    nvidia)
      "${PIP[@]}" "rembg[gpu]" "numba>=0.60" "llvmlite>=0.43" \
        nvidia-cuda-runtime nvidia-cublas nvidia-cufft nvidia-curand nvidia-cudnn-cu13
      ;;
    amd)
      "${PIP[@]}" "rembg[cpu]" "numba>=0.60" "llvmlite>=0.43"
      if "${PIP[@]}" onnxruntime-rocm >/dev/null 2>&1; then
        msg "Installed onnxruntime-rocm (ROCm acceleration)"
      else
        warn "onnxruntime-rocm not available from PyPI; using CPU."
        warn "For ROCm acceleration, install a matching onnxruntime-rocm wheel into:"
        warn "  $VENV/bin/python -m pip install <onnxruntime_rocm wheel>"
      fi
      ;;
    *)
      "${PIP[@]}" "rembg[cpu]" "numba>=0.60" "llvmlite>=0.43"
      ;;
  esac
}

install_desktop() {
  mkdir -p "$APPS" "$ICONS" "$BINDIR"
  # Symlink (not copy) so the launcher resolves back to this checkout.
  ln -sfn "$ROOT/bin/bg-be-gone" "$BINDIR/bg-be-gone"
  install -m 0644 "$ROOT/data/io.github.cristeigabriela.BgBeGone.png" \
    "$ICONS/io.github.cristeigabriela.BgBeGone.png"
  sed "s|^Exec=bg-be-gone|Exec=$BINDIR/bg-be-gone|" \
    "$ROOT/data/io.github.cristeigabriela.BgBeGone.desktop" > "$APPS/io.github.cristeigabriela.BgBeGone.desktop"
  if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPS" || true
  fi
  if command -v gtk4-update-icon-cache >/dev/null 2>&1; then
    gtk4-update-icon-cache -qtf "${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor" || true
  fi
}

make_venv
install_desktop
msg "Installed. Launch 'bg-be-gone' or find it in your app menu."
if [ "$SEGMENT_ONLY" = 1 ]; then
  msg "Segmentation-only build. First segment downloads a SAM model to"
  msg "  ~/.cache/bg-be-gone/models/ (~110-770 MB depending on your GPU/VRAM)."
else
  msg "First background removal downloads the model (~1 GB) to ~/.u2net/."
  msg "First segmentation downloads a SAM model to ~/.cache/bg-be-gone/models/."
fi
