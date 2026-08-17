#!/usr/bin/env bash
# Install bg-be-gone for the current user: build the worker virtualenv,
# register the desktop entry and icon. Re-runnable (idempotent).
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}/bg-be-gone"
VENV="$DATA/venv"
GPU_ENV="$DATA/gpu.env"
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
      echo "Usage: ./install.sh [--segment-only]"
      echo
      echo "Environment:"
      echo "  BGBG_VENDOR=nvidia|amd|cpu    skip GPU autodetection"
      echo "  BGBG_NO_ROCM=1                make the worker ignore the AMD GPU"
      exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

msg()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

# The amdgpu kernel driver publishes each compute agent's LLVM target through
# amdkfd as packed decimal (major*10000 + minor*100 + step), so 100301 is
# gfx1031. This is the authoritative answer and — unlike rocminfo — it is
# readable without ROCm installed, which is exactly the case on a first run.
amd_gfx_archs() {
  local f v maj min stp
  for f in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    [ -r "$f" ] || continue
    v="$(awk '$1 == "gfx_target_version" { print $2; exit }' "$f")"
    # Skip nodes with no or a non-numeric target (the CPU node reports 0).
    case "${v:-}" in ''|0|*[!0-9]*) continue ;; esac
    # minor and step are nibbles in the LLVM name (step 10 -> gfx90a).
    maj=$(( v / 10000 )); min=$(( v / 100 % 100 )); stp=$(( v % 100 ))
    printf 'gfx%d%x%x\n' "$maj" "$min" "$stp"
  done | sort -u
}

detect_vendor() {
  if [ -n "${BGBG_VENDOR:-}" ]; then echo "$BGBG_VENDOR"; return; fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo nvidia; return
  fi
  # /dev/kfd only exists once amdgpu has bound a compute-capable card.
  if [ -e /dev/kfd ] && [ -n "$(amd_gfx_archs)" ]; then echo amd; return; fi
  if command -v rocminfo >/dev/null 2>&1 \
     || lspci 2>/dev/null | grep -Eiq 'amd/ati|advanced micro devices.*\[.*(radeon|navi|vega)'; then
    echo amd; return
  fi
  echo cpu
}

# ---------------------------------------------------------------------------
# ROCm
# ---------------------------------------------------------------------------

# The ROCm provider is used only when it carries device code for the card that
# is actually installed — never via HSA_OVERRIDE_GFX_VERSION. Making a card run
# a sibling's kernels is the usual advice for GPUs ROCm does not support, but on
# an RX 6750 XT (gfx1031 posing as gfx1030 — same ISA, twice the CUs) it emits
# NaN on BiRefNet and then hangs the GPU hard enough to force a full reset,
# taking the desktop compositor with it.
#
# For a card the stock wheel does not cover, supply a provider built for it via
# BGBG_ROCM_WHEEL (see github.com/cristeigabriela/onnxruntime-rocm-gfx1031).
# Otherwise the CPU is used: slower, but correct, and it cannot wedge the box.

# Wheel to install for AMD. An explicit BGBG_ROCM_WHEEL wins; otherwise a
# sibling checkout of the gfx1031 build repo; otherwise PyPI.
amd_wheel_source() {
  if [ -n "${BGBG_ROCM_WHEEL:-}" ]; then echo "$BGBG_ROCM_WHEEL"; return; fi
  local sibling=()
  shopt -s nullglob
  sibling=("$ROOT/../onnxruntime-rocm-gfx1031/dist"/onnxruntime_rocm-*-cp"${PYVER/./}"-*.whl)
  shopt -u nullglob
  if [ "${#sibling[@]}" -gt 0 ]; then echo "${sibling[${#sibling[@]}-1]}"; return; fi
  echo "onnxruntime-rocm"
}

# True if the installed ROCm provider contains device code for $1. hipcc tags
# every embedded offload bundle with its target triple, so this asks the binary
# rather than trusting a list of what some wheel is assumed to ship.
ep_has_arch() {
  local so=()
  shopt -s nullglob
  so=("$VENV"/lib/python*/site-packages/onnxruntime/capi/libonnxruntime_providers_rocm.so)
  shopt -u nullglob
  [ "${#so[@]}" -gt 0 ] || return 1
  grep -qa "amdgcn-amd-amdhsa--$1" "${so[0]}"
}

# A 930-byte ONNX model: Conv -> Relu -> Mul -> Add -> Softmax -> Pool -> Gemm,
# which is enough to pull in MIOpen, onnxruntime's own HIP kernels and rocBLAS.
# Over a 1x3x8x8 input of ones the output is four 0.25s, so sum == 1. A Conv on
# its own is not enough: MIOpen JIT-compiles for the live arch and passes even
# where the precompiled onnxruntime kernels abort.
PROBE_MODEL_B64='CAg6lwcKPQoBeAoBdwoCY2ISAWMiBENvbnYqFQoMa2VybmVsX3NoYXBlQANAA6ABByoRCgRwYWRzQAFAAUABQAGgAQcKDAoBYxIBciIEUmVsdQoQCgFyCgN0d28SAW0iA011bAoQCgFtCgNvbmUSAWEiA0FkZAocCgFhEgFzIgdTb2Z0bWF4KgsKBGF4aXMYAaABAgoZCgFzEgFnIhFHbG9iYWxBdmVyYWdlUG9vbAoWCgFnCgVzaGFwZRIBZiIHUmVzaGFwZQoUCgFmCgJndwoCZ2ISAXkiBEdlbW0SBXByb2JlKsADCAQIAwgDCAMQAUIBd0qwAwAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAACoaCAQQAUICY2JKEAAAAAAAAAAAAAAAAAAAAAAqDRABQgN0d29KBAAAAEAqDRABQgNvbmVKBAAAgD8qHQgCEAdCBXNoYXBlShABAAAAAAAAAAQAAAAAAAAAKkwIBAgEEAFCAmd3SkAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAAAAAAIA/KhoIBBABQgJnYkoQAAAAAAAAAAAAAAAAAAAAAFobCgF4EhYKFAgBEhAKAggBCgIIAwoCCAgKAggIYhMKAXkSDgoMCAESCAoCCAEKAggEQgQKABAN'

# Exit 0 means the model genuinely ran on $1 (an execution provider name) and
# returned the right numbers. HSA_OVERRIDE_GFX_VERSION is explicitly cleared:
# the provider is only ever used when it has native code for this card, so an
# override could only push it onto the wrong chip's kernels.
gpu_probe() {
  env -u HSA_OVERRIDE_GFX_VERSION "$VENV/bin/python" - <<PY 2>&1
import base64, sys
import numpy as np
import onnxruntime as ort

so = ort.SessionOptions()
so.log_severity_level = 3   # errors only: the provider-load failure, no INFO flood
s = ort.InferenceSession(base64.b64decode("$PROBE_MODEL_B64"), so,
                         providers=["$1"])
# onnxruntime does NOT raise when a provider fails to load — it drops the
# request and quietly runs on the CPU, so check the provider is really there.
if "$1" not in s.get_providers():
    sys.exit("provider not active (fell back to %s)" % ", ".join(s.get_providers()))
out = s.run(None, {"x": np.ones((1, 3, 8, 8), np.float32)})[0]
if not np.isfinite(out).all() or abs(float(out.sum()) - 1.0) > 1e-4:
    sys.exit("provider ran but returned garbage (sum=%r)" % float(out.sum()))
PY
}

rocm_packages_hint() {
  local id="" like=""
  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    id="$(. /etc/os-release; echo "${ID:-}")"
    # shellcheck disable=SC1091
    like="$(. /etc/os-release; echo "${ID_LIKE:-}")"
  fi
  case "$id $like" in
    *arch*|*cachyos*|*manjaro*|*endeavouros*)
      echo "sudo pacman -S --needed rocm-hip-runtime miopen-hip rocblas hipblas hipfft hipsparse rocm-smi-lib roctracer" ;;
    *fedora*|*rhel*|*centos*)
      echo "sudo dnf install rocm-hip miopen rocblas hipblas hipfft hipsparse rocm-smi" ;;
    *debian*|*ubuntu*)
      echo "sudo apt install rocm-hip-runtime miopen-hip rocblas hipblas hipfft hipsparse rocm-smi-lib roctracer  # needs AMD's repo, see https://rocm.docs.amd.com" ;;
    *suse*)
      echo "sudo zypper install rocm-hip-runtime miopen-hip rocblas hipblas hipfft hipsparse" ;;
    *)
      echo "install the ROCm runtime for your distro — https://rocm.docs.amd.com" ;;
  esac
}

# Decide how the worker should talk to the AMD GPU and record it in gpu.env,
# which worker.py sources before it imports onnxruntime.
configure_rocm() {
  local archs arch out
  archs="$(amd_gfx_archs)"
  arch="$(echo "$archs" | head -n1)"

  if [ -z "$arch" ]; then
    warn "AMD GPU detected but the kernel exposes no compute agent."
    warn "Load the amdgpu driver (and check /dev/kfd exists), then re-run."
    rm -f "$GPU_ENV"; return
  fi
  msg "AMD compute agent: $arch"
  [ "$(echo "$archs" | wc -l)" -gt 1 ] && \
    warn "Multiple agents ($(echo "$archs" | tr '\n' ' ')); tuning for $arch."

  if [ ! -w /dev/kfd ]; then
    warn "No write access to /dev/kfd — ROCm needs it. Add yourself to the"
    warn "'render' and 'video' groups, then log back in."
  fi

  # Refuse the GPU unless the provider genuinely has code for this card.
  if ! ep_has_arch "$arch"; then
    warn "This onnxruntime-rocm build has no device code for $arch."
    warn "Using the CPU: impersonating another GPU with HSA_OVERRIDE_GFX_VERSION"
    warn "can produce silently wrong output and hard-reset the GPU."
    warn "For a build targeting $arch, see:"
    warn "  https://github.com/cristeigabriela/onnxruntime-rocm-gfx1031"
    warn "then re-run with BGBG_ROCM_WHEEL=/path/to/wheel ./install.sh"
    rm -f "$GPU_ENV"
    return
  fi
  msg "Provider has native $arch device code"

  msg "Verifying the ROCm provider (running a model on the GPU)"
  if out="$(gpu_probe ROCMExecutionProvider)"; then
    mkdir -p "$DATA"
    {
      echo "# Written by bg-be-gone install.sh — sourced by the worker at startup."
      echo "# Delete this file and re-run ./install.sh to redetect."
      echo "# GPU: $arch"
      echo "#"
      echo "# HSA_OVERRIDE_GFX_VERSION is deliberately absent and must stay that"
      echo "# way: the provider carries real $arch device code, so nothing needs"
      echo "# to impersonate anything. Forcing an override here would push"
      echo "# rocBLAS onto kernels for a different chip."
      echo "MIOPEN_LOG_LEVEL=3"
    } > "$GPU_ENV"
    ok "GPU acceleration verified — native $arch kernels"
    return
  fi

  warn "The ROCm provider could not run; bg-be-gone will fall back to the CPU."
  printf '%s\n' "$out" | tail -n 6 | sed 's/^/     /' >&2
  if printf '%s' "$out" | grep -q 'cannot open shared object file'; then
    warn "The ROCm runtime libraries are missing. Install them with:"
    warn "  $(rocm_packages_hint)"
    warn "then re-run ./install.sh"
  fi
  rm -f "$GPU_ENV"
}

# ---------------------------------------------------------------------------
# Virtualenv
# ---------------------------------------------------------------------------

VENDOR="$(detect_vendor)"

have_module() {
  # Look up the package without importing it (importing onnxruntime needs the
  # GPU libs on LD_LIBRARY_PATH, which are only set at worker runtime).
  "$VENV/bin/python" -c \
    "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)" \
    >/dev/null 2>&1
}

have_dist() {
  "$VENV/bin/python" - "$1" >/dev/null 2>&1 <<'PY'
import sys
from importlib.metadata import distribution, PackageNotFoundError
try:
    distribution(sys.argv[1])
except PackageNotFoundError:
    sys.exit(1)
PY
}

deps_present() {
  # In segment-only mode there is no rembg, so probe onnxruntime instead.
  local pkg="rembg"; [ "$SEGMENT_ONLY" = 1 ] && pkg="onnxruntime"
  have_module "$pkg" || return 1
  # A venv built before GPU support (or on other hardware) has the wrong
  # runtime; the vendor wheel is what makes it worth reusing.
  case "$VENDOR" in
    amd)    have_dist onnxruntime-rocm || return 1 ;;
    nvidia) have_dist onnxruntime-gpu  || return 1 ;;
  esac
  return 0
}

pip_uninstall() {
  if command -v uv >/dev/null 2>&1; then
    uv pip uninstall --python "$VENV/bin/python" "$@" >/dev/null 2>&1 || true
  else
    "$VENV/bin/python" -m pip uninstall -y "$@" >/dev/null 2>&1 || true
  fi
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

  msg "Detected GPU vendor: $VENDOR"

  # Every onnxruntime flavour unpacks over the same onnxruntime/ package, so they
  # cannot coexist: whichever lands second wins the files and leaves the other's
  # metadata dangling. Clear the others before the ROCm build goes in.
  if [ "$VENDOR" = amd ]; then
    for d in onnxruntime onnxruntime-migraphx; do
      if have_dist "$d"; then
        msg "Removing $d (conflicts with onnxruntime-rocm)"
        pip_uninstall "$d"
      fi
    done
  fi

  # onnxruntime-rocm declares no dependencies at all, so name what the stock
  # onnxruntime pulls in. A locally built wheel needs the same.
  local ORT_AMD_DEPS=(numpy flatbuffers packaging protobuf coloredlogs sympy)
  local ORT_AMD_WHEEL=""
  if [ "$VENDOR" = amd ]; then
    ORT_AMD_WHEEL="$(amd_wheel_source)"
    case "$ORT_AMD_WHEEL" in
      onnxruntime-rocm) msg "onnxruntime-rocm from PyPI" ;;
      *) msg "onnxruntime-rocm from $ORT_AMD_WHEEL" ;;
    esac
  fi

  if [ "$SEGMENT_ONLY" = 1 ]; then
    msg "Segmentation-only venv (no rembg/BiRefNet)"
    case "$VENDOR" in
      nvidia)
        "${PIP[@]}" onnxruntime-gpu numpy pillow \
          nvidia-cuda-runtime nvidia-cublas nvidia-cufft nvidia-curand nvidia-cudnn-cu13
        ;;
      amd)
        "${PIP[@]}" "$ORT_AMD_WHEEL" "${ORT_AMD_DEPS[@]}" pillow
        ;;
      *)
        "${PIP[@]}" onnxruntime numpy pillow
        ;;
    esac
    return
  fi
  case "$VENDOR" in
    nvidia)
      "${PIP[@]}" "rembg[gpu]" "numba>=0.60" "llvmlite>=0.43" \
        nvidia-cuda-runtime nvidia-cublas nvidia-cufft nvidia-curand nvidia-cudnn-cu13
      ;;
    amd)
      # Plain "rembg", not rembg[rocm]: that extra pins its own onnxruntime-rocm
      # from PyPI, which would clobber a wheel built for this card.
      "${PIP[@]}" rembg "numba>=0.60" "llvmlite>=0.43" \
        "$ORT_AMD_WHEEL" "${ORT_AMD_DEPS[@]}"
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
# Runs on every install, not just a fresh venv: a driver, ROCm or wheel upgrade
# can change which arch actually works.
if [ "$VENDOR" = amd ]; then configure_rocm; else rm -f "$GPU_ENV"; fi
install_desktop
msg "Installed. Launch 'bg-be-gone' or find it in your app menu."
if [ "$SEGMENT_ONLY" = 1 ]; then
  msg "Segmentation-only build. First segment downloads a SAM model to"
  msg "  ~/.cache/bg-be-gone/models/ (~110-770 MB depending on your GPU/VRAM)."
else
  msg "First background removal downloads the model (~1 GB) to ~/.u2net/."
  msg "First segmentation downloads a SAM model to ~/.cache/bg-be-gone/models/."
fi
