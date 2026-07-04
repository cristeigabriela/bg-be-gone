#!/usr/bin/env bash
# Remove bg-be-gone desktop integration and (optionally) the worker venv.
set -euo pipefail

DATA="${XDG_DATA_HOME:-$HOME/.local/share}/bg-be-gone"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/512x512/apps"
BINDIR="$HOME/.local/bin"

rm -f "$APPS/io.github.cristeigabriela.BgBeGone.desktop"
rm -f "$ICONS/io.github.cristeigabriela.BgBeGone.png"
rm -f "$BINDIR/bg-be-gone"
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APPS" || true
fi

if [ "${1:-}" = "--purge" ]; then
  rm -rf "$DATA"
  echo "Removed venv at $DATA"
else
  echo "Left worker venv at $DATA (re-run with --purge to remove it)."
fi
echo "Uninstalled desktop integration."
