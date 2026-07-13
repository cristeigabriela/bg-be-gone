#!/usr/bin/env bash
# Every gate the engine extraction must hold. Run from the repo root:
#
#     ./spec/check.sh
#
# The two golden suites are the spine: the render goldens prove the pixels, the
# display-list goldens prove the engine's decisions (and are what the TypeScript
# core must reproduce byte-for-byte).
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0
run() {                       # run <label> <cmd...>
  local label="$1"; shift
  printf '%-34s ' "$label"
  if "$@" >/tmp/_gate.log 2>&1; then
    echo "PASS"
  else
    echo "FAIL"
    sed 's/^/    /' /tmp/_gate.log | tail -12
    fail=1
  fi
}

echo "── static ──────────────────────────────────"
run "compile"                 python -m compileall -q src

echo "── engine (stdlib only, no GTK) ────────────"
run "stdlib-only import"      python tests/test_engine_stdlib_only.py
run "pane / coords"           python tests/test_engine_pane.py
run "hit-testing"             python tests/test_engine_hittest.py
run "animation state machine" python tests/test_engine_anim.py
run "interaction (events->effects)" python tests/test_engine_interaction.py

echo "── renderer ────────────────────────────────"
run "render goldens (22)"     python spec/tools/rasterize.py --check
run "display-list goldens"    python spec/tools/displaylist.py --check
run "sidebar (row-for-row)"   timeout 60 python spec/tools/uidump.py --check
run "mask union (regression)" python tests/test_mask_union.py

echo "── live (real GTK window) ──────────────────"
run "tick / dwell / press"    timeout 60 python tests/test_live_tick.py
run "app seam (hover/click/sidebar)" timeout 60 python tests/test_live_app_segment.py

echo
if [ "$fail" -eq 0 ]; then
  echo "all gates pass"
else
  echo "SOME GATES FAILED"
fi
exit "$fail"
