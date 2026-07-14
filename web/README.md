# web — the browser core

The TypeScript mirror of `src/bgbg/engine`, plus the Canvas2D render backend.

## What is done, and proven

`bun run web/conform.ts` puts the TS core on trial against the goldens the Python
engine froze:

| gate | what it proves |
| --- | --- |
| 25 display lists | the TS builder emits **byte-identical JSON** for every fixture — every eased curve, dash offset, pulse phase, morph lerp and op ordering |
| UiSchema | `describe()` is byte-identical, so the web sidebar builds from the same declaration |
| CV kernels | components, contour tracing, RDP, IoU/NMS, stability all match `spec/goldens/cv.json` |

Plus `web/tests/pyround_test.ts` (the rounding primitive vs CPython's own answers)
and `web/tests/canvas2d_test.ts` (the op → Canvas2D mapping, against a recording
context).

All of it runs headlessly under `bun`, and is part of `./spec/check.sh`.

## What is NOT done

* **the ORT-Web compute worker** — `engine/protocol.py` declares the 12 jobs and
  20 events it must speak, and `engine/ports.py` has the port interface, but the
  Web Worker itself is not written.
* **the Vite bundle, the DOM shell, and Playwright** — nothing here has been run
  in a real browser. `canvas2d.ts` is verified only against a stub context, so it
  proves the *mapping*, not the pixels. The cross-backend SSIM check the plan asks
  for needs a browser.
* **model hosting** — see the plan: weights must come from the HuggingFace CDN
  (it sends `ACAO: *`); GitHub release assets send no CORS header, and Pages will
  not serve Git-LFS.

## The two traps that cost the most

1. **Rounding.** Python's `round()` is half-to-**even** on the exact binary value.
   `toFixed` is half-away-from-zero; `Math.round(x*1e4)/1e4` is inexact in a third
   way. They disagree on ~0.6% of real values (`0.03125` → Python `0.0312`, JS
   `0.0313`). Every float in the display list is rounded, so this had to be exact
   before anything else could be. See `core/pyround.ts`.
2. **Iteration order.** A Python `set` iterates in hash order, a JS `Set` in
   insertion order. The builder sorts the selection explicitly for this reason —
   otherwise the two cores emit ops in different orders for the same scene.
