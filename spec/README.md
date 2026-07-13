# spec/ — the cross-platform contract

This directory is the anti-drift spine for the engine extraction and the web port.
It exists so that a second implementation of the engine (TypeScript, in the browser)
can be written **mechanically** rather than archaeologically, and so that the desktop
refactor can be proven not to have changed a single pixel.

## Layout

| path | what |
| --- | --- |
| `assets/gen_scene.py` | builds the golden scene through the **real** `segmentation.save_objects`, so mask/idmap/contour encoding is byte-for-byte production |
| `assets/scene/` | the generated scene (committed): source + 3 object masks + label/general/depth maps + `meta.json` |
| `fixtures/*.json` | 22 render fixtures: pane state × animation state × segmentation state |
| `tools/rasterize.py` | deterministic offscreen render of `ImageView` at a fixed state and time |
| `tools/displaylist.py` | freeze/check the **display lists** the engine builds for those fixtures |
| `goldens/render/*.png` | frozen output of the renderer — the pixel regression gate |
| `goldens/display_list/*.json` | frozen output of the engine — the **cross-core** gate |

## Two gates, and why both

`goldens/render/*.png` prove the *pixels*. `goldens/display_list/*.json` prove the
*decisions* — the eased curves, dash offsets, pulse phase, morph lerp, wave schedule —
as structured JSON, before any backend touches them.

That second one is what makes the TypeScript core a mechanical port rather than an
archaeological dig: it must emit byte-identical JSON for the same fixture, and a
disagreement points at the exact op that drifted instead of at a blurry pixel diff.
Every display list is also round-tripped through the JSON codec, so a lossy wire
format can never be frozen.

## The scene

Three objects, chosen to exercise every visual:

| id | name | why |
| --- | --- | --- |
| 1 | `field` | the large blob — the *general* object of a stack |
| 2 | `island` | two disconnected squares — **multi-polygon** contour |
| 3 | `core` | a disc nested inside `field` — gives **depth ≥ 2** (layered masks) |

`core` is deliberately **off-centre** inside `field`: `_obj_color` hashes the centroid
bucket (`cx//8`, `cy//8`), so a *concentric* child would hash to the same colour as its
parent and the layered-mask visual (general vs specific in **distinct** colours) would be
untestable. (This is a real quirk of `_obj_color`, not just a fixture concern.)

## Time

The widget has no frame clock offscreen, so `ImageView._now()` returns 0. Every animation
timestamp in `viewer.py` is absolute frame-clock µs, so a fixture states an **age**
(`"press_ms_ago": 300`) and the rasteriser writes it as negative µs. That makes the entire
animation state addressable without a running main loop.

## Usage

```sh
python spec/tools/rasterize.py --freeze   # write goldens
python spec/tools/rasterize.py --check    # compare (writes *.png.actual on a diff)
python spec/tools/rasterize.py --list
```

`--freeze` renders every fixture **twice** and refuses to write unless both runs are
byte-identical, so a non-deterministic renderer can never be baked into the corpus.

Regenerate the scene (needs the worker venv + `LD_LIBRARY_PATH` for onnxruntime):

```sh
LIBS=$(ls -d ~/.local/share/bg-be-gone/venv/lib/python*/site-packages/nvidia/*/lib | tr '\n' ':')
LD_LIBRARY_PATH="$LIBS" ~/.local/share/bg-be-gone/venv/bin/python spec/assets/gen_scene.py
```

## The bug this corpus caught, and fixed

The corpus was frozen from the *pre-refactor* renderer, and immediately paid for itself.

Object masks used to be written as `L` PNGs (`segmentation._save_soft`), which load back
**opaque**. The "dim outside the selection" mask is built by stacking one texture per
selected object inside a single mask node — and stacking *opaque* textures makes each one
**replace** the last. The mask therefore collapsed to whichever object was iterated last,
and every other selected object got the 0.55 black dim painted over it, underneath its own
tint.

Measured on this scene: **object 1 rendered 23% darker when co-selected with object 2 than
when selected alone.** It never looked obviously broken — it read as "the highlight goes
muddy when I select several objects" — which is exactly why it needed a number.

Masks are now **alpha-coverage** (white RGB, alpha = coverage), so stacking them
alpha-composites into a real union. GSK keys the mask off `ALPHA`, and so does Canvas2D's
`destination-in` — which is the other half of why the change matters: a luminance mask
could not have been ported to the browser at all.

The fix moved exactly two goldens — `09_select_two_UNION` and `22_seg_zoom4_select`, the
only two fixtures with more than one object selected — and left the other twenty
byte-identical. `tests/test_mask_union.py` pins it: co-selection now drifts object 1's
luma by 0.0%, and unselected objects still dim.
