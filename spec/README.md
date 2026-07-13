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
| `goldens/render/*.png` | frozen output of the **current** renderer — the regression gate |

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

## Known bug frozen into the corpus

`09_select_two_UNION` and `22_seg_zoom4_select` currently freeze a **bug**, on purpose.

Object masks are written as `L` PNGs (`segmentation._save_soft`), which GdkPixbuf loads
as **opaque** RGB. `viewer.py` builds the "dim outside the selection" mask by stacking
several `append_texture` calls inside one `push_mask` — but because each texture is opaque
across the full image rect, each one **overwrites** the previous. The mask is therefore
*last-wins*, not a union: every selected object except the last is dimmed underneath its
own tint.

Measured on this scene: **obj1 renders 23% darker when co-selected with obj2 than when
selected alone.** It reads as "the highlight gets muddy when I select several objects".

Step 4 of the refactor reproduces this **exactly** (goldens must not move). Step 5 switches
masks to alpha-coverage, which makes the union real and fixes it — at which point these two
goldens are expected to change, and that delta is the proof of the fix.
