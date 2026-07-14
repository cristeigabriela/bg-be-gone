"""The CV kernels: masks in, geometry out. numpy + PIL, no model.

Everything the segmenter does to a mask *after* the network has produced it:
score it (stability, IoU, NMS), trace its outline, simplify it, colour it, pack
the per-pixel lookup maps the engine hit-tests against, and composite it.

This is split out of the SAM session on purpose. None of it needs onnxruntime,
none of it needs a GPU, and all of it has to be reimplemented in TypeScript for
the web build -- so it is the half that gets a golden corpus (see
spec/tools/cvgold.py). Given a mask, these functions must produce byte-identical
contours, labelmaps and NMS decisions in both cores.

The interesting subtleties, all golden-pinned:

* ``_label`` is 8-connected. Four-connectivity splits diagonal touches into two
  objects and the object count silently changes.
* ``_trace_boundary`` is a Moore neighbourhood walk; the start pixel and the
  initial direction determine the winding, and the ants are dashed along it.
* ``save_objects`` paints the id maps largest-first for ``general`` and
  smallest-last for ``label``, which is what makes hovering a stack give you the
  whole object and dwelling give you the part.
* Masks are saved as **alpha coverage** (RGBA, A=coverage), not luminance -- an
  L mask is opaque everywhere, so stacking them in one mask node replaces rather
  than unions. See the step-5 bug in the plan.
"""
import os
import zlib
import colorsys
from collections import deque

import numpy as np
from PIL import Image, ImageFilter


def _stability(logit, offset=1.0):
    """SAM stability score: area(>+t) / area(>-t)."""
    high = np.count_nonzero(logit > offset)
    low = np.count_nonzero(logit > -offset)
    return (high / low) if low else 0.0


def _bbox(mask):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _iou(a, b):
    inter = np.count_nonzero(a & b)
    if inter == 0:
        return 0.0
    union = np.count_nonzero(a) + np.count_nonzero(b) - inter
    return inter / union if union else 0.0


def _bbox_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def _nms(records, iou_thresh=0.7, max_objects=256):
    """Greedy IoU-NMS. ``records`` = list of ``(score, mask_bool)``. Bbox
    pre-filter then mask IoU; returns survivors sorted by score desc."""
    records = sorted(records, key=lambda r: r[0], reverse=True)
    kept, kept_bb = [], []
    for score, mask in records:
        bb = _bbox(mask)
        if bb[2] == 0:
            continue
        drop = False
        for km, kbb in zip(kept, kept_bb):
            if _bbox_overlap(bb, kbb) and _iou(mask, km[1]) > iou_thresh:
                drop = True
                break
        if not drop:
            kept.append((score, mask))
            kept_bb.append(bb)
            if len(kept) >= max_objects:
                break
    return kept


# ---------------------------------------------------------------------------


def _obj_color(cx, cy):
    """Deterministic, position-hashed colour so an object keeps the same colour
    across selections and re-runs (not tied to detection order)."""
    key = f"{int(cx) // 8}:{int(cy) // 8}".encode()
    h = (zlib.crc32(key) % 3600) / 3600.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.68, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def _save_soft(mask, path, feather):
    """Write a mask as an alpha-coverage PNG: white RGB, alpha = coverage.

    Alpha, not luminance. Both GSK's mask node and Canvas2D's `destination-in`
    key off ALPHA, so coverage masks composite as a real union when several are
    stacked. Written as an `L` image they load back *opaque*, and stacking those
    made each mask silently REPLACE the previous one instead of unioning with it
    — which is why selecting several objects used to dim all but the last.
    """
    im = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), "L")
    if feather > 0:
        im = im.filter(ImageFilter.GaussianBlur(feather))
    rgba = Image.new("RGBA", im.size, (255, 255, 255, 0))
    rgba.putalpha(im)
    rgba.save(path)


# Clockwise 8-neighbour offsets (row, col) starting at West — for Moore tracing.
_CW = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]


def _trace_boundary(mask):
    """Moore-neighbour boundary tracing. Returns the ordered outer contour
    (list of (x, y)) of the component containing the topmost-leftmost pixel."""
    H, W = mask.shape
    m = np.zeros((H + 2, W + 2), bool)
    m[1:-1, 1:-1] = mask
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return []
    k = np.lexsort((xs, ys))[0]              # topmost, then leftmost
    start = (int(ys[k]), int(xs[k]))
    cur = start
    back = (start[0], start[1] - 1)          # approached from the West (bg)
    out = [cur]
    limit = 8 * int(m.sum()) + 16
    for _ in range(limit):
        bi = _CW.index((back[0] - cur[0], back[1] - cur[1]))
        nxt = None
        for j in range(1, 9):
            dy, dx = _CW[(bi + j) % 8]
            p = (cur[0] + dy, cur[1] + dx)
            if m[p[0], p[1]]:
                pdy, pdx = _CW[(bi + j - 1) % 8]
                back = (cur[0] + pdy, cur[1] + pdx)
                nxt = p
                break
        if nxt is None or nxt == start:
            break
        cur = nxt
        out.append(cur)
    return [(c[1] - 1, c[0] - 1) for c in out]   # -> (x, y), unpad


def _rdp(pts, eps):
    """Ramer-Douglas-Peucker polyline simplification (keeps endpoints)."""
    n = len(pts)
    if n < 3:
        return list(pts)
    P = np.asarray(pts, float)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = P[b] - P[a]
        L2 = float(seg.dot(seg))
        rel = P[a + 1:b] - P[a]
        if L2 == 0.0:
            d = np.hypot(rel[:, 0], rel[:, 1])
        else:
            t = np.clip((rel @ seg) / L2, 0.0, 1.0)
            proj = P[a] + t[:, None] * seg
            d = np.hypot(*(P[a + 1:b] - proj).T)
        if d.size == 0:
            continue
        mi = int(np.argmax(d))
        if d[mi] > eps:
            idx = a + 1 + mi
            keep[idx] = True
            stack.append((a, idx))
            stack.append((idx, b))
    return [pts[i] for i in range(n) if keep[i]]


def _label(mask, connectivity=8):
    """Label connected components (iterative BFS). Returns (labels, count)."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    if connectivity == 8:
        nbr = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    else:
        nbr = ((-1, 0), (1, 0), (0, -1), (0, 1))
    cur = 0
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if lab[sy, sx]:
            continue
        cur += 1
        dq = deque(((sy, sx),))
        lab[sy, sx] = cur
        while dq:
            y, x = dq.popleft()
            for dy, dx in nbr:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = cur
                    dq.append((ny, nx))
    return lab, cur


def contour(mask, epsilon=1.8, min_area=80, max_pts=500, max_dim=480):
    """All boundary polylines of ``mask`` (each a list of [x, y] in original-image
    px): the outer contour of every connected component PLUS its holes, so
    complex or disjoint objects (branchy trees, rings) outline correctly. The
    mask is traced at reduced resolution for speed on large images, then scaled
    back. Empty for tiny masks."""
    if np.count_nonzero(mask) < min_area:
        return []
    H, W = mask.shape
    if max(H, W) > max_dim:
        sc = max_dim / max(H, W)
        sw, sh = max(1, round(W * sc)), max(1, round(H * sc))
        small = np.asarray(Image.fromarray(
            np.where(mask, 255, 0).astype(np.uint8)).resize(
            (sw, sh), Image.NEAREST)) > 127
    else:
        small, sw, sh = mask, W, H
    up = W / sw                                  # small px -> original px
    small_min = max(8, int(min_area * (sw * sh) / (W * H)))

    regions = []
    lab, n = _label(small, 8)
    for k in range(1, n + 1):
        comp = lab == k
        if np.count_nonzero(comp) >= small_min:
            regions.append(comp)
    # holes = background components that don't touch the border
    blab, bn = _label(~small, 4)
    border = (set(blab[0, :].tolist()) | set(blab[-1, :].tolist())
              | set(blab[:, 0].tolist()) | set(blab[:, -1].tolist()))
    for k in range(1, bn + 1):
        if k in border:
            continue
        hole = blab == k
        if np.count_nonzero(hole) >= max(6, small_min // 4):
            regions.append(hole)

    polys = []
    for reg in regions:
        poly = _trace_boundary(reg)
        if len(poly) < 3:
            continue
        poly = [(x * up, y * up) for x, y in poly]
        poly = _rdp(poly, epsilon)
        if len(poly) > max_pts:
            step = int(np.ceil(len(poly) / max_pts))
            poly = poly[::step]
        polys.append([[int(round(x)), int(round(y))] for x, y in poly])
    return polys


def _save_idmap(arr, path):
    """Encode a uint16 id-per-pixel array as ``R + G*256`` in an RGB PNG."""
    H, W = arr.shape
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[..., 0] = (arr & 0xFF).astype(np.uint8)
    rgb[..., 1] = (arr >> 8).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(path)
    return path


def save_objects(masks, size, out_dir, prefix, feather=0.8):
    """Write soft per-object mask PNGs + three lookup maps for hit-testing
    (``masks`` is largest-first). Returns ``(maps, objects)`` where maps has:
      - ``label``   : the most *specific* object per pixel (smallest on top)
      - ``general`` : the most *general* object per pixel (largest on top)
      - ``depth``   : how many objects overlap each pixel (an ``L`` PNG)
    ids are encoded ``R + G*256`` (0 = background)."""
    W, H = size
    os.makedirs(out_dir, exist_ok=True)
    label = np.zeros((H, W), np.uint16)      # specific: smallest painted last
    general = np.zeros((H, W), np.uint16)    # general: largest painted last
    depth = np.zeros((H, W), np.uint16)
    objs = []
    for idx, mask in enumerate(masks):
        oid = idx + 1
        label[mask] = oid
        depth += mask
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        mp = os.path.join(out_dir, f"{prefix}_obj{oid}.png")
        _save_soft(mask, mp, feather)
        x, y, w, h = _bbox(mask)
        objs.append({"id": oid, "color": "#%02x%02x%02x" % _obj_color(cx, cy),
                     "bbox": [x, y, w, h], "area": int(xs.size), "mask": mp,
                     "contour": contour(mask)})
    for idx in range(len(masks) - 1, -1, -1):   # largest painted last -> on top
        general[masks[idx]] = idx + 1
    maps = {
        "label": _save_idmap(label, os.path.join(out_dir, f"{prefix}_labelmap.png")),
        "general": _save_idmap(general, os.path.join(out_dir, f"{prefix}_general.png")),
        "depth": os.path.join(out_dir, f"{prefix}_depth.png"),
    }
    Image.fromarray(np.minimum(depth, 255).astype(np.uint8), "L").save(maps["depth"])
    return maps, objs


def save_mask(mask, path, feather=0.8):
    """Write a single soft-edged mask PNG; return its (x, y, w, h) bbox."""
    _save_soft(mask, path, feather)
    return list(_bbox(mask))


def load_union(paths):
    """Combine soft mask PNGs into one uint8 alpha (max), keeping soft edges."""
    out = None
    for p in paths:
        im = Image.open(p)
        if im.mode in ("RGBA", "LA"):          # coverage masks (see _save_soft)
            a = np.asarray(im.convert("RGBA"))[..., 3]
        else:                                  # legacy L masks
            a = np.asarray(im.convert("L"))
        out = a if out is None else np.maximum(out, a)
    return out


def composite_image(src_pil, alpha, bg, blur=20):
    """The ``alpha`` (uint8 HxW) cutout of ``src_pil`` over ``bg``, as an image.

    The background itself is the outputter's job (``compute.outputs_impl``) — the
    same one the background-removal and GIF paths use, so a new effect is added in
    one place rather than three.
    """
    from compute import outputs_impl

    src = src_pil.convert("RGB")
    cut = outputs_impl.cutout(src, Image.fromarray(alpha.astype(np.uint8), "L"))
    return outputs_impl.apply_bg(cut, bg, source=src, blur=blur)


def composite_extract(src_pil, alpha, bg, dst, blur=20, rot=0, fh=False, fv=False):
    """...and saved to ``dst``, with the view transform baked in.

    The masks live in un-rotated source space (so they can ride the pane
    transform instead of being invalidated by it), which means the rotation the
    user was looking at has to be applied here, at the end.
    """
    from compute import outputs_impl

    img = composite_image(src_pil, alpha, bg, blur=blur)
    outputs_impl.apply_transform(img, rot, fh, fv).save(dst)
    return dst
