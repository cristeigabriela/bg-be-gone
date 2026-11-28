"""Segment everything's progress and cancel, on a stand-in for the SAM decoder.

The real models are hundreds of MB; the stand-in answers each grid point with
logits that cover one of two boxes, which is all `everything` needs.
"""
import threading

import numpy as np
import pytest

import segmentation as seg

# The image: 400x300, so 1024-space is 1024x768 and the logits cover 256x192.
SIZE = (400, 300)


class BoxesSession(seg.SamSession):
    """SamSession with the model replaced: left points find the left box, right
    points the right one, each with a confident score."""

    def __init__(self):  # noqa: D107 - no encoder or decoder to load
        self.family = "sam2"
        self._size = SIZE
        self._scale = 1024.0 / max(SIZE)
        self._nw = round(SIZE[0] * self._scale)
        self._nh = round(SIZE[1] * self._scale)

    def _grid_decode(self, x, y):
        logits = np.full((256, 256), -8.0, np.float32)
        if x < self._nw / 2:
            logits[20:170, 10:110] = 8.0
        else:
            logits[40:150, 140:240] = 8.0
        return logits, 0.95


def test_progress_is_reported_for_every_grid_point():
    """The bar used to move only when a point passed the filters: 4 calls for
    256 points, the last at 217 of 256."""
    calls = []
    masks = BoxesSession().everything(points_per_side=8, progress=lambda d, t: calls.append((d, t)))
    assert calls == [(i, 64) for i in range(1, 65)], "one call per point, ending at 64 of 64"
    assert len(masks) == 2, "the duplicates of each box collapse to one object"
    assert all(m.shape == (SIZE[1], SIZE[0]) for m in masks), "masks come back at full size"


def test_a_cancel_stops_the_grid_early():
    cancel = threading.Event()
    seen = []

    def progress(done, total):
        seen.append(done)
        if done == 10:
            cancel.set()
    BoxesSession().everything(points_per_side=8, cancel=cancel, progress=progress)
    assert seen[-1] == 10, "no point is decoded after the cancel"


def test_building_layers_reports_each_object(tmp_path):
    masks = BoxesSession().everything(points_per_side=4)
    calls = []
    _, objects = seg.save_objects(masks, SIZE, str(tmp_path), "e1",
                                  progress=lambda d, t: calls.append((d, t)))
    assert [o["id"] for o in objects] == [1, 2], "both objects are written"
    assert calls == [(1, 2), (2, 2)], "progress after each object, ending at 2 of 2"


def test_building_layers_stops_on_cancel(tmp_path):
    """Writing the objects out had no cancel: ~0.4 s each at 4K, up to 256 of them."""
    masks = BoxesSession().everything(points_per_side=4)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(seg.SegCanceled):
        seg.save_objects(masks, SIZE, str(tmp_path), "e1", cancel=cancel)
    assert not list(tmp_path.glob("*_obj*.png")), "no object is written after the cancel"


def test_a_cancel_after_the_grid_stops_before_dedup():
    """De-duplication and scaling up had no cancel check of their own."""
    cancel = threading.Event()

    def progress(done, total):
        if done == total:
            cancel.set()
    masks = BoxesSession().everything(points_per_side=4, cancel=cancel, progress=progress)
    assert masks == [], "a cancel set after the last point publishes nothing"


def test_layer_progress_ends_only_once_the_maps_are_written(tmp_path):
    """The counter reached its total before the lookup maps were built."""
    masks = BoxesSession().everything(points_per_side=4)
    seen = []

    def progress(done, total):
        seen.append((done, total, (tmp_path / "e1_depth.png").exists()))
    seg.save_objects(masks, SIZE, str(tmp_path), "e1", progress=progress)
    assert seen[-1] == (2, 2, True), "the last report comes after the maps"


def test_a_cancel_during_the_last_object_writes_no_maps(tmp_path):
    """A late cancel still published the new objects over the ones being kept."""
    masks = BoxesSession().everything(points_per_side=4)
    cancel = threading.Event()

    def progress(done, total):
        if done == total - 1:
            cancel.set()
    with pytest.raises(seg.SegCanceled):
        seg.save_objects(masks, SIZE, str(tmp_path), "e1", cancel=cancel, progress=progress)
    assert not (tmp_path / "e1_labelmap.png").exists(), "no lookup map is written after a cancel"
