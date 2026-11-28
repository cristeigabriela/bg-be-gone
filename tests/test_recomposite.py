"""Re-compositing a kept result onto another background, over the worker's pipe.

The result must be the file Generate writes with the same settings, so a
live change of background is indistinguishable from generating again.
u2netp on the CPU is deterministic, so two Generates of one input agree.
"""
import shutil

import pytest
from PIL import Image

from workerpipe import TESTDATA

SUBJECT = TESTDATA / "subject.png"
ANIMATION = TESTDATA / "animated.webp"
GREEN = "#00b140"


def generate(worker, tmp_path, name, bg, blur=20):
    """Generate with `bg`, keeping what recomposite needs; returns (output, keep folder)."""
    out, keep = tmp_path / f"{name}.png", tmp_path / f"{name}-kept"
    reply, _ = worker.ask(
        {"op": "single", "model": "u2netp", "id": 1, "input": str(SUBJECT),
         "output": str(out), "bg": bg, "blur": blur, "keep_dir": str(keep)},
        until={"done_single", "error"})
    assert reply["type"] == "done_single", reply.get("message")
    return out, keep


def recomposite(worker, tmp_path, name, keep, bg, blur=20):
    out = tmp_path / f"{name}.png"
    reply, _ = worker.ask(
        {"op": "recomposite", "id": 2, "keep_dir": str(keep), "output": str(out),
         "bg": bg, "blur": blur},
        until={"recomposited", "error"})
    assert reply["type"] == "recomposited", reply.get("message")
    return out


@pytest.mark.parametrize(
    "bg, blur",
    [
        # A colour, flattened.
        (GREEN, 20),
        # The blurred picture, at the default strength and at a strong one.
        ("blur", 20),
        ("blur", 80),
    ],
    ids=["colour", "blur", "strong blur"],
)
def test_a_recomposite_is_the_file_generate_writes(worker, tmp_path, bg, blur):
    """What's kept comes from a transparent Generate; the background comes later."""
    fresh, _ = generate(worker, tmp_path, "fresh", bg, blur)
    _, keep = generate(worker, tmp_path, "transparent", "transparent")
    live = recomposite(worker, tmp_path, "live", keep, bg, blur)
    assert live.read_bytes() == fresh.read_bytes(), \
        "the same cut-out and settings give the same file, byte for byte"


def test_transparent_to_colour_to_blur_and_back_is_the_original(worker, tmp_path):
    original, keep = generate(worker, tmp_path, "original", "transparent")
    recomposite(worker, tmp_path, "green", keep, GREEN)
    recomposite(worker, tmp_path, "blurred", keep, "blur")
    back = recomposite(worker, tmp_path, "back", keep, "transparent")
    assert back.read_bytes() == original.read_bytes(), \
        "going round the backgrounds leaves the transparent result as it was"


def gif(worker, tmp_path, name, bg, keep=None):
    out = tmp_path / f"{name}.webp"
    req = {"op": "gif", "model": "u2netp", "id": 3, "input": str(ANIMATION),
           "output": str(out), "bg": bg}
    if keep:
        req["keep_dir"] = str(keep)
    reply, _ = worker.ask(req, until={"gif_done", "error"})
    assert reply["type"] == "gif_done", reply.get("message")
    return out


def test_an_animation_recomposite_is_the_animation_generate_writes(worker, tmp_path):
    fresh = gif(worker, tmp_path, "fresh", GREEN)
    keep = tmp_path / "kept"
    gif(worker, tmp_path, "transparent", "transparent", keep)
    live = tmp_path / "live.webp"
    reply, seen = worker.ask(
        {"op": "recomposite", "id": 4, "keep_dir": str(keep), "input": str(ANIMATION),
         "output": str(live), "bg": GREEN},
        until={"recomposited", "error"})
    assert reply["type"] == "recomposited", reply.get("message")
    progress = [r["done"] for r in seen if r.get("type") == "gif_progress"]
    assert progress == [0, 1, 2, 3], "every frame is reported, so the footer bar moves"
    with Image.open(live) as a, Image.open(fresh) as b:
        assert a.n_frames == b.n_frames == 3, "every frame is recomposed"
        for i in range(a.n_frames):
            a.seek(i)
            b.seek(i)
            assert a.convert("RGBA").tobytes() == b.convert("RGBA").tobytes(), \
                f"frame {i} is the frame Generate writes"



def test_an_animation_whose_source_changed_is_not_recomposited(worker, tmp_path):
    """The kept cut-outs belong to the frames as they were; another file's
    frames would get them wrong."""
    source = tmp_path / "moving.webp"
    shutil.copyfile(ANIMATION, source)
    keep = tmp_path / "kept"
    out = tmp_path / "transparent.webp"
    reply, _ = worker.ask(
        {"op": "gif", "model": "u2netp", "id": 3, "input": str(source), "output": str(out),
         "bg": "transparent", "keep_dir": str(keep)},
        until={"gif_done", "error"})
    assert reply["type"] == "gif_done", reply.get("message")
    shutil.copyfile(SUBJECT, source)                 # the file changes on disk
    live = tmp_path / "live.webp"
    reply, _ = worker.ask(
        {"op": "recomposite", "id": 4, "keep_dir": str(keep), "input": str(source),
         "output": str(live), "bg": GREEN},
        until={"recomposited", "error"})
    assert reply["type"] == "error", "a changed source isn't recomposited"
    assert reply["message"] == "The original file changed; press Generate again.", \
        "the user learns why, and what to do"
    assert not live.exists(), "nothing is written"
