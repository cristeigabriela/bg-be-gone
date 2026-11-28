"""The Batch page's preview art re-composites as a kept Generate result does."""
import pytest
from PIL import Image

from workerpipe import WORKER

# What the page hands the worker's recomposite op as its keep_dir.
ART = WORKER.parent / "assets" / "preview"
# A pixel of the bust's head, and one of the sky beside it (720x480 art).
HEAD, SKY = (360, 250), (60, 60)


@pytest.mark.parametrize("bg", [
    # Remove: the cut-out, with a checkered copy to show.
    "transparent",
    # Blur: the picture behind the bust, blurred.
    "blur",
    # Colour: the bust on white.
    "#ffffff",
], ids=["transparent", "blur", "colour"])
def test_the_preview_art_recomposites_onto_each_background(worker, tmp_path, bg):
    """The page shows nothing if the art's files aren't the ones recomposite reads."""
    out = tmp_path / "preview.png"
    reply, _ = worker.ask(
        {"op": "recomposite", "keep_dir": str(ART), "output": str(out), "bg": bg, "blur": 20},
        until={"recomposited", "error"})
    assert reply["type"] == "recomposited", reply.get("message")
    shown = Image.open(reply["preview"]).convert("RGB")
    assert shown.size == (720, 480), "the preview is the art's size"
    cutout = Image.open(ART / "cutout.png").convert("RGB")
    assert shown.getpixel(HEAD) == cutout.getpixel(HEAD), "the bust stays sharp on top"
    if bg == "#ffffff":
        assert shown.getpixel(SKY) == (255, 255, 255), "the sky becomes the colour"
