"""Segment's Save works from the window's masks, with no SAM model loaded."""
from PIL import Image

from workerpipe import BACKGROUND_PIXEL, SUBJECT_PIXEL, TESTDATA


def test_save_selection_works_with_no_model_loaded(worker, tmp_path):
    """Leaving the Segment page frees the model; the objects found must still save.

    A fresh worker has never loaded SAM, like one after an unload or a restart.
    subject-mask.png covers a disc of radius 48 around subject.png's centre.
    """
    out = tmp_path / "selection.png"
    reply, _ = worker.ask(
        {"op": "seg_extract", "input": str(TESTDATA / "subject.png"),
         "masks": [str(TESTDATA / "subject-mask.png")], "bg": "transparent",
         "output": str(out)},
        until={"seg_extracted", "error"})
    assert reply["type"] == "seg_extracted", "extracting needs only the image and the masks: %s" \
        % reply.get("message")
    alpha = Image.open(out).getchannel("A")
    assert alpha.getpixel(SUBJECT_PIXEL) == 255, "the masked object is kept"
    assert alpha.getpixel(BACKGROUND_PIXEL) == 0, "everything outside the mask is cut away"
