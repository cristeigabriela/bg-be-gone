"""What the worker tells the window, and the log, when a request fails."""
import pytest

from workerpipe import TESTDATA


def single(tmp_path, source):
    return {"op": "single", "model": "u2netp", "input": str(source),
            "output": str(tmp_path / "out.png")}


def test_a_failed_request_writes_its_traceback_to_the_log(worker, tmp_path):
    """The traceback used to go to the window, which dropped it, so the log had nothing."""
    worker.ask(single(tmp_path, TESTDATA / "not_an_image.webp"),
               until={"done_single", "error"})
    log = worker.log_text()
    assert "request single" in log, "the log says which request failed"
    assert "Traceback (most recent call last)" in log, "the log keeps the traceback"
    assert "UnidentifiedImageError" in log, "the log keeps the original exception"


@pytest.mark.parametrize("source, message", [
    # A file that isn't an image: our words, not Pillow's own
    # `cannot identify image file 'C:\…'`.
    (TESTDATA / "not_an_image.webp", "not_an_image.webp isn't an image bg-be-gone can read."),
    # A file that isn't there at all.
    (TESTDATA / "gone.png", "gone.png was not found."),
], ids=["not-an-image", "missing"])
def test_a_failed_input_is_named_in_plain_words(worker, tmp_path, source, message):
    reply, _ = worker.ask(single(tmp_path, source), until={"done_single", "error"})
    assert reply["type"] == "error", "the input can't produce a cut-out"
    assert reply["message"] == message, "the error names the file and says what is wrong"
