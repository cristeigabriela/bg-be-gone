"""A SAM model download stops on cancel and leaves no partial file behind.

The network is the boundary replaced here: urlopen returns chunks from memory.
"""
import io

import pytest

import segmentation as seg


class Stop(Exception):
    """What the worker's progress callback raises on a cancel."""


class ChunkedResponse(io.BytesIO):
    """A response with no Content-Length, as a chunked transfer sends it."""

    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_cancel_stops_a_download_of_unknown_size_and_removes_its_part(tmp_path,
                                                                        monkeypatch):
    """Without Content-Length the callback that checks the cancel never ran,
    and a cancelled download left its .part file."""
    monkeypatch.setattr(seg.urllib.request, "urlopen",
                        lambda url, timeout=None: ChunkedResponse(b"x" * (3 << 20)))
    calls = []

    def progress(done, total):
        calls.append(done)
        raise Stop()
    dst = tmp_path / "model.onnx"
    with pytest.raises(Stop):
        seg._download("https://example.invalid/model.onnx", str(dst), progress)
    assert calls == [1 << 20], "the cancel is checked after the first chunk"
    assert list(tmp_path.iterdir()) == [], "neither the model nor a partial file is left"


class ShortResponse(ChunkedResponse):
    """A response that promises 3 MB and ends cleanly after 1 MB, as a dropped
    connection can."""

    headers = {"Content-Length": str(3 << 20)}


def test_a_download_that_ends_early_is_an_error_and_publishes_nothing(tmp_path, monkeypatch):
    """The short file was published as the model, and its damage surfaced later
    as a load failure that stepped down to a smaller model."""
    monkeypatch.setattr(seg.urllib.request, "urlopen",
                        lambda url, timeout=None: ShortResponse(b"x" * (1 << 20)))
    dst = tmp_path / "model.zip"
    with pytest.raises(seg.SegDownloadIncompleteError) as raised:
        seg._download("https://example.invalid/model.zip", str(dst))
    assert (raised.value.done, raised.value.total) == (1 << 20, 3 << 20), \
        "the error says how much arrived of how much"
    assert list(tmp_path.iterdir()) == [], "neither the model nor a partial file is left"
