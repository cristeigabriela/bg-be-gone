"""A model's first download is reported to the window, not only to the log.

Downloads rembg's smallest model (u2netp, ~4.5 MB) into an empty folder, so it
needs the network.
"""
import os

from workerpipe import TESTDATA


def test_the_first_use_of_a_model_reports_its_download(start_worker, tmp_path):
    """The status said "Loading…" through a ~1 GB download, with no size or progress."""
    worker = start_worker(dict(os.environ, U2NET_HOME=str(tmp_path / "models")))
    reply, seen = worker.ask(
        {"op": "single", "model": "u2netp", "input": str(TESTDATA / "subject.png"),
         "output": str(tmp_path / "out.png")},
        until={"done_single", "error"})
    assert reply["type"] == "done_single", "the cut-out still succeeds: %s" % reply.get("message")
    downloads = [r for r in seen if r.get("type") == "download"]
    assert downloads, "the download is reported while it runs"
    last = downloads[-1]
    assert last["model"] == "u2netp", "the report names the model"
    assert last["total"] > 1_000_000, "the report has the size in bytes"
    assert last["done"] == last["total"], "the last report is the finished download"


def test_a_cancel_stops_a_first_download_and_leaves_no_model(start_worker, tmp_path):
    """Cancel did nothing during a download; the user waited out ~1 GB."""
    models = tmp_path / "models"
    worker = start_worker(dict(os.environ, U2NET_HOME=str(models)))
    rid = worker.send({"op": "single", "model": "u2netp",
                       "input": str(TESTDATA / "subject.png"),
                       "output": str(tmp_path / "out.png")})
    while True:
        reply = worker.next_reply()
        if reply.get("type") == "download":
            worker.write({"op": "cancel", "id": rid})
        if reply.get("type") in ("done_single", "canceled", "error"):
            break
    assert reply["type"] == "canceled", "the download stops: %s" % reply
    assert list(models.iterdir()) == [], "neither the model nor a partial download is left"
