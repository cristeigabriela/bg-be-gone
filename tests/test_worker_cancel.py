"""A cancel names the request it is for, and stops only that one."""
from workerpipe import TESTDATA


def single(tmp_path, name):
    return {"op": "single", "model": "u2netp", "input": str(TESTDATA / "subject.png"),
            "output": str(tmp_path / name)}


def replies_for(worker, ids):
    """Read replies until every id in `ids` has ended; return {id: last reply}."""
    ended = {}
    while set(ended) != set(ids):
        reply = worker.next_reply()
        if reply.get("type") in ("done_single", "canceled", "error") and reply.get("id") in ids:
            ended[reply["id"]] = reply
    return ended


def test_a_cancel_sent_right_after_its_request_is_not_lost(worker, tmp_path):
    """The worker cleared its one cancel flag when it started a request, so a
    cancel that arrived first was forgotten and the request ran anyway."""
    rid = worker.send(single(tmp_path, "out.png"))
    worker.write({"op": "cancel", "id": rid})
    ended = replies_for(worker, [rid])
    assert ended[rid]["type"] == "canceled", "the request was cancelled before it ran"


def test_cancelling_a_queued_request_skips_it_and_spares_the_running_one(worker, tmp_path):
    """The flag hit whichever request ran; the queued one went on to write its file."""
    first = worker.send(single(tmp_path, "first.png"))
    queued = worker.send(single(tmp_path, "queued.png"))
    worker.write({"op": "cancel", "id": queued})
    ended = replies_for(worker, [first, queued])
    assert ended[first]["type"] == "done_single", "the running request was not the one cancelled"
    assert ended[queued]["type"] == "canceled", "the queued request is skipped"
    assert not (tmp_path / "queued.png").exists(), "a skipped request writes nothing"


def test_a_cancel_for_a_request_that_already_ended_is_dropped(worker, tmp_path):
    """A cancel that crossed its request's last reply was kept for good: ids
    only grow, so no later request ever took it back out."""
    rid = worker.send(single(tmp_path, "out.png"))
    assert replies_for(worker, [rid])[rid]["type"] == "done_single"
    worker.write({"op": "cancel", "id": rid})
    later = worker.send(single(tmp_path, "later.png"))   # read after the cancel
    assert replies_for(worker, [later])[later]["type"] == "done_single", \
        "a late cancel touches no other request"
    assert f"cancel for request {rid} dropped" in worker.log_text(), \
        "the worker drops a cancel for a request it no longer has"
