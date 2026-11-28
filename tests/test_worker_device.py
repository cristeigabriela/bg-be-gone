"""Every finished job says where it ran, so the device label can't go stale."""
from workerpipe import TESTDATA


def test_a_finished_cutout_reports_the_device_it_ran_on(worker, tmp_path):
    """After one CPU fallback the label said "CPU" for good: nothing sent the GPU again."""
    reply, _ = worker.ask(
        {"op": "single", "model": "u2netp", "input": str(TESTDATA / "subject.png"),
         "output": str(tmp_path / "out.png")},
        until={"done_single", "error"})
    assert reply["type"] == "done_single", "the cut-out succeeds: %s" % reply.get("message")
    assert reply["device"] == {"provider": "CPUExecutionProvider", "label": "CPU",
                               "gpu": False}, "the CPU test stack runs on the CPU, and says so"
