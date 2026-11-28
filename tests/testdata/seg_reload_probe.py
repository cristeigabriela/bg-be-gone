"""Load SAM for a second image the way the worker does, and report what the
first image's session held while the second was picked.

Usage: python seg_reload_probe.py <src/bgbg> <image>

The model loader is replaced by a stand-in that records, when it is asked
for a session, whether the worker still holds the previous one: free VRAM
is probed then, and an old session still holding its memory makes the probe
lower than it should be. Prints one JSON object.
"""
import json
import sys

sys.path.insert(0, sys.argv[1])
import worker  # noqa: E402

seen = {}


class StandInSession:
    family = "sam2"

    def provider(self):
        return "CPUExecutionProvider"


def pick(providers, gpu, want="auto", probe_image=None, on_step=None, on_progress=None):
    # What the worker still holds from the previous image, at the moment the
    # next model is picked (where the real code probes free VRAM).
    seen["old_session_held"] = worker._seg is not None and worker._seg["session"] is not None
    return worker.seg.SamPick(StandInSession(), "tiny", "manual", None)


worker.seg.auto_select_session = pick
worker.out = lambda reply: seen.setdefault("replies", []).append(reply["type"])
worker._seg = {"session": StandInSession(), "image": None, "input": None, "rung": "tiny",
               "size": (1, 1), "maskdir": None, "objects": [], "prev_low": None}
worker.handle_seg_load({"op": "seg_load", "input": sys.argv[2], "model": "tiny", "id": 1})
print(json.dumps(seen))
