"""Run the worker with a batch that stops at one step, so a test can act there.

Usage: python held_batch_worker.py <src/bgbg> <step>

When a batch reaches <step>, the worker replies {"type": "held", "step": S}
first, so the test knows where it is. The steps:

- "model": the cut-out never finishes, as with a slow model on a big photo.
- "write": encoding a result takes half a second, then completes.
- "write-slow": encoding takes 3 s, as to a slow network share, then completes.
- "write-stuck": encoding never finishes, as on a share that stopped answering.
- "write-fails": encoding a result fails, as on a full disk.
"""
import errno
import sys
import threading
import time

sys.path.insert(0, sys.argv[1])
import worker  # noqa: E402  (prepares the process first, as the worker does)

STEP = sys.argv[2]
# How long each write step takes, well inside the worker's 5 s exit wait.
WRITE_S = {"write": 0.5, "write-slow": 3.0}


def reached():
    worker.out({"type": "held", "step": STEP})


def model_never_finishes(*_args, **_kwargs):
    reached()
    threading.Event().wait()


real_save = worker.save_image


def slow_save(im, dst):
    reached()
    time.sleep(WRITE_S[STEP])
    real_save(im, dst)


def stuck_save(_im, _dst):
    reached()
    threading.Event().wait()


def failing_save(_im, dst):
    reached()
    raise OSError(errno.ENOSPC, "No space left on device", dst)


if STEP == "model":
    worker._remove_resilient = model_never_finishes
elif STEP in WRITE_S:
    worker.save_image = slow_save
elif STEP == "write-stuck":
    worker.save_image = stuck_save
elif STEP == "write-fails":
    worker.save_image = failing_save
else:
    sys.exit("unknown step %r" % STEP)
worker.main()
