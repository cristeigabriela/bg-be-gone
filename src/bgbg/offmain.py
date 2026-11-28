"""Run blocking work off the GTK main thread.

Image encoding and decoding on the UI thread froze the window: re-encoding a 4K
photo to PNG before Generate took ~3 s, decoding a 4K result ~240 ms, decoding
Segment's object masks ~650 ms. run() does that work on a background thread and
hands the result back on the main thread.
"""
import threading

from gi.repository import GLib


def run(work, done):
    """Call work() on a background thread, then done(result) on the main thread.

    If work raises, done receives the exception instead of a result, so the
    caller decides how to report it. The reply is queued above redraw priority:
    at the default idle priority a running animation could starve it.
    """
    def thread():
        try:
            result = work()
        except Exception as e:  # noqa: BLE001 - handed to done to report
            result = e
        GLib.idle_add(_deliver, done, result, priority=GLib.PRIORITY_DEFAULT)

    threading.Thread(target=thread, daemon=True).start()


def _deliver(done, result):
    done(result)
    return False
