"""A batch: the files it runs on, what it writes, and how it ends.

It never writes over a file (clashing names get a number), and it never
leaves its own unfinished file behind.
"""
import shutil
import subprocess
import sys

import pytest
from PIL import Image

from formats import Alpha
from workerpipe import BACKGROUND_PIXEL, SUBJECT_PIXEL, TESTDATA, WORKER

_ONLY_BARE_TOKENS = ("The rename pattern can only use {name} and {n}, with nothing else in "
                     "the braces.")
# How long a worker may take to exit when no write holds it up.
_EXIT_S = 5
# ... and with a write stuck: the worker waits 5 s for it, then deletes its part.
_STUCK_EXIT_S = 10


def batch_request(indir, outdir, pattern="{name}_nobg", **options):
    return {"op": "batch", "model": "u2netp", "input_dir": str(indir),
            "output_dir": str(outdir), "pattern": pattern, **options}


def run_request(worker, request):
    reply, seen = worker.ask(request, until={"done_batch", "error"})
    assert reply["type"] == "done_batch", "the batch must finish: %s" % reply.get("message")
    return reply, seen


def run_batch(worker, indir, outdir, pattern):
    return run_request(worker, batch_request(indir, outdir, pattern))[0]


def add_image(folder, name, fixture="subject.png"):
    path = folder / name
    shutil.copyfile(TESTDATA / fixture, path)
    return path


def names(folder):
    return sorted(p.name for p in folder.iterdir())


@pytest.fixture
def folders(tmp_path):
    """An input folder, and an output folder that doesn't exist yet."""
    indir = tmp_path / "in"
    indir.mkdir()
    return indir, tmp_path / "out"


def test_the_same_folder_with_a_bare_name_pattern_keeps_every_source(worker, tmp_path):
    """The audit's worst case: input = output and `{name}` replaced each PNG
    with its own cut-out."""
    photos = tmp_path / "photos"
    photos.mkdir()
    source = add_image(photos, "a.png")
    original = source.read_bytes()
    reply = run_batch(worker, photos, photos, "{name}")
    assert source.read_bytes() == original, "the source must not be written over"
    assert names(photos) == ["a (2).png", "a.png"], \
        "the cut-out takes the next free number instead"
    assert reply["renamed"] == 1, "the reply counts the renamed output"


def test_an_existing_output_is_kept_and_the_new_one_numbered(worker, folders):
    indir, outdir = folders
    outdir.mkdir()
    add_image(indir, "a.png")
    earlier = outdir / "a_nobg.png"
    earlier.write_bytes(b"an earlier run's result")
    run_batch(worker, indir, outdir, "{name}_nobg")
    assert earlier.read_bytes() == b"an earlier run's result", "an existing file is never replaced"
    assert names(outdir) == ["a_nobg (2).png", "a_nobg.png"], "the new result is numbered"


def test_a_pattern_without_tokens_gives_every_file_its_own_output(worker, folders):
    """`out` used to write every image to the same out.png."""
    indir, outdir = folders
    add_image(indir, "a.png")
    add_image(indir, "b.png")
    run_batch(worker, indir, outdir, "out")
    assert names(outdir) == ["out (2).png", "out.png"], "each source keeps its own result"


def test_two_sources_with_the_same_stem_both_keep_their_result(worker, folders):
    """a.jpg and a.png both map to a_nobg.png."""
    indir, outdir = folders
    add_image(indir, "a.png")
    add_image(indir, "a.jpg", fixture="subject.jpg")
    run_batch(worker, indir, outdir, "{name}_nobg")
    assert names(outdir) == ["a_nobg (2).png", "a_nobg.png"], "neither result overwrites the other"


def test_an_output_name_never_holds_an_unfinished_file(worker, folders):
    """The name was claimed with an empty file before the image ran, so a sync
    client could swap in a real file there, which the result then overwrote."""
    indir, outdir = folders
    outdir.mkdir()
    add_image(indir, "a.png")
    add_image(indir, "b.png")
    worker.send({"op": "batch", "model": "u2netp", "input_dir": str(indir),
                 "output_dir": str(outdir), "pattern": "{name}_nobg"})
    unfinished = set()
    while True:
        # Looks between replies too: a name must never be visible half-done.
        unfinished |= {p.name for p in outdir.iterdir()
                       if not p.name.startswith(".") and p.stat().st_size == 0}
        reply = worker.reply_within(0.005)
        if reply is not None and reply.get("type") in ("done_batch", "error"):
            break
    assert reply["type"] == "done_batch", "the batch finishes: %s" % reply.get("message")
    assert unfinished == set(), "an output's name appears only with its finished result"
    assert names(outdir) == ["a_nobg.png", "b_nobg.png"], "no unfinished file is left behind"


def test_a_batch_cancelled_mid_run_leaves_no_unfinished_file(worker, folders):
    """"Stop and quit" shut the worker down mid-file, which left the file being
    written behind; the window now cancels first, and this is what it waits on."""
    indir, outdir = folders
    for name in ("a.png", "b.png", "c.png"):
        add_image(indir, name)
    rid = worker.send({"op": "batch", "model": "u2netp", "input_dir": str(indir),
                       "output_dir": str(outdir), "pattern": "{name}_nobg"})
    while True:
        reply = worker.next_reply()
        if reply.get("type") == "progress" and reply.get("name") == "b.png":
            worker.write({"op": "cancel", "id": rid})
        if reply.get("type") in ("canceled", "done_batch", "error"):
            break
    assert reply["type"] == "canceled", "the batch stops: %s" % reply
    assert [n for n in names(outdir) if n.startswith(".")] == [], "no unfinished file is left"
    assert "a_nobg.png" in names(outdir), "the images done before the cancel stay"


def test_names_that_differ_only_in_case_get_their_own_outputs(worker, folders):
    """On a case-insensitive disk (Windows, macOS) A_nobg.png and a_nobg.png are one file."""
    indir, outdir = folders
    add_image(indir, "A.png")
    add_image(indir, "a.jpg", fixture="subject.jpg")
    run_batch(worker, indir, outdir, "{name}_nobg")
    assert sorted(n.lower() for n in names(outdir)) == ["a_nobg (2).png", "a_nobg.png"], \
        "the two results never share a name, whatever the disk's case rules"


def test_one_unreadable_file_does_not_stop_the_rest(worker, folders):
    """The whole batch used to stop at the first file it couldn't read."""
    indir, outdir = folders
    add_image(indir, "a.png")
    add_image(indir, "b.png", fixture="not_an_image.webp")
    add_image(indir, "c.png")
    reply = run_batch(worker, indir, outdir, "{name}_nobg")
    assert names(outdir) == ["a_nobg.png", "c_nobg.png"], "the files around the bad one still run"
    assert reply["count"] == 2, "the summary counts only the files that were done"
    assert reply["failed"] == [
        {"name": "b.png", "message": "b.png isn't an image bg-be-gone can read."}], \
        "the summary names the file that failed, and why"
    assert "batch: " in worker.log_text(), "the failure's traceback goes to the log"


@pytest.mark.parametrize("fixture, kind", [
    # A GIF, still or not: batch would keep only its first frame.
    ("subject.gif", "it's a GIF"),
    # An animated WebP: the same, though a still WebP runs.
    ("animated.webp", "it's an animated WebP"),
], ids=["gif", "animated-webp"])
def test_a_skipped_animation_is_listed_with_its_reason(worker, folders, fixture, kind):
    """GIFs used to be left out without a word."""
    indir, outdir = folders
    add_image(indir, "a.png")
    name = "moving." + fixture.rsplit(".", 1)[1]
    add_image(indir, name, fixture=fixture)
    reply = run_batch(worker, indir, outdir, "{name}_nobg")
    assert reply["skipped"] == [{
        "name": name,
        "message": kind + "; open it on the Single page, which keeps every frame."}], \
        "the summary names the skipped file, and why"
    assert names(outdir) == ["a_nobg.png"], "the other images still run"


@pytest.mark.parametrize("fixture, name, kind", [
    # An animated PNG (make_apng_fixture.py): a batch would write its first frame alone.
    ("animated.png", "walk.png", "PNG"),
    # An animated WebP saved as .png: what's inside decides, not the name.
    ("animated.webp", "moving.png", "WebP"),
], ids=["apng", "webp-named-png"])
def test_an_animation_is_skipped_whatever_its_name_says(worker, folders, fixture, name, kind):
    """Only WebP was checked for frames, so an animated PNG came out a still."""
    indir, outdir = folders
    add_image(indir, "a.png")
    add_image(indir, name, fixture=fixture)
    reply = run_batch(worker, indir, outdir, "{name}_nobg")
    # A .png opens as a still on the Single page too, so it isn't sent there.
    assert reply["skipped"] == [{
        "name": name,
        "message": "it's an animated %s; a batch would keep only its first frame." % kind}], \
        "the summary names the skipped file, and why"
    assert names(outdir) == ["a_nobg.png"], "the other images still run"


@pytest.mark.parametrize("pattern, message", [
    # A token that doesn't exist: refused, never quietly replaced by "{name}_nobg".
    ("{foo}", "The rename pattern uses {foo}, which isn't a token. Use {name} or {n}."),
    # A typo'd brace, which str.format raises on: refused the same way.
    ("{name", "The rename pattern has a { or } without its pair."),
    # Would write outside the output folder.
    ("../{name}", "File names can't contain /, so the rename pattern can't either."),
    # A format spec: "{n:c}" made item 47 "/", so its output left the folder.
    ("{n:c}outside", _ONLY_BARE_TOKENS),
    # A nested field, which str.format raises KeyError on, out of the UI callback.
    ("{name:{missing}}", _ONLY_BARE_TOKENS),
    # A conversion is the other way to put more than a name in the braces.
    ("{name!r}", _ONLY_BARE_TOKENS),
], ids=["unknown-token", "unpaired-brace", "folder-separator", "format-spec",
        "nested-field", "conversion"])
def test_a_bad_pattern_is_refused_before_anything_is_written(worker, folders, pattern, message):
    indir, outdir = folders
    add_image(indir, "a.png")
    reply, _ = worker.ask(
        {"op": "batch", "model": "u2netp", "input_dir": str(indir),
         "output_dir": str(outdir), "pattern": pattern},
        until={"done_batch", "error"})
    assert reply["type"] == "error", "a pattern that can't name files must not run"
    assert reply["message"] == message, "the error says what is wrong with the pattern"
    assert not outdir.exists(), "nothing is written, not even the output folder"


def test_a_file_that_appears_during_the_run_is_not_overwritten(worker, folders):
    """Names were checked before the run, then written blindly: a file copied
    into the output folder meanwhile was replaced."""
    indir, outdir = folders
    outdir.mkdir()
    add_image(indir, "a.png")
    add_image(indir, "b.png")
    late = outdir / "b_nobg.png"
    worker.send({"op": "batch", "model": "u2netp", "input_dir": str(indir),
                 "output_dir": str(outdir), "pattern": "{name}_nobg"})
    while True:
        reply = worker.next_reply()
        if reply.get("type") == "progress" and reply.get("name") == "a.png":
            late.write_bytes(b"copied in while a.png was processed")
        if reply.get("type") in ("done_batch", "error"):
            break
    assert reply["type"] == "done_batch", "the batch finishes: %s" % reply.get("message")
    assert late.read_bytes() == b"copied in while a.png was processed", \
        "a file that appeared after the run started is still not overwritten"
    assert names(outdir) == ["a_nobg.png", "b_nobg (2).png", "b_nobg.png"], \
        "b's result takes the next free number instead"


# --- The files a batch is given ------------------------------------------------

def test_only_the_files_left_selected_are_processed(worker, folders):
    """The gallery sends the images still selected; the rest of the folder is left alone."""
    indir, outdir = folders
    a, b, c = (add_image(indir, name) for name in ("a.png", "b.png", "c.png"))
    original = b.read_bytes()
    reply, _ = run_request(worker, batch_request(indir, outdir, files=[str(a), str(c)]))
    assert names(outdir) == ["a_nobg.png", "c_nobg.png"], "only the selected images get a result"
    assert reply["total"] == 2, "the summary counts the selected images, not the folder"
    assert b.read_bytes() == original, "an unselected image is never touched"


def test_every_image_reports_how_it_ended(worker, folders):
    """Each gallery tile shows its own image's state, from these replies."""
    indir, outdir = folders
    files = [add_image(indir, "a.png"), add_image(indir, "b.png", fixture="not_an_image.webp"),
             add_image(indir, "moving.gif", fixture="subject.gif"), add_image(indir, "c.png")]
    _, seen = run_request(worker, batch_request(indir, outdir, files=[str(f) for f in files]))
    states = {}
    for reply in seen:
        if reply.get("type") == "batch_file":
            states.setdefault(reply["name"], []).append(reply["state"])
    assert states == {
        "moving.gif": ["skipped"],
        "a.png": ["working", "done"],
        "b.png": ["working", "failed"],
        "c.png": ["working", "done"],
    }, "every image starts, if it can run, and ends exactly once"
    done = [r for r in seen if r.get("type") == "batch_file" and r["state"] == "done"]
    want = [str(outdir / "a_nobg.png"), str(outdir / "c_nobg.png")]
    assert [r["output"] for r in done] == want, "a finished image names the file its result went to"


def test_a_finished_image_gets_a_small_thumbnail_to_flip_to(worker, folders, tmp_path):
    indir, outdir = folders
    add_image(indir, "a.png")
    _, seen = run_request(worker, batch_request(indir, outdir, thumbs=str(tmp_path / "thumbs")))
    (done,) = [r for r in seen if r.get("type") == "batch_file" and r["state"] == "done"]
    thumb = Image.open(done["thumb"])
    assert thumb.format == "PNG" and max(thumb.size) <= 320, "a thumbnail is a small PNG"
    assert thumb.mode == "RGB", "a transparent result is shown over a checkerboard, as in Single"


# --- What a batch writes --------------------------------------------------------

@pytest.mark.parametrize("key, bg, written_as, alpha, background_colour", [
    # The default: a cut-out with its transparency.
    ("png", "transparent", "PNG", Alpha.TRANSPARENT, None),
    # WebP keeps the transparency too.
    ("webp", "transparent", "WEBP", Alpha.TRANSPARENT, None),
    # JPEG only once the background is flattened, here onto white.
    ("jpeg", "#ffffff", "JPEG", Alpha.FLATTENED, (255, 255, 255)),
    # Blur keeps the photo, so the background is no flat colour to check.
    ("png", "blur", "PNG", Alpha.FLATTENED, None),
    # A colour, in a lossy format: close to it, not exact.
    ("webp", "#00b140", "WEBP", Alpha.FLATTENED, (0, 177, 64)),
], ids=["png-transparent", "webp-transparent", "jpeg-white", "png-blur", "webp-colour"])
def test_each_format_and_background_gives_its_own_kind_of_file(
        worker, folders, key, bg, written_as, alpha, background_colour):
    indir, outdir = folders
    add_image(indir, "a.png")
    run_request(worker, batch_request(indir, outdir, format=key, bg=bg, blur=8))
    # Spelled out, not read from formats.py, so a changed suffix fails here.
    suffix = {"png": ".png", "webp": ".webp", "jpeg": ".jpg"}[key]
    assert names(outdir) == ["a_nobg" + suffix], "the result is named in its format"
    result = Image.open(outdir / ("a_nobg" + suffix))
    assert result.format == written_as, "the file is the format asked for, not just named it"
    rgba = result.convert("RGBA")
    if alpha is Alpha.TRANSPARENT:
        assert rgba.getchannel("A").getpixel(SUBJECT_PIXEL) > 224, "the subject stays opaque"
        assert rgba.getchannel("A").getpixel(BACKGROUND_PIXEL) < 32, "the background is cut away"
    else:
        assert rgba.getchannel("A").getextrema() == (255, 255), "a flattened result is opaque"
        assert rgba.convert("L").getpixel(SUBJECT_PIXEL) < 100, "the dark subject is kept on top"
    if background_colour is not None:
        got = rgba.getpixel(BACKGROUND_PIXEL)[:3]
        assert all(abs(g - w) <= 16 for g, w in zip(got, background_colour)), \
            "the background is the colour asked for: %s" % (got,)


def test_a_transparent_batch_is_never_written_as_jpeg(worker, folders):
    """JPEG has no alpha channel: the cut-out would come out on black."""
    indir, outdir = folders
    add_image(indir, "a.png")
    reply, _ = worker.ask(batch_request(indir, outdir, format="jpeg", bg="transparent"),
                          until={"done_batch", "error"})
    assert reply["type"] == "error", "the run is refused"
    assert reply["message"] == ("JPEG can't keep a transparent background. Choose PNG or WebP, "
                                "or a background to flatten onto."), "the error says what to pick"
    assert not outdir.exists(), "nothing is written, not even the output folder"


def test_a_result_in_another_format_still_never_overwrites(worker, folders):
    indir, outdir = folders
    outdir.mkdir()
    add_image(indir, "a.png")
    earlier = outdir / "a_nobg.webp"
    earlier.write_bytes(b"an earlier run's result")
    run_request(worker, batch_request(indir, outdir, format="webp"))
    assert earlier.read_bytes() == b"an earlier run's result", "an existing file is never replaced"
    assert names(outdir) == ["a_nobg (2).webp", "a_nobg.webp"], "the new result is numbered"


@pytest.mark.parametrize("orientation", [
    # Stored turned a quarter, sides swapped on disk, as phones write portraits.
    6,
    # Stored mirrored: the same size, so only the pixels tell.
    2,
], ids=["turned", "mirrored"])
@pytest.mark.parametrize("bg", [
    # The cut-out alone.
    "transparent",
    # The blur, which reads the picture around the subject.
    "blur",
], ids=["transparent", "blur"])
def test_a_batch_writes_a_photo_as_generate_does_upright(worker, folders, tmp_path,
                                                         orientation, bg):
    """oriented-N.jpg (make_orientation_fixtures.py): the gallery shows it upright,
    so the result a tile flips to must be upright too, and Generate's own file."""
    indir, outdir = folders
    add_image(indir, "a.jpg", fixture=f"oriented-{orientation}.jpg")
    generated = tmp_path / "generated.png"
    reply, _ = worker.ask(
        {"op": "single", "model": "u2netp", "input": str(indir / "a.jpg"),
         "output": str(generated), "bg": bg},
        until={"done_single", "error"})
    assert reply["type"] == "done_single", reply.get("message")
    run_request(worker, batch_request(indir, outdir, bg=bg))
    batch, single = Image.open(outdir / "a_nobg.png"), Image.open(generated)
    assert batch.size == (256, 192), "the result is the upright picture's size"
    assert (batch.mode, batch.tobytes()) == (single.mode, single.tobytes()), \
        "a batch writes, pixel for pixel, what Generate writes"


# --- Unfinished files -------------------------------------------------------------

@pytest.fixture
def held_worker(start_worker):
    """A worker whose batch stops at `step` (see testdata/held_batch_worker.py)."""
    def start(step):
        command = [sys.executable, str(TESTDATA / "held_batch_worker.py"), str(WORKER.parent),
                   step]
        return start_worker(None, command=command)
    return start


def run_until_held(worker, request):
    """Send a batch and read its replies until the worker is held at its step."""
    worker.send(request)
    while True:
        reply = worker.next_reply()
        assert reply["type"] not in ("done_batch", "error"), "the batch ended early: %s" % reply
        if reply["type"] == "held":
            return


def hidden(folder):
    """The batch's own part files: a published result is never hidden."""
    return [n for n in names(folder) if n.startswith(".")] if folder.exists() else []


def test_a_worker_killed_during_the_cut_out_leaves_no_unfinished_file(held_worker, folders):
    """A crash, or a quit that gave up waiting, used to leave a 0-byte part file:
    it was made when the image started, long before its result existed."""
    indir, outdir = folders
    add_image(indir, "a.png")
    w = held_worker("model")
    run_until_held(w, batch_request(indir, outdir))
    w.proc.kill()
    w.proc.wait()
    assert hidden(outdir) == [], "no file exists until the result does"


def test_a_shutdown_during_the_cut_out_exits_at_once_and_leaves_nothing(held_worker, folders):
    indir, outdir = folders
    add_image(indir, "a.png")
    w = held_worker("model")
    run_until_held(w, batch_request(indir, outdir))
    w.write({"op": "shutdown"})
    try:
        w.proc.wait(timeout=_EXIT_S)
    except subprocess.TimeoutExpired:
        pytest.fail("the worker must exit while the model runs: an image can't be stopped midway")
    assert hidden(outdir) == [], "no file exists until the result does"


@pytest.mark.parametrize("step", [
    # Five times over: exiting while the worker thread still sent its last
    # replies aborted at shutdown in a third of the runs (stdout's lock).
    "write", "write", "write", "write", "write",
    # Longer than the 1.5 s the exit used to wait.
    "write-slow",
], ids=["write-1", "write-2", "write-3", "write-4", "write-5", "write-slow"])
def test_a_shutdown_during_a_write_lets_the_result_get_its_name(held_worker, folders, step):
    """The worker thread dies wherever it is at exit; mid-write, that was its part file."""
    indir, outdir = folders
    add_image(indir, "a.png")
    w = held_worker(step)
    run_until_held(w, batch_request(indir, outdir))
    w.write({"op": "shutdown"})
    code = w.proc.wait(timeout=_EXIT_S)
    assert names(outdir) == ["a_nobg.png"], \
        "the result being written is finished and named before the worker exits"
    assert w.last_replies()[-1] == {"type": "exiting", "part_left": False}, \
        "the worker says it's exiting, with nothing left behind, for the frontend to wait on"
    assert (code, "Fatal Python error" in w.log_text()) == (0, False), \
        "the worker exits cleanly, not by an abort at interpreter shutdown"


def test_a_write_stuck_past_the_exit_wait_has_its_part_deleted(held_worker, folders):
    """A write that never ends (a share that stopped answering) can't hold the exit."""
    indir, outdir = folders
    add_image(indir, "a.png")
    w = held_worker("write-stuck")
    run_until_held(w, batch_request(indir, outdir))
    w.write({"op": "shutdown"})
    w.proc.wait(timeout=_STUCK_EXIT_S)
    assert names(outdir) == [], "the part file is deleted, and nothing was published"
    assert w.last_replies()[-1] == {"type": "exiting", "part_left": False}, \
        "the worker says nothing was left behind"


def test_a_write_that_fails_leaves_no_unfinished_file(held_worker, folders):
    """The part file is the batch's own, made before the write: Pillow doesn't delete it."""
    indir, outdir = folders
    add_image(indir, "a.png")
    w = held_worker("write-fails")
    w.send(batch_request(indir, outdir))
    while True:
        reply = w.next_reply()
        if reply["type"] in ("done_batch", "error"):
            break
    assert reply["failed"] == [{"name": "a.png", "message": "Not enough disk space to write "
                                                            "the result."}], \
        "the image is reported as failed, and why"
    assert names(outdir) == [], "its part file is gone"
