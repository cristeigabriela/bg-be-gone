"""A batch never writes over a file: clashing names get a number."""
import shutil

import pytest

from workerpipe import TESTDATA

_ONLY_BARE_TOKENS = ("The rename pattern can only use {name} and {n}, with nothing else in "
                     "the braces.")


def run_batch(worker, indir, outdir, pattern):
    reply, _ = worker.ask(
        {"op": "batch", "model": "u2netp", "input_dir": str(indir),
         "output_dir": str(outdir), "pattern": pattern},
        until={"done_batch", "error"})
    assert reply["type"] == "done_batch", "the batch must finish: %s" % reply.get("message")
    return reply


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
