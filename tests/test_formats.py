"""The shared format list, src/bgbg/formats.py, without the worker or GTK."""
import configparser
from pathlib import Path

import pytest

import formats
from formats import JPEG, PNG, WEBP, Alpha, SuffixChange

REPO = Path(__file__).resolve().parent.parent


def test_the_desktop_entry_offers_exactly_the_formats_the_app_reads():
    """Open with lists bg-be-gone for these; a missing one hides it, an extra one fails."""
    entry = configparser.ConfigParser(interpolation=None)
    entry.read(REPO / "data" / "io.github.cristeigabriela.BgBeGone.desktop", encoding="utf-8")
    advertised = {m for m in entry["Desktop Entry"]["MimeType"].split(";") if m}
    assert advertised == set(formats.MIME_TYPES)


TRANSPARENT = formats.save_formats(Alpha.TRANSPARENT)
FLATTENED = formats.save_formats(Alpha.FLATTENED)


@pytest.mark.parametrize("typed, choices, want_path, want_format, want_change", [
    # The suffix the person typed picks the format.
    ("cut.webp", TRANSPARENT, "cut.webp", WEBP, SuffixChange.KEPT),
    # Any case: Windows users type .WEBP as often as .webp.
    ("cut.WEBP", TRANSPARENT, "cut.WEBP", WEBP, SuffixChange.KEPT),
    # No suffix: the default's is added.
    ("cut", TRANSPARENT, "cut.png", PNG, SuffixChange.ADDED),
    # A dot that isn't an image suffix is part of the name.
    ("cut.v2", TRANSPARENT, "cut.v2.png", PNG, SuffixChange.ADDED),
    # The odd one out: JPEG would drop the transparency, so it becomes a PNG.
    ("cut.jpg", TRANSPARENT, "cut.png", PNG, SuffixChange.REPLACED),
    # Once flattened, JPEG is a choice, under either of its suffixes.
    ("cut.jpeg", FLATTENED, "cut.jpeg", JPEG, SuffixChange.KEPT),
], ids=["webp", "upper-case", "no-suffix", "dotted-name", "jpeg-on-transparent",
        "jpeg-on-flattened"])
def test_the_typed_name_picks_the_save_format_when_it_can_hold_the_result(
        typed, choices, want_path, want_format, want_change):
    target = formats.save_target(typed, choices, PNG)
    assert (target.path, target.format, target.change) == (want_path, want_format, want_change)


@pytest.mark.parametrize("typed", [
    # A suffix the result can't use, replaced: the dialog asked about cut.jpg.
    "cut.jpg",
    # No suffix, added: the dialog asked about cut, a file that doesn't exist.
    "cut",
], ids=["replaced-suffix", "added-suffix"])
def test_a_renamed_save_over_an_existing_file_needs_asking(tmp_path, typed):
    """The dialog confirmed the typed name; it never asked about the cut.png saved instead."""
    (tmp_path / "cut.png").write_bytes(b"")
    target = formats.save_target(str(tmp_path / typed), TRANSPARENT, PNG)
    assert target.path == str(tmp_path / "cut.png"), "the chosen format's suffix is in place"
    assert formats.overwrites_unasked(target), "cut.png exists and nobody agreed to replace it"


@pytest.mark.parametrize("typed, existing", [
    # The typed name itself: the dialog already asked about it.
    ("cut.png", "cut.png"),
    # Renamed, but onto a free name.
    ("cut.jpg", None),
], ids=["confirmed-by-the-dialog", "renamed-to-a-free-name"])
def test_a_save_the_dialog_covered_needs_no_asking(tmp_path, typed, existing):
    if existing:
        (tmp_path / existing).write_bytes(b"")
    target = formats.save_target(str(tmp_path / typed), TRANSPARENT, PNG)
    assert not formats.overwrites_unasked(target)


@pytest.mark.parametrize("source, choices, want", [
    # A WebP source comes back as a WebP.
    ("photo.webp", TRANSPARENT, WEBP),
    # A JPEG can't hold the cut-out's transparency: PNG.
    ("photo.jpg", TRANSPARENT, PNG),
    # ... but a flattened one is still a JPEG.
    ("photo.JPG", FLATTENED, JPEG),
    # Formats we read but don't write fall back to PNG.
    ("photo.heic", TRANSPARENT, PNG),
    # No source path yet.
    (None, TRANSPARENT, PNG),
], ids=["webp", "jpeg-transparent", "jpeg-flattened", "heic", "no-source"])
def test_the_default_save_format_follows_the_source_when_it_can(source, choices, want):
    assert formats.default_save_format(source, choices) is want


TESTDATA = Path(__file__).parent / "testdata"


@pytest.mark.parametrize("fixture, want", [
    # An animated WebP: its extended header sets the animation flag.
    ("animated.webp", formats.Animation.ANIMATED),
    # A still WebP runs in a batch.
    ("subject.webp", formats.Animation.STILL),
    # Named .webp but not one: a still, which the batch then reports it can't read.
    ("not_an_image.webp", formats.Animation.STILL),
], ids=["animated", "still", "not-a-webp"])
def test_the_gallery_tells_an_animated_webp_from_its_header(fixture, want):
    assert formats.webp_animation(str(TESTDATA / fixture)) is want
