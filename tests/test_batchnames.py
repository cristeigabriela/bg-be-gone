"""Rename patterns: only bare tokens, and every expanded name stays a file name.

Publishing: a finished result takes a free name and never replaces a file.
"""
import enum

import pytest

import batchnames


class Disk(enum.Enum):
    HARD_LINKS = "hard links"          # NTFS, ext4 and the like
    NO_HARD_LINKS = "no hard links"    # FAT, exFAT, some network shares


@pytest.fixture(params=list(Disk), ids=[d.name.lower() for d in Disk])
def disk(request, monkeypatch):
    """Publishing on each kind of disk; one without hard links refuses os.link."""
    if request.param is Disk.NO_HARD_LINKS:
        def no_link(src, dst):
            raise PermissionError(1, "Operation not permitted", dst)
        monkeypatch.setattr(batchnames.os, "link", no_link)
    return request.param


def test_publishing_never_replaces_a_file_that_took_the_name(tmp_path, disk):
    """A file that appeared under the name while the result rendered is kept."""
    existing = tmp_path / "cat_nobg.png"
    existing.write_bytes(b"a file that appeared meanwhile")
    part = batchnames.new_part(str(tmp_path), "cat_nobg", ".png")
    with open(part, "wb") as f:
        f.write(b"the finished result")
    published = batchnames.publish(part, str(tmp_path), "cat_nobg", ".png", set())
    assert existing.read_bytes() == b"a file that appeared meanwhile", "the existing file is kept"
    assert published == str(tmp_path / "cat_nobg (2).png"), "the result takes the next number"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cat_nobg (2).png", "cat_nobg.png"], \
        "the unfinished part file is gone"


@pytest.mark.parametrize("pattern", [
    # "{n:c}" expands item 47 to "/", which os.path.join takes as the root.
    "{n:c}outside",
    # A nested field raised KeyError, not BatchPatternError, from the UI callback.
    "{name:{missing}}",
    # A zero-padded number reads well, but is still a format spec.
    "{n:03}",
], ids=["char-format", "nested-field", "padding"])
def test_anything_but_a_bare_token_is_refused_as_a_pattern_error(pattern):
    with pytest.raises(batchnames.BatchPatternError):
        batchnames.check_pattern(pattern)


@pytest.mark.parametrize("source", [
    # The parent folder's name: the check that runs before anything is joined.
    "..",
    # Nothing left to call the file.
    "",
], ids=["parent-folder", "empty"])
def test_an_expanded_name_that_is_no_plain_file_name_is_refused(source):
    with pytest.raises(batchnames.BatchPatternError):
        batchnames.stem_for("{name}", source, 1)


def test_a_bare_token_pattern_names_files():
    assert batchnames.stem_for("{name}_{n}", "/photos/cat.jpg", 7) == "cat_7", \
        "the supported tokens still fill in"
