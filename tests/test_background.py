"""The blurred background: the same strength blurs any picture by the same
share of it, and the blur leaves the subject out."""
from pathlib import Path

import pytest
from PIL import Image, ImageFilter

from background import STRENGTH_DEFAULT, STRENGTH_MAX, STRENGTH_MIN, blur_behind, blur_radius

TESTDATA = Path(__file__).parent / "testdata"


@pytest.mark.parametrize(
    "size, expected",
    [
        # A small photo: 2% of its 512 px side.
        ((512, 512), 10.24),
        # A large landscape photo: 2% of its shorter, 3000 px side.
        ((4000, 3000), 60.0),
        # A tall strip: the shorter side decides, whichever way round.
        ((300, 4000), 6.0),
    ],
    ids=["512 px square", "4000x3000", "tall strip"],
)
def test_the_default_strength_blurs_by_a_share_of_the_shorter_side(size, expected):
    assert blur_radius(STRENGTH_DEFAULT, size) == pytest.approx(expected), \
        "the radius is the strength's share of the picture's shorter side"


def test_the_same_strength_gives_a_radius_in_proportion_to_the_picture():
    small = blur_radius(STRENGTH_DEFAULT, (512, 512))
    large = blur_radius(STRENGTH_DEFAULT, (4096, 4096))
    assert large / small == pytest.approx(8), \
        "a picture 8 times larger gets an 8 times larger blur, so both look alike"


def test_the_weakest_strength_still_blurs_a_tiny_picture():
    assert blur_radius(STRENGTH_MIN, (64, 64)) >= 1, \
        "a blurred background always blurs by at least a pixel"


@pytest.mark.parametrize(
    "x",
    [
        # 1 px outside the disc's edge (x = 140), where a plain blur is greyest.
        141,
        # 10 and 20 px out: a plain blur still greys these at full strength.
        150,
        160,
    ],
    ids=["1 px out", "10 px out", "20 px out"],
)
def test_the_background_next_to_the_subject_keeps_its_own_colour(x):
    """halo-*.png: a black disc on white, and its alpha (make_halo_fixtures.py)."""
    picture = Image.open(TESTDATA / "halo-picture.png")
    alpha = Image.open(TESTDATA / "halo-alpha.png")
    plain = picture.filter(ImageFilter.GaussianBlur(blur_radius(STRENGTH_MAX, picture.size)))
    assert min(plain.getpixel((x, 100))) < 230, "the fixture shows the halo a plain blur makes"
    behind = blur_behind(picture, alpha, STRENGTH_MAX)
    assert min(behind.getpixel((x, 100))) >= 250, \
        "the white next to the disc stays white: the disc isn't blurred into it"
