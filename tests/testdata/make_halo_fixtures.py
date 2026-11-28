# /// script
# requires-python = ">=3.9"
# dependencies = ["pillow>=11.3"]
# ///
"""Write halo-picture.png and halo-alpha.png, for tests/test_background.py.

A black disc on white, and its exact alpha: the case where a plain blur of
the background darkens the white just outside the disc.

Run with `uv run tests/testdata/make_halo_fixtures.py`.
"""
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
SIZE = 200
DISC = (60, 60, 140, 140)        # a disc of radius 40 around the centre


def main():
    picture = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    ImageDraw.Draw(picture).ellipse(DISC, fill=(0, 0, 0))
    picture.save(HERE / "halo-picture.png")
    alpha = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(alpha).ellipse(DISC, fill=255)
    alpha.save(HERE / "halo-alpha.png")


if __name__ == "__main__":
    main()
