# /// script
# requires-python = ">=3.9"
# dependencies = ["pillow>=11.3"]
# ///
"""Write animated.png, for tests/test_batch.py and tests/test_formats.py.

A two-frame animated PNG (APNG): a disc that moves, on light grey. Its name
says PNG, and so does its signature; only its acTL chunk says it animates.

Run with `uv run tests/testdata/make_apng_fixture.py`.
"""
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent


def frame(x):
    im = Image.new("RGB", (64, 64), (225, 225, 225))
    ImageDraw.Draw(im).ellipse((x, 16, x + 32, 48), fill=(40, 70, 120))
    return im


def main():
    first, second = frame(4), frame(28)
    first.save(HERE / "animated.png", save_all=True, append_images=[second],
               duration=200, loop=0)


if __name__ == "__main__":
    main()
