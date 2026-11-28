# /// script
# requires-python = ">=3.9"
# dependencies = ["pillow>=11.3"]
# ///
"""Write oriented-1.jpg … oriented-8.jpg, for tests/test_worker.py.

The same upright picture (a dark disc off-centre on light grey, wider than
tall) stored as a camera would for each EXIF orientation: the pixels turned
the other way, and the tag that turns them upright again.

Run with `uv run tests/testdata/make_orientation_fixtures.py`.
"""
from pathlib import Path

from PIL import ExifTags, Image, ImageDraw

HERE = Path(__file__).parent
SIZE = (256, 192)
DISC = (30, 46, 130, 146)          # off-centre, so a wrong turn shows

# EXIF orientation -> how to store the upright picture for it: the inverse of
# the turn that ImageOps.exif_transpose applies when reading it.
STORED = {
    1: None,
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_90,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_270,
}


def main():
    upright = Image.new("RGB", SIZE, (225, 225, 225))
    ImageDraw.Draw(upright).ellipse(DISC, fill=(40, 70, 120))
    for orientation, turn in STORED.items():
        stored = upright if turn is None else upright.transpose(turn)
        exif = Image.Exif()
        exif[ExifTags.Base.Orientation] = orientation
        stored.save(HERE / f"oriented-{orientation}.jpg", quality=95, exif=exif)


if __name__ == "__main__":
    main()
