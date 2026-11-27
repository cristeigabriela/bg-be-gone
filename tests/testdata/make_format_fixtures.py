# /// script
# requires-python = ">=3.9"
# dependencies = ["pillow>=11.3", "pillow-heif"]
# ///
"""Write subject.png in every format the worker reads, for tests/test_formats.py.

Run with `uv run tests/testdata/make_format_fixtures.py`. pillow-heif is only
needed here, to encode the HEIC copy: the worker decodes HEIC with pi-heif,
which has no encoder.
"""
from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener

HERE = Path(__file__).parent


def main():
    register_heif_opener()
    subject = Image.open(HERE / "subject.png").convert("RGB")
    subject.save(HERE / "subject.jpg", quality=90)
    subject.save(HERE / "subject.webp", quality=90)
    subject.save(HERE / "subject.avif", quality=90)
    subject.save(HERE / "subject.heic", quality=90)
    subject.save(HERE / "subject.tif", compression="tiff_deflate")
    # Two colours: a palette BMP is a quarter the size of a 24-bit one.
    subject.quantize(colors=2).save(HERE / "subject.bmp")
    subject.save(HERE / "subject.gif")
    # 16-bit greyscale (mode I;16) over the full 0-65535 range, as scanners write it.
    grey16 = subject.convert("L").point(lambda v: v * 257, mode="I").convert("I;16")
    grey16.save(HERE / "subject16.tif", compression="tiff_deflate")

    # Three frames, each with its own duration, so a test can tell whether the
    # timing survives. The disc stays put so the same pixels can be checked.
    frames = [subject, subject.point(lambda v: min(255, v + 10)), subject]
    frames[0].save(HERE / "animated.webp", save_all=True, append_images=frames[1:],
                   duration=[80, 120, 160], loop=0, quality=90)

    (HERE / "not_an_image.webp").write_text("bg-be-gone test: this file is not an image\n")


if __name__ == "__main__":
    main()
