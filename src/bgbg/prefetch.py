"""Download a background-removal model before first use.

Installers run this on the worker's interpreter, so the model is fetched by
the same code the worker loads it with, into the folder the worker reads:

    python prefetch.py onnx birefnet-general <U2NET_HOME>

"onnx" is rembg's own ONNX download, which verifies the file's hash and
prints progress. Exit status 0 means the model is in place; anything else is
safe to ignore, since the app downloads on first use.
"""
import enum
import os
import sys


class Engine(enum.Enum):
    ONNX = "onnx"


def _fetch_onnx(model, dest):
    # rembg reads U2NET_HOME when a session class asks for its folder.
    os.environ["U2NET_HOME"] = dest
    from rembg.sessions import sessions_class

    for cls in sessions_class:
        if cls.name() == model:
            return cls.download_models()
    raise SystemExit(f"rembg has no model named {model}")


_FETCHERS = {Engine.ONNX: _fetch_onnx}


def main(argv):
    if len(argv) != 4:
        raise SystemExit(__doc__)
    engine, model, dest = Engine(argv[1]), argv[2], argv[3]
    print(f"{model}: {_FETCHERS[engine](model, dest)}")


if __name__ == "__main__":
    main(sys.argv)
