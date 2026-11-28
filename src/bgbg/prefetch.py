"""Download a background-removal model before first use.

Installers run this on the worker's interpreter, so the model is fetched by
the same code the worker loads it with, into the folder the worker reads:

    python prefetch.py onnx birefnet-general <U2NET_HOME>
    python prefetch.py torch birefnet-general <BiRefNet checkpoint folder>

"onnx" is rembg's own ONNX download (any ONNX Runtime worker); "torch" is
BiRefNet's original checkpoint, which the PyTorch engine loads. Both verify
the file's hash and print progress. Exit status 0 means the model is in
place; anything else is safe to ignore, since the app downloads on first use.
"""
import enum
import os
import sys


class Engine(enum.Enum):
    ONNX = "onnx"
    TORCH = "torch"


def _fetch_onnx(model, dest):
    # rembg reads U2NET_HOME when a session class asks for its folder.
    os.environ["U2NET_HOME"] = dest
    from rembg.sessions import sessions_class

    for cls in sessions_class:
        if cls.name() == model:
            return cls.download_models()
    raise SystemExit(f"rembg has no model named {model}")


def _fetch_torch(model, dest):
    from engines import torch_birefnet   # needs no torch to download

    def progress(done, total):
        if total:
            print(f"\r{done >> 20} / {total >> 20} MB", end="", flush=True)

    path = torch_birefnet.ensure_checkpoint(model, dest, progress)
    print()
    return path


_FETCHERS = {Engine.ONNX: _fetch_onnx, Engine.TORCH: _fetch_torch}


def main(argv):
    if len(argv) != 4:
        raise SystemExit(__doc__)
    engine, model, dest = Engine(argv[1]), argv[2], argv[3]
    print(f"{model}: {_FETCHERS[engine](model, dest)}")


if __name__ == "__main__":
    main(sys.argv)
