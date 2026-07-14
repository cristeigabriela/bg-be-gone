"""The heavy half: numpy, PIL, onnxruntime — everything the engine may not import.

This runs in the bundled virtualenv, in its own process, so the GTK shell can
keep running on the system Python with none of it installed. `service.py` is the
entry point: it speaks `engine.protocol` on stdin/stdout.

On the web this package's twin is a Web Worker running ONNX Runtime Web, speaking
the same protocol over postMessage. That is why the protocol is declared in the
engine and not here.
"""
