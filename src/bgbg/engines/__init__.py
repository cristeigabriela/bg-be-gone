"""The engines that run the background-removal models.

``engines.ort`` runs them on ONNX Runtime through rembg, on every platform.
Each engine is a module of its own, so one the worker doesn't use is never
imported. This package itself imports nothing heavy, so the frontend can
use it too.
"""


class EngineError(Exception):
    """Base for every failure the engines package raises."""
