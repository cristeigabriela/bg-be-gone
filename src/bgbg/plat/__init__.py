"""What differs between operating systems, behind one interface.

The rest of bg-be-gone imports this package, never a platform module, and
never checks ``os.name`` itself. Each platform module provides:

- ``prepare_process()``: start-up work that must happen before onnxruntime is
  imported. It may re-exec the process.
- ``unusable_providers()``: onnxruntime providers this machine must not use.
- ``probe_vram()``: total and free VRAM of the GPU, or None.
- ``models_dir()`` and ``config_dir()``: where downloaded models and settings
  live. Read them after ``prepare_process()``, which can change them.

NOTE: Windows has no module of its own yet, so it runs the Linux one, as it did
before this package existed. Its probes find nothing there and do nothing.
"""
from .linux import (
    config_dir,
    models_dir,
    prepare_process,
    probe_vram,
    unusable_providers,
)

__all__ = ["config_dir", "models_dir", "prepare_process", "probe_vram", "unusable_providers"]
