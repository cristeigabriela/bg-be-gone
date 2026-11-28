"""What differs between operating systems, behind one interface.

The rest of bg-be-gone imports this package, never a platform module, and
never checks ``os.name`` itself. Each platform module provides, for the worker:

- ``prepare_process()``: start-up work that must happen before onnxruntime is
  imported. It may re-exec the process.
- ``unusable_providers()``: onnxruntime providers this machine must not use.
- ``probe_vram()``: total and free VRAM of the GPU, or None.
- ``models_dir()`` and ``config_dir()``: where downloaded models and settings
  live. Read them after ``prepare_process()``, which can change them.

and for the frontend:

- ``default_worker_python()``: the worker interpreter when no launcher names one.
- ``worker_log_path()``: the file the worker's stderr goes to.
- ``subprocess_flags()`` and ``worker_env()``: Popen's ``creationflags`` and
  ``env`` for the worker (an ``env`` of None means inherit).
- ``frontend_started(app_id)``: platform set-up once GTK is up.
"""
import os

if os.name == "nt":
    from .windows import (
        config_dir,
        default_worker_python,
        frontend_started,
        models_dir,
        prepare_process,
        probe_vram,
        subprocess_flags,
        unusable_providers,
        worker_env,
        worker_log_path,
    )
else:
    from .linux import (
        config_dir,
        default_worker_python,
        frontend_started,
        models_dir,
        prepare_process,
        probe_vram,
        subprocess_flags,
        unusable_providers,
        worker_env,
        worker_log_path,
    )

__all__ = ["config_dir", "default_worker_python", "frontend_started", "models_dir",
           "prepare_process", "probe_vram", "subprocess_flags", "unusable_providers",
           "worker_env", "worker_log_path"]
