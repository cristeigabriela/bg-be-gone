"""What differs between operating systems, behind one interface.

The rest of bg-be-gone imports this package, never a platform module, and
never checks ``os.name`` itself. Each platform module provides, for the worker:

- ``prepare_process()``: start-up work that must happen before onnxruntime is
  imported. It may re-exec the process.
- ``unusable_providers()``: onnxruntime providers this machine must not use.
- ``provider_options()``: options per provider name, e.g. DirectML's adapter.
- ``probe_vram()``: total and free VRAM of the GPU, or None.
- ``models_dir()`` and ``config_dir()``: where downloaded models and settings
  live. Read them after ``prepare_process()``, which can change them.
- ``kernel_cache_dir()``: where the PyTorch engine's GPU kernels are cached.

and for the frontend:

- ``default_worker_python()``: the worker interpreter when no launcher names one.
- ``worker_log_path()``: the file the worker's stderr goes to.
- ``subprocess_flags()`` and ``worker_env()``: Popen's ``creationflags`` and
  ``env`` for the worker (an ``env`` of None means inherit).
- ``frontend_starting()``: set-up that must happen before GTK initialises.
- ``frontend_started(app_id)``: platform set-up once GTK is up.
- ``frontend_window(window)``: called with the main window once presented.
- ``frontend_handed_off()``: this start passed its files to a bg-be-gone
  already running, and ends without a window.
"""
import os

if os.name == "nt":
    from .windows import (
        config_dir,
        default_worker_python,
        frontend_handed_off,
        frontend_started,
        frontend_starting,
        frontend_window,
        kernel_cache_dir,
        models_dir,
        prepare_process,
        probe_vram,
        provider_options,
        subprocess_flags,
        unusable_providers,
        worker_env,
        worker_log_path,
    )
else:
    from .linux import (
        config_dir,
        default_worker_python,
        frontend_handed_off,
        frontend_started,
        frontend_starting,
        frontend_window,
        kernel_cache_dir,
        models_dir,
        prepare_process,
        probe_vram,
        provider_options,
        subprocess_flags,
        unusable_providers,
        worker_env,
        worker_log_path,
    )

__all__ = ["config_dir", "default_worker_python", "frontend_handed_off", "frontend_started",
           "frontend_starting", "frontend_window", "kernel_cache_dir", "models_dir",
           "prepare_process", "probe_vram", "provider_options", "subprocess_flags",
           "unusable_providers", "worker_env", "worker_log_path"]
