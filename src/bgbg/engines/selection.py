"""Which engine runs background removal, from the settings that can name one.

The first layer with an opinion wins:

1. the in-app setting: ``engine`` in ``settings.json`` in the config folder
2. the ``BGBG_ENGINE`` environment variable
3. the installer's profile: ``engine`` in ``profile.json`` in the worker
   interpreter's ``sys.prefix`` (the venv on Linux). ``BGBG_PROFILE`` names
   another file instead, as ``BGBG_GPU_ENV`` does for ``gpu.env``.
4. ONNX Runtime

Each layer says ``auto``, ``ort`` or ``torch``. ``auto``, or a missing file,
key or variable, defers to the next layer. So does a layer that can't be
read; the choice keeps its error for the worker to log.

Standard library only, so the frontend can import it to offer the setting.
"""
from __future__ import annotations

import enum
import json
import os
import sys
from dataclasses import dataclass

import plat
from engines import EngineError

ENV_VAR = "BGBG_ENGINE"
PROFILE_ENV_VAR = "BGBG_PROFILE"
# The key both JSON files name the engine under.
_ENGINE_KEY = "engine"


class Engine(enum.Enum):
    AUTO = "auto"      # no opinion: the next layer decides
    ORT = "ort"        # ONNX Runtime, through rembg
    TORCH = "torch"    # PyTorch, for BiRefNet


class Source(enum.Enum):
    """Where a choice came from, worded to go in a message."""

    SETTING = "the engine setting"
    ENV = ENV_VAR
    PROFILE = "the install profile"
    DEFAULT = "the default"


class EngineSettingError(EngineError):
    """A layer names an engine in a way that can't be used, so it is skipped."""

    def __init__(self, source: Source, message: str):
        super().__init__(message)
        self.source = source


class UnknownEngineError(EngineSettingError):
    def __init__(self, source: Source, value: object):
        super().__init__(source, f"{source.value} names an unknown engine {value!r} "
                                 "(expected auto, ort or torch)")
        self.value = value


class UnreadableSettingError(EngineSettingError):
    def __init__(self, source: Source, path: str, reason: str):
        super().__init__(source, f"can't read {source.value} at {path}: {reason}")
        self.path = path
        self.reason = reason


@dataclass(frozen=True)
class Choice:
    engine: Engine                # ORT or TORCH, never AUTO
    source: Source                # the layer that decided
    problems: tuple[EngineSettingError, ...] = ()   # layers skipped on the way


def _parse(source: Source, value: object) -> Engine:
    if not isinstance(value, str):
        raise UnknownEngineError(source, value)
    try:
        return Engine(value.strip().lower())
    except ValueError as err:
        raise UnknownEngineError(source, value) from err


def _from_file(source: Source, path: str) -> Engine:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return Engine.AUTO
    # ValueError covers bad JSON and bytes that aren't UTF-8; RecursionError,
    # JSON nested deeper than the parser can follow.
    except (OSError, ValueError, RecursionError) as err:
        raise UnreadableSettingError(source, path, str(err)) from err
    if not isinstance(data, dict):
        raise UnreadableSettingError(source, path, "not a JSON object")
    if _ENGINE_KEY not in data:
        return Engine.AUTO
    return _parse(source, data[_ENGINE_KEY])


def _from_env(environ) -> Engine:
    value = environ.get(ENV_VAR, "")
    if not value.strip():
        return Engine.AUTO
    return _parse(Source.ENV, value)


def default_settings_path() -> str:
    # Read when called: gpu.env can move the config folder at start-up.
    return os.path.join(plat.config_dir(), "settings.json")


def default_profile_path(environ=os.environ) -> str:
    return environ.get(PROFILE_ENV_VAR) or os.path.join(sys.prefix, "profile.json")


def resolve(settings_path: str | None = None, environ=os.environ,
            profile_path: str | None = None) -> Choice:
    """The engine the settings ask for, and which layer asked.

    Args:
        settings_path: the app's settings file; by default the config folder's.
        environ: the environment to read ``BGBG_ENGINE`` and ``BGBG_PROFILE`` from.
        profile_path: the installer's profile; by default the worker's.

    Returns:
        The first layer's choice that isn't ``auto``, or ONNX Runtime. Never
        raises: a layer that can't be read is in ``problems`` instead.
    """
    settings_path = settings_path or default_settings_path()
    profile_path = profile_path or default_profile_path(environ)
    layers = [
        (Source.SETTING, lambda: _from_file(Source.SETTING, settings_path)),
        (Source.ENV, lambda: _from_env(environ)),
        (Source.PROFILE, lambda: _from_file(Source.PROFILE, profile_path)),
    ]
    problems = []
    for source, read in layers:
        try:
            engine = read()
        except EngineSettingError as err:
            problems.append(err)
            continue
        if engine is not Engine.AUTO:
            return Choice(engine, source, tuple(problems))
    return Choice(Engine.ORT, Source.DEFAULT, tuple(problems))
