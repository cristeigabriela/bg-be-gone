"""The user's remembered choices, in ``settings.json`` in the config folder.

Other parts of the app keep their own keys in the same file (the engine
choice, which engines/selection.py reads, for one), so every write reads the
file, changes only the keys it is given, and replaces the file in one step. A
file that can't be read is never overwritten: it may hold keys this code
doesn't know.

Standard library only, so the frontend and the worker can both import it.
"""
import json
import os
import tempfile
import threading

import plat

# Serializes read-modify-replace within this process, so two writers can't
# each publish a copy missing the other's key. Nothing locks across
# processes: only the frontend writes this file; the worker only reads it.
_WRITE_LOCK = threading.Lock()


def default_path():
    """settings.json in the config folder.

    Read when called: gpu.env can move the config folder at the worker's start-up.
    """
    return os.path.join(plat.config_dir(), "settings.json")


class SettingsError(Exception):
    """Base for failures reading or writing settings.json."""


class UnreadableSettingsError(SettingsError):
    def __init__(self, path, reason):
        super().__init__(f"can't read {path}: {reason}")
        self.path = path
        self.reason = reason


class UnwritableSettingsError(SettingsError):
    def __init__(self, path, reason):
        super().__init__(f"can't write {path}: {reason}")
        self.path = path
        self.reason = reason


def load(path=None):
    """Every key in the settings file; empty when there is no file yet.

    Raises:
        UnreadableSettingsError: the file exists but isn't a JSON object.
    """
    path = path or default_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    # ValueError covers both bad JSON and bytes that aren't UTF-8; RecursionError,
    # JSON nested deeper than the parser can follow.
    except (OSError, ValueError, RecursionError) as err:
        raise UnreadableSettingsError(path, str(err)) from err
    if not isinstance(data, dict):
        raise UnreadableSettingsError(path, "not a JSON object")
    return data


def update(path=None, **values):
    """Set `values` in the settings file, keeping every other key.

    Raises:
        UnreadableSettingsError: the file exists but can't be read; it is left alone.
        UnwritableSettingsError: the new file couldn't be written.
    """
    path = path or default_path()
    with _WRITE_LOCK:
        data = load(path)
        data.update(values)
        tmp = None
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # A temp file of this write's own, so no other writer truncates it.
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".settings.",
                                       suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except OSError as err:
            if tmp is not None and os.path.exists(tmp):
                os.remove(tmp)
            raise UnwritableSettingsError(path, str(err)) from err
