import sys
from pathlib import Path

import pytest

# The app's modules import each other by bare name (the worker runs as a script
# from src/bgbg), so the unit tests import them the same way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "bgbg"))

from workerpipe import Worker  # noqa: E402

# Where the app keeps its settings, logs and downloaded models: %LOCALAPPDATA%
# on Windows, the XDG folders elsewhere.
_APP_FOLDERS = ("LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_CACHE_HOME")


@pytest.fixture(autouse=True)
def _scratch_app_folders(tmp_path, monkeypatch):
    """Point the app's folders at the test's own, for it and the workers it starts.

    Otherwise a test reads this machine's real settings (an Engine choice
    turns into a start-up notice) and can write into an installed app's
    folders. A test that sets one of these itself still wins.
    """
    for name in _APP_FOLDERS:
        monkeypatch.setenv(name, str(tmp_path / "app-folders" / name.lower()))


@pytest.fixture
def worker(tmp_path):
    w = Worker(tmp_path / "worker.log")
    yield w
    w.close()


@pytest.fixture
def start_worker(tmp_path):
    """Start workers with a given environment; every one is shut down after the test."""
    started = []

    def start(env, command=None):
        w = Worker(tmp_path / f"worker-{len(started)}.log", env=env, command=command)
        started.append(w)
        return w

    yield start
    for w in started:
        w.close()
