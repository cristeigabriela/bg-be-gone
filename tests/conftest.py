import sys
from pathlib import Path

import pytest

# The app's modules import each other by bare name (the worker runs as a script
# from src/bgbg), so the unit tests import them the same way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "bgbg"))

from workerpipe import Worker  # noqa: E402


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
