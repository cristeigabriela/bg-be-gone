"""The Direct Composition start marker: after a start that never showed a
window, the next start uses Cairo once, and the one after tries the GPU again.

These drive the marker file only; nothing here starts GTK. A marker names
the process that wrote it by pid and start time, so "a start that has
exited" is a pid with a start time no running process has.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="Direct Composition is Windows-only")

if sys.platform == "win32":
    from plat import wincompose
    from plat.wincompose import Renderer

SRC = Path(__file__).resolve().parent.parent / "src" / "bgbg"
EXITED_START = f"{os.getpid()} 1"   # this pid, but not this process's start time


@pytest.fixture
def marker(tmp_path):
    return tmp_path / "data" / wincompose.START_MARKER


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("GDK_DEBUG", raising=False)
    monkeypatch.delenv("GSK_RENDERER", raising=False)


def plant(marker, text):
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(text, encoding="utf-8")


def test_a_first_start_asks_for_direct_composition(marker, clean_env):
    choice = wincompose.choose_renderer(str(marker))
    assert choice.renderer is Renderer.DCOMP, "no earlier start went wrong"
    assert "dcomp" in os.environ["GDK_DEBUG"].split(","), "GTK reads the opt-in from GDK_DEBUG"
    assert marker.read_text(encoding="utf-8") == wincompose._this_start(), \
        "the start is marked as this process's"


def test_a_start_that_never_showed_a_window_makes_the_next_one_use_cairo(
        marker, clean_env, monkeypatch):
    plant(marker, EXITED_START)
    monkeypatch.setenv("GDK_DEBUG", "misc,dcomp")
    monkeypatch.setenv("GSK_RENDERER", "gl")
    choice = wincompose.choose_renderer(str(marker))
    assert choice.renderer is Renderer.CAIRO, "a crashed dcomp start must not crash the next one"
    assert choice.reason, "the log says why it fell back"
    assert os.environ["GDK_DEBUG"] == "misc", "dcomp is off, the user's other flags stay"
    assert os.environ["GSK_RENDERER"] == "cairo", "GTK really renders with Cairo"
    assert not marker.exists(), "only one start falls back"


def test_the_start_after_a_cairo_fallback_tries_the_gpu_again(marker, clean_env):
    plant(marker, EXITED_START)
    wincompose.choose_renderer(str(marker))
    os.environ.pop("GSK_RENDERER")
    assert wincompose.choose_renderer(str(marker)).renderer is Renderer.DCOMP, \
        "a driver update may have fixed it, so the fallback lasts one start"


def test_a_start_still_in_flight_is_left_to_finish(marker, clean_env):
    """Two starts before the first frame: the second must not take the first's
    mark, or a crash of the first would go unnoticed."""
    plant(marker, wincompose._this_start())   # a running process wrote it
    choice = wincompose.choose_renderer(str(marker))
    assert choice.renderer is Renderer.DCOMP, "nothing went wrong yet"
    assert marker.read_text(encoding="utf-8") == wincompose._this_start(), "its mark stays"


def test_a_shown_window_clears_the_marker(marker, clean_env):
    wincompose.choose_renderer(str(marker))
    wincompose.start_confirmed(str(marker))
    assert not marker.exists(), "a start that reached its first frame went fine"


def test_another_process_s_marker_is_left_alone(marker, clean_env):
    plant(marker, EXITED_START)
    wincompose.start_confirmed(str(marker))
    assert marker.read_text(encoding="utf-8") == EXITED_START, \
        "only the start that wrote it clears it"


def test_a_start_that_dies_before_its_first_frame_keeps_its_marker(marker, tmp_path):
    """An error during start-up ends Python normally; that is no reason to
    trust dcomp again."""
    script = tmp_path / "dies.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "from plat import wincompose\n"
        f"wincompose.choose_renderer({str(marker)!r})\n"
        "raise SystemExit('renderer error during start-up')\n", encoding="utf-8")
    subprocess.run([sys.executable, str(script)], capture_output=True, timeout=60)
    assert marker.exists(), "the next start must fall back to Cairo"
