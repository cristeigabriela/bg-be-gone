"""settings.json is shared, so a write must keep the keys it wasn't given."""
import json
import shutil
import threading

import pytest

import settings
from engines.selection import Engine, Source, resolve
from workerpipe import TESTDATA

FIXTURES = TESTDATA / "settings"


def test_an_update_keeps_the_engine_choice_the_worker_reads(tmp_path):
    """The engine choice lives in the same file; remembering a folder must not drop it."""
    path = tmp_path / "settings.json"
    shutil.copyfile(FIXTURES / "with-engine.json", path)
    settings.update(str(path), open_folder="/photos", subject="isnet-anime")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["subject"] == "isnet-anime", "a named key is replaced"
    assert saved["open_folder"] == "/photos", "a new key is added"
    choice = resolve(settings_path=str(path), environ={}, profile_path=str(tmp_path / "none"))
    assert (choice.engine, choice.source) == (Engine.TORCH, Source.SETTING), \
        "engine selection still finds the engine key the update didn't name"


def test_an_unreadable_file_is_left_alone(tmp_path):
    """It may hold keys this code doesn't know, so it must not be replaced by {}."""
    path = tmp_path / "settings.json"
    shutil.copyfile(FIXTURES / "broken.json", path)
    with pytest.raises(settings.UnreadableSettingsError):
        settings.update(str(path), open_folder="/photos")
    assert path.read_bytes() == (FIXTURES / "broken.json").read_bytes(), "the file is untouched"


def test_no_file_yet_reads_as_no_settings(tmp_path):
    assert settings.load(str(tmp_path / "settings.json")) == {}, "a first launch has no settings"


def test_concurrent_updates_keep_every_key(tmp_path):
    """Writers shared one .tmp file and didn't serialize read-modify-replace."""
    path = str(tmp_path / "settings.json")
    errors = []

    def writer(key):
        for i in range(40):
            try:
                settings.update(path, **{key: i})
            except settings.SettingsError as e:
                errors.append(e)
    threads = [threading.Thread(target=writer, args=(k,)) for k in ("width", "height")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], "no write fails"
    saved = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert saved == {"width": 39, "height": 39}, "neither writer undoes the other's key"
    assert [p.name for p in tmp_path.iterdir()] == ["settings.json"], "no temp file is left"
