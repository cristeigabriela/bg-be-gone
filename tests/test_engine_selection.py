"""Which engine the settings pick: the app setting, BGBG_ENGINE, then the profile."""
from pathlib import Path

import pytest

from engines.selection import (
    Engine,
    Source,
    UnknownEngineError,
    UnreadableSettingError,
    resolve,
)

ENGINE_DATA = Path(__file__).parent / "testdata" / "engine"


@pytest.fixture
def layers(tmp_path):
    """resolve() over fixture files by name; None stands for a file that isn't there."""
    def run(setting=None, env="", profile=None):
        def path(name):
            return str(ENGINE_DATA / name if name else tmp_path / "absent.json")
        return resolve(settings_path=path(setting), environ={"BGBG_ENGINE": env},
                       profile_path=path(profile))
    return run


@pytest.mark.parametrize(
    "profile, expected",
    [
        # What a ROCm install will write: the variant and gfx keys are not the engine's business.
        ("profile-torch.json", Engine.TORCH),
        ("profile-ort.json", Engine.ORT),
    ],
    ids=["torch", "ort"],
)
def test_the_profile_picks_the_engine_when_nothing_else_does(layers, profile, expected):
    choice = layers(profile=profile)
    assert (choice.engine, choice.source) == (expected, Source.PROFILE), \
        "with no setting and no BGBG_ENGINE, the installer's profile decides"
    assert choice.problems == (), "a valid profile is not a problem"


@pytest.mark.parametrize(
    "profile",
    [
        # Every install and AppImage today.
        None,
        # A profile that only records the variant.
        "profile-no-engine.json",
    ],
    ids=["no profile", "no engine key"],
)
def test_without_an_engine_anywhere_it_is_onnx_runtime(layers, profile):
    choice = layers(profile=profile)
    assert (choice.engine, choice.source) == (Engine.ORT, Source.DEFAULT), \
        "nothing names an engine, so the worker keeps the engine it has always used"
    assert choice.problems == (), "an absent profile or key is normal, not a problem to log"


@pytest.mark.parametrize(
    "profile, problem",
    [
        # Not JSON at all.
        ("profile-not-json.json", UnreadableSettingError),
        # JSON, but not an object.
        ("profile-list.json", UnreadableSettingError),
        # Bytes that aren't UTF-8.
        ("profile-not-utf8.json", UnreadableSettingError),
        # An engine this build doesn't know.
        ("profile-unknown.json", UnknownEngineError),
        # An engine that isn't a string.
        ("profile-number.json", UnknownEngineError),
    ],
    ids=["not json", "not an object", "not utf-8", "unknown engine", "engine not a string"],
)
def test_a_broken_profile_means_onnx_runtime_and_is_reported(layers, profile, problem):
    choice = layers(profile=profile)
    assert (choice.engine, choice.source) == (Engine.ORT, Source.DEFAULT), \
        "a profile that can't be used must not stop the worker from starting"
    assert [type(p) for p in choice.problems] == [problem], \
        "the worker logs why the profile was skipped"
    assert choice.problems[0].source is Source.PROFILE, "the problem names the profile"


@pytest.mark.parametrize(
    "setting, env, profile, expected",
    [
        # The in-app setting beats the environment...
        ("settings-torch.json", "ort", None, (Engine.TORCH, Source.SETTING)),
        # ...and the profile.
        ("settings-ort.json", "", "profile-torch.json", (Engine.ORT, Source.SETTING)),
        # "auto" in the app hands the choice to the environment.
        ("settings-auto.json", "torch", "profile-ort.json", (Engine.TORCH, Source.ENV)),
        # The environment beats the profile.
        (None, "ort", "profile-torch.json", (Engine.ORT, Source.ENV)),
        # "auto" in the environment hands it to the profile.
        (None, "auto", "profile-torch.json", (Engine.TORCH, Source.PROFILE)),
        # The environment is read the way people type it.
        (None, " Torch ", None, (Engine.TORCH, Source.ENV)),
    ],
    ids=["setting beats env", "setting beats profile", "auto setting defers to env",
         "env beats profile", "auto env defers to profile", "env ignores case and spaces"],
)
def test_the_first_layer_with_an_opinion_wins(layers, setting, env, profile, expected):
    choice = layers(setting=setting, env=env, profile=profile)
    assert (choice.engine, choice.source) == expected, \
        "precedence is the in-app setting, then BGBG_ENGINE, then the profile"


def test_json_nested_too_deep_to_parse_is_an_unreadable_profile(tmp_path):
    """json raises RecursionError here, not ValueError; it must not reach the worker."""
    deep = tmp_path / "profile.json"
    deep.write_text("[" * 100_000)  # generated: far past any recursion limit
    choice = resolve(settings_path=str(tmp_path / "absent.json"), environ={},
                     profile_path=str(deep))
    assert choice.engine is Engine.ORT, "the worker still starts on ONNX Runtime"
    assert [type(p) for p in choice.problems] == [UnreadableSettingError], \
        "and the profile is reported as unreadable"


def test_bgbg_profile_points_at_another_profile(tmp_path):
    """Packagers and tests can name the profile, as BGBG_GPU_ENV names gpu.env."""
    choice = resolve(settings_path=str(tmp_path / "absent.json"),
                     environ={"BGBG_PROFILE": str(ENGINE_DATA / "profile-torch.json")})
    assert (choice.engine, choice.source) == (Engine.TORCH, Source.PROFILE), \
        "the profile BGBG_PROFILE names is the one read"


def test_a_broken_layer_hands_the_choice_to_the_next_one(layers):
    choice = layers(env="cuda", profile="profile-torch.json")
    assert (choice.engine, choice.source) == (Engine.TORCH, Source.PROFILE), \
        "an unusable BGBG_ENGINE has no opinion, so the profile still decides"
    assert [p.source for p in choice.problems] == [Source.ENV], "and the bad value is reported"
