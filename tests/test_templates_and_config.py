import pytest

from prompt_optimizer.config import Config, load_config
from prompt_optimizer.errors import ConfigError, TemplateNotFoundError
from prompt_optimizer.templates import (
    get_template,
    list_templates,
    template_for_intent,
    template_names,
)


def test_every_template_is_populated():
    for template in list_templates():
        assert template.role
        assert template.description
        assert template.output_format
        assert template.requirements


def test_get_template_is_case_insensitive():
    assert get_template("Code_Generation").name == "code_generation"


def test_unknown_template_raises():
    with pytest.raises(TemplateNotFoundError):
        get_template("does-not-exist")


@pytest.mark.parametrize(
    "intent, expected",
    [
        ("code", "code_generation"),
        ("creative", "creative_writing"),
        ("image", "image_generation"),
        ("extraction", "extraction"),
        ("nonsense", "general"),
    ],
)
def test_template_for_intent(intent, expected):
    assert template_for_intent(intent).name == expected


def test_template_names_match_library():
    assert template_names() == [t.name for t in list_templates()]


def test_config_validates_engine():
    with pytest.raises(ConfigError):
        Config(engine="banana")


def test_config_validates_temperature():
    with pytest.raises(ConfigError):
        Config(temperature=9.0)


def test_resolve_engine_without_key():
    assert Config(engine="auto", api_key=None).resolve_engine() == "heuristic"
    assert Config(engine="auto", api_key="k").resolve_engine() == "llm"
    assert Config(engine="heuristic", api_key="k").resolve_engine() == "heuristic"


def test_load_config_reads_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("PROMPT_OPTIMIZER_MODEL", "gemini-test")
    monkeypatch.setenv("PROMPT_OPTIMIZER_TEMPERATURE", "0.9")
    monkeypatch.setenv("PROMPT_OPTIMIZER_ENGINE", "heuristic")

    config = load_config()
    assert config.api_key == "test-key"
    assert config.model == "gemini-test"
    assert config.temperature == 0.9
    assert config.engine == "heuristic"


def test_explicit_arguments_beat_environment(monkeypatch):
    monkeypatch.setenv("PROMPT_OPTIMIZER_ENGINE", "llm")
    assert load_config(engine="heuristic").engine == "heuristic"


def test_blank_api_key_is_treated_as_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert load_config().has_api_key is False


def test_bad_temperature_env_raises(monkeypatch):
    monkeypatch.setenv("PROMPT_OPTIMIZER_TEMPERATURE", "hot")
    with pytest.raises(ConfigError):
        load_config()
