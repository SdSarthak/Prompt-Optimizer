"""Provider faults, batch resilience, config validation and CLI input errors."""

import json

import pytest

from prompt_optimizer.cli import main
from prompt_optimizer.config import Config, load_config
from prompt_optimizer.errors import ConfigError, ProviderError, PromptOptimizerError
from prompt_optimizer.gemini import (
    GeminiRewriter,
    clean_response,
    is_transient,
    response_text,
)
from prompt_optimizer.optimizer import PromptOptimizer
from prompt_optimizer.templates import get_template
from prompt_optimizer.analysis import analyze

ROUGH = "write a story"


def rewriter(client, **config_kwargs):
    """A rewriter whose backoff is recorded instead of actually slept."""
    slept = []
    config = Config(api_key="k", **config_kwargs)
    return GeminiRewriter(config, client=client, sleep=slept.append), slept


class ScriptedClient:
    """Returns/raises the next scripted item on every generate_content call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        outer = self

        class models:
            @staticmethod
            def generate_content(**kwargs):
                outer.calls += 1
                item = outer.script.pop(0) if outer.script else outer.script_default()
                if isinstance(item, Exception):
                    raise item
                return item

        self.models = models

    def script_default(self):
        raise AssertionError("client called more times than the script allows")


def reply(text):
    return type("Response", (), {"text": text})()


# --------------------------------------------------------------------------- #
# Retry / backoff
# --------------------------------------------------------------------------- #


def test_transient_failure_is_retried_then_succeeds():
    client = ScriptedClient([RuntimeError("503 Service Unavailable"), reply("# Task\nok")])
    rw, slept = rewriter(client)
    assert rw.rewrite(analyze(ROUGH), get_template("general")) == "# Task\nok"
    assert client.calls == 2
    assert slept == [1.0]


def test_retries_are_bounded_by_max_retries():
    client = ScriptedClient([RuntimeError("429 RESOURCE_EXHAUSTED")] * 3)
    rw, slept = rewriter(client, max_retries=2)
    with pytest.raises(ProviderError, match="after 3 attempts"):
        rw.rewrite(analyze(ROUGH), get_template("general"))
    assert client.calls == 3
    assert slept == [1.0, 2.0]  # exponential, not constant


def test_zero_retries_calls_once():
    client = ScriptedClient([RuntimeError("503 unavailable")])
    rw, slept = rewriter(client, max_retries=0)
    with pytest.raises(ProviderError):
        rw.rewrite(analyze(ROUGH), get_template("general"))
    assert client.calls == 1
    assert slept == []


def test_a_permanent_failure_is_not_retried():
    client = ScriptedClient([RuntimeError("API key not valid")])
    rw, slept = rewriter(client)
    with pytest.raises(ProviderError, match="API key not valid"):
        rw.rewrite(analyze(ROUGH), get_template("general"))
    assert client.calls == 1
    assert slept == []


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("429 Too Many Requests"),
        RuntimeError("503 Service Unavailable"),
        RuntimeError("deadline exceeded"),
        TimeoutError("read timed out"),
        ConnectionError("connection reset by peer"),
    ],
)
def test_transient_errors_are_recognised(exc):
    assert is_transient(exc)


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("invalid argument: model not found"),
        RuntimeError("permission denied"),
        RuntimeError("400 INVALID_ARGUMENT"),
    ],
)
def test_permanent_errors_are_not_retried(exc):
    assert not is_transient(exc)


def test_a_status_code_attribute_marks_a_transient_error():
    exc = RuntimeError("upstream said no")
    exc.code = 503
    assert is_transient(exc)


# --------------------------------------------------------------------------- #
# Reading the response
# --------------------------------------------------------------------------- #


def test_a_raising_text_property_becomes_a_provider_error():
    class Blocked:
        prompt_feedback = type("F", (), {"block_reason": "SAFETY"})()

        @property
        def text(self):
            raise ValueError("no Part in the response")

    with pytest.raises(ProviderError, match="SAFETY"):
        response_text(Blocked())


def test_a_truncated_candidate_is_reported():
    class Truncated:
        prompt_feedback = None
        candidates = [type("C", (), {"finish_reason": "MAX_TOKENS"})()]
        text = None

    with pytest.raises(ProviderError, match="MAX_TOKENS"):
        response_text(Truncated())


def test_a_plain_none_text_is_treated_as_empty():
    class Empty:
        text = None

    assert response_text(Empty()) == ""


def test_blocked_response_falls_back_instead_of_crashing():
    """The whole point of guarding .text: auto mode must still produce output."""

    class Blocked:
        prompt_feedback = type("F", (), {"block_reason": "SAFETY"})()

        @property
        def text(self):
            raise ValueError("no Part in the response")

    optimizer = PromptOptimizer(
        Config(engine="auto", api_key="k"),
        rewriter=GeminiRewriter(
            Config(api_key="k", max_retries=0),
            client=ScriptedClient([Blocked()]),
            sleep=lambda _: None,
        ),
    )
    result = optimizer.optimize(ROUGH)
    assert result.engine == "heuristic"
    assert "SAFETY" in result.fallback_reason


# --------------------------------------------------------------------------- #
# Response cleaning
# --------------------------------------------------------------------------- #


def test_lead_in_before_a_fence_is_stripped():
    assert clean_response("Here is the rewritten prompt:\n```\n# Task\ndo it\n```") == (
        "# Task\ndo it"
    )


def test_trailing_chatter_after_the_fence_is_dropped():
    assert clean_response("```markdown\n# Task\ndo it\n```\n\nHope that helps!") == (
        "# Task\ndo it"
    )


def test_a_nested_fence_survives():
    raw = "```markdown\n# Task\nReturn JSON:\n\n```json\n{\"a\": 1}\n```\n```"
    cleaned = clean_response(raw)
    assert cleaned.startswith("# Task")
    assert "```json" in cleaned
    assert '{"a": 1}' in cleaned


def test_an_unwrapped_answer_containing_a_fence_is_untouched():
    raw = "# Task\nWrite code.\n\n```python\nprint(1)\n```"
    assert clean_response(raw) == raw


def test_an_empty_fence_is_empty():
    assert clean_response("```\n```") == ""


def test_long_prose_after_a_fence_is_kept():
    raw = "```\n# Task\ndo it\n```\n\n" + ("explanation " * 40)
    assert raw.strip() == clean_response(raw)


# --------------------------------------------------------------------------- #
# Batch resilience
# --------------------------------------------------------------------------- #


class BrokenRewriter:
    def __init__(self, fail_on):
        self.fail_on = fail_on

    def rewrite(self, analysis, template, target_model="general"):
        if self.fail_on in analysis.prompt:
            raise ProviderError("provider is down")
        return "# Task\n" + analysis.prompt


def test_batch_keeps_going_when_one_prompt_fails():
    optimizer = PromptOptimizer(
        Config(engine="llm", api_key="k"), rewriter=BrokenRewriter("story")
    )
    results = optimizer.optimize_batch(["write a story", "explain machine learning"])
    assert len(results) == 1
    assert len(optimizer.failures) == 1
    assert optimizer.failures[0].index == 0
    assert "provider is down" in optimizer.failures[0].error


def test_batch_can_still_fail_fast_on_request():
    optimizer = PromptOptimizer(
        Config(engine="llm", api_key="k"), rewriter=BrokenRewriter("story")
    )
    with pytest.raises(ProviderError):
        optimizer.optimize_batch(["write a story", "explain ai"], stop_on_error=True)


def test_batch_records_non_string_entries():
    optimizer = PromptOptimizer(Config(engine="heuristic"))
    results = optimizer.optimize_batch(["write a story", None, 7])
    assert len(results) == 1
    assert [f.index for f in optimizer.failures] == [1, 2]


def test_batch_rejects_a_bare_string():
    with pytest.raises(PromptOptimizerError, match="sequence"):
        PromptOptimizer(Config(engine="heuristic")).optimize_batch("write a story")


def test_failures_reset_between_batches():
    optimizer = PromptOptimizer(Config(engine="heuristic"))
    optimizer.optimize_batch([None])
    assert len(optimizer.failures) == 1
    optimizer.optimize_batch(["write a story"])
    assert optimizer.failures == []


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [-0.1, 2.1, float("nan"), float("inf")])
def test_bad_temperature_is_rejected(bad):
    with pytest.raises(ConfigError):
        Config(temperature=bad)


@pytest.mark.parametrize("bad", [0, -1, float("inf")])
def test_bad_timeout_is_rejected(bad):
    with pytest.raises(ConfigError):
        Config(timeout=bad)


def test_negative_retries_are_rejected():
    with pytest.raises(ConfigError):
        Config(max_retries=-1)


def test_blank_model_is_rejected():
    with pytest.raises(ConfigError):
        Config(model="   ")


def test_environment_values_are_validated(monkeypatch):
    monkeypatch.setenv("PROMPT_OPTIMIZER_TIMEOUT", "soon")
    with pytest.raises(ConfigError, match="PROMPT_OPTIMIZER_TIMEOUT"):
        load_config()


def test_environment_retries_are_validated(monkeypatch):
    monkeypatch.setenv("PROMPT_OPTIMIZER_MAX_RETRIES", "2.5")
    with pytest.raises(ConfigError, match="PROMPT_OPTIMIZER_MAX_RETRIES"):
        load_config()


def test_environment_overrides_are_read(monkeypatch):
    monkeypatch.setenv("PROMPT_OPTIMIZER_TIMEOUT", "5")
    monkeypatch.setenv("PROMPT_OPTIMIZER_MAX_RETRIES", "0")
    config = load_config()
    assert config.timeout == 5.0
    assert config.max_retries == 0


# --------------------------------------------------------------------------- #
# CLI input handling
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("PROMPT_OPTIMIZER_ENGINE", "heuristic")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_a_non_utf8_file_is_an_error_not_a_traceback(tmp_path, capsys):
    source = tmp_path / "binary.txt"
    source.write_bytes(b"\x80\x81\xfe not text")
    assert main(["analyze", "-f", str(source)]) == 2
    assert "not valid UTF-8" in capsys.readouterr().err


def test_a_directory_passed_as_a_file_is_an_error(tmp_path, capsys):
    assert main(["analyze", "-f", str(tmp_path)]) == 2
    assert "is a directory" in capsys.readouterr().err


def test_batch_out_dir_that_is_a_file_is_an_error(tmp_path, capsys):
    source = tmp_path / "prompts.txt"
    source.write_text("write a story", encoding="utf-8")
    blocker = tmp_path / "blocker"
    blocker.write_text("in the way", encoding="utf-8")
    assert main(["batch", str(source), "-o", str(blocker)]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_batch_splits_crlf_files_into_prompts(tmp_path):
    source = tmp_path / "prompts.txt"
    source.write_bytes(b"write a story\r\n\r\nfix my python code\r\n")
    out_dir = tmp_path / "out"
    assert main(["batch", str(source), "-o", str(out_dir)]) == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["runs"] == 2


def test_batch_summary_carries_failures(tmp_path):
    source = tmp_path / "prompts.txt"
    source.write_text("write a story", encoding="utf-8")
    out_dir = tmp_path / "out"
    assert main(["batch", str(source), "-o", str(out_dir)]) == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["failures"] == []


def test_a_non_utf8_batch_file_is_an_error(tmp_path, capsys):
    source = tmp_path / "prompts.txt"
    source.write_bytes(b"\xff\xfe rubbish")
    assert main(["batch", str(source)]) == 2
    assert "not valid UTF-8" in capsys.readouterr().err
