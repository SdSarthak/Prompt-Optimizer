import json

import pytest

from prompt_optimizer import PromptOptimizer, compare
from prompt_optimizer.analysis import analyze
from prompt_optimizer.config import Config
from prompt_optimizer.errors import ProviderError, PromptOptimizerError
from prompt_optimizer.gemini import GeminiRewriter, build_meta_prompt, clean_response
from prompt_optimizer.heuristic import build_prompt
from prompt_optimizer.templates import get_template

ROUGH = "write a story"


def heuristic_optimizer():
    return PromptOptimizer(Config(engine="heuristic", api_key=None))


class FakeRewriter:
    """Stands in for GeminiRewriter so the LLM path is testable offline."""

    def __init__(self, text="# Role\nYou are a writer.\n\n# Task\nWrite a story.", error=None):
        self.text = text
        self.error = error
        self.calls = []

    def rewrite(self, analysis, template, target_model="general"):
        self.calls.append((analysis.prompt, template.name, target_model))
        if self.error:
            raise self.error
        return self.text


# --------------------------------------------------------------------------- #
# Heuristic engine
# --------------------------------------------------------------------------- #


def test_heuristic_adds_the_missing_sections():
    built = build_prompt(ROUGH)
    for heading in ("# Role", "# Task", "# Requirements", "# Output format"):
        assert heading in built
    assert "Write a story." in built


def test_heuristic_does_not_duplicate_an_existing_role():
    prompt = "You are a novelist. Write a story about the sea."
    assert "# Role" not in build_prompt(prompt)


def test_heuristic_normalizes_vague_wording():
    built = build_prompt("write some stuff about robots, make it good")
    assert "some stuff" not in built.lower()


def test_heuristic_requirements_are_unique():
    built = build_prompt("analyze the sales data")
    requirements = [
        line for line in built.splitlines() if line.startswith("- ")
    ]
    assert len(requirements) == len(set(r.lower() for r in requirements))


def test_target_model_adds_a_model_note():
    assert "# Model note" in build_prompt(ROUGH, target_model="claude")
    assert "# Model note" not in build_prompt(ROUGH, target_model="general")


def test_heuristic_always_improves_effectiveness():
    for rough in ["write a story", "explain ai", "fix my code", "make an image of a cat"]:
        before = analyze(rough).effectiveness
        after = analyze(build_prompt(rough)).effectiveness
        assert after > before, rough


# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #


def test_optimize_uses_the_heuristic_engine_when_asked():
    result = heuristic_optimizer().optimize(ROUGH)
    assert result.engine == "heuristic"
    assert result.template == "creative_writing"
    assert result.improvement > 0


def test_optimize_rejects_an_empty_prompt():
    with pytest.raises(PromptOptimizerError):
        heuristic_optimizer().optimize("   ")


def test_optimize_rejects_an_unknown_engine():
    with pytest.raises(PromptOptimizerError):
        heuristic_optimizer().optimize(ROUGH, engine="telepathy")


def test_explicit_template_overrides_intent():
    result = heuristic_optimizer().optimize(ROUGH, template="data_analysis")
    assert result.template == "data_analysis"


def test_llm_engine_is_used_when_available():
    fake = FakeRewriter()
    optimizer = PromptOptimizer(Config(engine="llm", api_key="k"), rewriter=fake)
    result = optimizer.optimize(ROUGH, target_model="gemini")
    assert result.engine == "llm"
    assert result.optimized == fake.text
    assert fake.calls == [(ROUGH, "creative_writing", "gemini")]


def test_auto_falls_back_to_heuristic_when_the_provider_fails():
    fake = FakeRewriter(error=ProviderError("boom"))
    optimizer = PromptOptimizer(Config(engine="auto", api_key="k"), rewriter=fake)
    result = optimizer.optimize(ROUGH)
    assert result.engine == "heuristic"
    assert result.fallback_reason == "boom"
    assert "# Task" in result.optimized


def test_auto_without_a_key_never_calls_the_provider():
    fake = FakeRewriter()
    optimizer = PromptOptimizer(Config(engine="auto", api_key=None), rewriter=fake)
    result = optimizer.optimize(ROUGH)
    assert result.engine == "heuristic"
    assert fake.calls == []
    assert result.fallback_reason == "no API key configured"


def test_explicit_llm_engine_propagates_provider_errors():
    fake = FakeRewriter(error=ProviderError("boom"))
    optimizer = PromptOptimizer(Config(engine="llm", api_key="k"), rewriter=fake)
    with pytest.raises(ProviderError):
        optimizer.optimize(ROUGH, engine="llm")


def test_history_and_summary():
    optimizer = heuristic_optimizer()
    assert optimizer.summary() == {"runs": 0, "average_improvement": 0.0, "engines": {}}
    optimizer.optimize("write a story")
    optimizer.optimize("explain machine learning")
    summary = optimizer.summary()
    assert summary["runs"] == 2
    assert summary["engines"] == {"heuristic": 2}
    assert summary["average_improvement"] > 0
    assert len(optimizer.history) == 2


def test_batch_skips_blank_prompts():
    results = heuristic_optimizer().optimize_batch(["write a story", "", "   ", "fix my code"])
    assert len(results) == 2


def test_result_serializes_to_json():
    result = heuristic_optimizer().optimize(ROUGH)
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["engine"] == "heuristic"
    assert payload["analysis_after"]["scores"]["effectiveness"] > 0


def test_result_save_writes_a_report(tmp_path):
    result = heuristic_optimizer().optimize(ROUGH)
    path = result.save(str(tmp_path / "nested" / "report.txt"))
    body = open(path, encoding="utf-8").read()
    assert "ORIGINAL PROMPT" in body
    assert "OPTIMIZED PROMPT" in body
    assert "PROMPT COMPARISON" in body


def test_render_reports_the_fallback_on_one_line():
    fake = FakeRewriter(error=ProviderError("line one\nline two " + "x" * 400))
    optimizer = PromptOptimizer(Config(engine="auto", api_key="k"), rewriter=fake)
    note = [
        line
        for line in optimizer.optimize(ROUGH).render().splitlines()
        if line.startswith("note:")
    ]
    assert len(note) == 1
    assert len(note[0]) < 240


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def test_compare_picks_the_stronger_prompt():
    result = compare(ROUGH, build_prompt(ROUGH), "before", "after")
    assert result.winner == "after"
    assert result.deltas["effectiveness"] > 0
    assert "PROMPT COMPARISON" in result.render()


def test_compare_reports_a_tie():
    assert compare(ROUGH, ROUGH).winner is None


def test_compare_serializes():
    payload = compare(ROUGH, build_prompt(ROUGH)).to_dict()
    assert set(payload) == {"label_a", "label_b", "scores_a", "scores_b", "deltas", "winner"}


# --------------------------------------------------------------------------- #
# Gemini provider (no network)
# --------------------------------------------------------------------------- #


def test_meta_prompt_carries_the_analysis():
    report = analyze(ROUGH)
    meta = build_meta_prompt(report, get_template("creative_writing"), "gemini")
    assert ROUGH in meta
    assert "creative_writing" in meta
    assert "missing role" in meta
    assert "Target model family: gemini" in meta


def test_clean_response_strips_fences_and_lead_ins():
    assert clean_response("```markdown\n# Role\nhi\n```") == "# Role\nhi"
    assert clean_response("Here is the rewritten prompt:\n# Role") == "# Role"
    assert clean_response("Optimized prompt: do the thing") == "do the thing"
    assert clean_response("  plain  ") == "plain"
    assert clean_response(None) == ""


def test_rewriter_without_a_key_raises_provider_error():
    rewriter = GeminiRewriter(Config(api_key=None))
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        _ = rewriter.client


def test_rewriter_wraps_sdk_failures():
    class ExplodingClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                raise RuntimeError("network down")

    rewriter = GeminiRewriter(Config(api_key="k"), client=ExplodingClient())
    with pytest.raises(ProviderError, match="network down"):
        rewriter.rewrite(analyze(ROUGH), get_template("general"))


def test_rewriter_rejects_an_empty_response():
    class EmptyClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                return type("R", (), {"text": "   "})()

    rewriter = GeminiRewriter(Config(api_key="k"), client=EmptyClient())
    with pytest.raises(ProviderError, match="empty"):
        rewriter.rewrite(analyze(ROUGH), get_template("general"))


def test_rewriter_returns_cleaned_text():
    captured = {}

    class OkClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                captured.update(kwargs)
                return type("R", (), {"text": "```\n# Role\nYou are a writer.\n```"})()

    config = Config(api_key="k", model="gemini-test", temperature=0.7)
    rewriter = GeminiRewriter(config, client=OkClient())
    assert rewriter.rewrite(analyze(ROUGH), get_template("general")) == "# Role\nYou are a writer."
    assert captured["model"] == "gemini-test"
    assert captured["config"]["temperature"] == 0.7
