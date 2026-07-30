import pytest

from prompt_optimizer.analysis import (
    analyze,
    count_syllables,
    detect_intent,
    find_ambiguities,
    flesch_reading_ease,
)

WELL_FORMED = (
    "You are a senior Python engineer.\n\n"
    "Task: refactor the function below so it handles empty input.\n\n"
    "Requirements:\n"
    "- You must keep the public signature unchanged\n"
    "- Include an example call\n\n"
    "Output format: a single fenced code block in markdown.\n"
    "Audience: developers who are new to this codebase. Keep a professional tone.\n"
)


def test_syllable_counting():
    assert count_syllables("cat") == 1
    assert count_syllables("water") == 2
    assert count_syllables("make") == 1
    assert count_syllables("") == 0


def test_reading_ease_is_bounded():
    assert 0.0 <= flesch_reading_ease("The cat sat on the mat.") <= 100.0
    assert flesch_reading_ease("") == 0.0


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("refactor this python function and add a unit test", "code"),
        ("write a short story about a lighthouse", "creative"),
        ("generate an image of a robot, photo style illustration", "image"),
        ("extract all names and dates and classify them", "extraction"),
        ("reply to the customer as a support agent chatbot persona", "conversation"),
    ],
)
def test_detect_intent(prompt, expected):
    assert detect_intent(prompt) == expected


def test_detect_intent_falls_back_to_general():
    assert detect_intent("blorp zizzle") == "general"


def test_short_prompt_scores_poorly():
    report = analyze("write a story")
    assert report.effectiveness < 0.4
    assert report.missing  # nothing is specified
    assert any("too short" in item for item in report.ambiguities)
    assert report.suggestions


def test_well_formed_prompt_scores_well():
    report = analyze(WELL_FORMED)
    assert report.effectiveness > 0.7
    assert report.signals["role"]
    assert report.signals["constraints"]
    assert report.signals["output_format"]
    assert report.signals["audience"]
    assert report.structure_quality > 0.5


def test_scores_stay_in_unit_range():
    for prompt in ["", "a", "write " * 400, WELL_FORMED]:
        report = analyze(prompt)
        for name, value in report.scores.items():
            assert 0.0 <= value <= 1.0, f"{name} out of range for {prompt[:20]!r}"


def test_analysis_is_deterministic():
    assert analyze(WELL_FORMED).to_dict() == analyze(WELL_FORMED).to_dict()


def test_empty_prompt_is_flagged():
    report = analyze("   ")
    assert report.word_count == 0
    assert report.ambiguities == ["prompt is empty"]


def test_ambiguity_detection():
    findings = find_ambiguities("Make it good and add some stuff, etc.")
    assert any("vague wording" in f for f in findings)

    findings = find_ambiguities("Write a brief but comprehensive guide to Rust ownership.")
    assert any("conflicting instructions" in f for f in findings)

    findings = find_ambiguities("Summarize {{DOCUMENT}} in three bullet points please.")
    assert any("placeholder" in f for f in findings)

    findings = find_ambiguities("It should be rewritten with clear sections and examples.")
    assert any("pronoun" in f for f in findings)


def test_analyze_rejects_non_string():
    with pytest.raises(TypeError):
        analyze(None)


def test_render_contains_every_section():
    rendered = analyze("write a story").render()
    for heading in ("PROMPT ANALYSIS", "SCORES", "SIGNALS", "AMBIGUITIES", "SUGGESTIONS"):
        assert heading in rendered


def test_to_dict_shape():
    data = analyze("write a story").to_dict()
    assert set(data) == {
        "char_count",
        "word_count",
        "sentence_count",
        "avg_sentence_length",
        "reading_ease",
        "intent",
        "signals",
        "ambiguities",
        "scores",
        "suggestions",
    }
