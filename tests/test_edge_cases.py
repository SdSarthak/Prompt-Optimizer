"""Edge cases and failure modes: structure preservation, bad input, provider faults.

Everything here is deterministic and offline - no network, no API key, no files
outside pytest's tmp_path.
"""

import json

import pytest

from prompt_optimizer.analysis import analyze, find_ambiguities
from prompt_optimizer.heuristic import _normalize_task, build_prompt

FENCED = (
    "write a python function that:\n"
    "- reads a CSV\n"
    "- filters rows where amount > 100\n"
    "\n"
    "```python\n"
    "def stub():\n"
    "    return None\n"
    "```"
)


# --------------------------------------------------------------------------- #
# The heuristic engine must not destroy structure the user wrote
# --------------------------------------------------------------------------- #


def test_normalize_task_keeps_line_structure():
    task = _normalize_task(FENCED)
    assert "- reads a CSV" in task.splitlines()
    assert "- filters rows where amount > 100" in task.splitlines()


def test_normalize_task_preserves_code_block_indentation():
    task = _normalize_task(FENCED)
    assert "    return None" in task.splitlines()
    assert task.splitlines().count("```python") == 1


def test_normalize_task_does_not_punctuate_a_closing_fence():
    assert not _normalize_task(FENCED).endswith("```.")
    assert _normalize_task(FENCED).endswith("```")


def test_normalize_task_collapses_blank_line_runs():
    task = _normalize_task("first line\n\n\n\n\nsecond line")
    assert task == "First line\n\nsecond line."


def test_normalize_task_collapses_whitespace_inside_a_line():
    assert _normalize_task("write   a\tstory") == "Write a story."


def test_normalize_task_leaves_a_leading_bullet_alone():
    assert _normalize_task("- do the thing").startswith("- do")


def test_normalize_task_does_not_double_punctuate():
    assert _normalize_task("what is this?") == "What is this?"
    assert _normalize_task("do this:") == "Do this:"


def test_normalize_task_handles_whitespace_only_input():
    assert _normalize_task("   \n\n  \t ") == ""


def test_build_prompt_keeps_the_users_code_block():
    built = build_prompt(FENCED)
    assert "```python" in built
    assert "    return None" in built.splitlines()


# --------------------------------------------------------------------------- #
# Action-verb detection must respect word boundaries
# --------------------------------------------------------------------------- #


def test_leading_verb_requires_a_word_boundary():
    # "listen" starts with the verb "list" but is not a request to list anything.
    assert analyze("listen to this recording").clarity < analyze("list the files").clarity


def test_no_explicit_request_finding_ignores_verb_prefixes():
    findings = find_ambiguities("listening carefully to the recording of the meeting")
    assert any("no explicit request" in f for f in findings)
    assert not any("no explicit request" in f for f in find_ambiguities("list the files"))


# --------------------------------------------------------------------------- #
# Degenerate analysis inputs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "\t"])
def test_analyze_survives_blank_input(text):
    report = analyze(text)
    assert report.word_count == 0
    assert report.sentence_count == 0
    assert report.avg_sentence_length == 0.0
    assert report.reading_ease == 0.0
    assert report.intent == "general"
    assert all(0.0 <= v <= 1.0 for v in report.scores.values())
    json.dumps(report.to_dict())


@pytest.mark.parametrize(
    "text",
    [
        "?" * 200,
        ".....",
        "a" * 5000,
        "\x00 null byte",
        "emoji only \U0001f600\U0001f680",
        "你好，写一个故事",
    ],
)
def test_analyze_never_leaves_the_unit_interval(text):
    report = analyze(text)
    for name, value in report.scores.items():
        assert 0.0 <= value <= 1.0, (name, value)
    assert 0.0 <= report.reading_ease <= 100.0
    assert report.render()


def test_analyze_rejects_non_strings():
    for bad in (None, 42, ["prompt"]):
        with pytest.raises(TypeError):
            analyze(bad)


def test_many_ambiguities_cannot_push_clarity_negative():
    noisy = (
        "it is some good nice stuff and things, very really interesting, "
        "make it brief but comprehensive, use {{ NAME }} etc"
    )
    assert analyze(noisy).clarity >= 0.0


def test_build_prompt_is_deterministic():
    assert build_prompt(FENCED) == build_prompt(FENCED)
