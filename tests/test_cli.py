import json

import pytest

from prompt_optimizer.cli import main


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Keep every CLI test on the heuristic engine, whatever the local .env says."""
    monkeypatch.setenv("PROMPT_OPTIMIZER_ENGINE", "heuristic")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_no_command_prints_help(capsys):
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out


def test_analyze_text(capsys):
    assert main(["analyze", "write", "a", "story"]) == 0
    assert "PROMPT ANALYSIS" in capsys.readouterr().out


def test_analyze_json(capsys):
    assert main(["analyze", "--json", "write a story"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["intent"] == "creative"
    assert payload["scores"]["effectiveness"] < 0.5


def test_analyze_from_file(tmp_path, capsys):
    source = tmp_path / "p.txt"
    source.write_text("write a story", encoding="utf-8")
    assert main(["analyze", "--json", "-f", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["word_count"] == 3


def test_missing_file_is_an_error(capsys):
    assert main(["analyze", "-f", "nope.txt"]) == 2
    assert "file not found" in capsys.readouterr().err


def test_optimize_quiet_prints_only_the_prompt(capsys):
    assert main(["optimize", "-q", "--engine", "heuristic", "write a story"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Role")
    assert "OPTIMIZED PROMPT" not in out


def test_optimize_json(capsys):
    assert main(["optimize", "--json", "--engine", "heuristic", "write a story"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == "heuristic"
    assert payload["improvement"] > 0


def test_optimize_writes_a_report(tmp_path, capsys):
    out = tmp_path / "report.txt"
    assert main(["optimize", "--engine", "heuristic", "-o", str(out), "write a story"]) == 0
    assert "OPTIMIZED PROMPT" in out.read_text(encoding="utf-8")
    assert "saved to" in capsys.readouterr().out


def test_optimize_show_analysis(capsys):
    assert main(["optimize", "--engine", "heuristic", "--show-analysis", "write a story"]) == 0
    assert "PROMPT COMPARISON" in capsys.readouterr().out


def test_optimize_with_a_template_and_target(capsys):
    assert main([
        "optimize", "--json", "--engine", "heuristic",
        "--template", "extraction", "--target", "claude", "pull out the names",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["template"] == "extraction"
    assert payload["target_model"] == "claude"


def test_compare_two_prompts(capsys):
    assert main(["compare", "write a story", "You are a novelist. Write a 500 word story."]) == 0
    assert "winner" in capsys.readouterr().out


def test_compare_from_files(tmp_path, capsys):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("write a story", encoding="utf-8")
    b.write_text("You are a novelist. Write a 500 word story in markdown.", encoding="utf-8")
    assert main(["compare", "--file-a", str(a), "--file-b", str(b), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["winner"] == "B"


def test_compare_needs_two_prompts(capsys):
    assert main(["compare", "only one"]) == 2
    assert "two prompts" in capsys.readouterr().err


def test_templates_listing(capsys):
    assert main(["templates"]) == 0
    assert "code_generation" in capsys.readouterr().out


def test_templates_json(capsys):
    assert main(["templates", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {t["name"] for t in payload} >= {"general", "code_generation", "extraction"}


def test_batch_over_a_file(tmp_path, capsys):
    source = tmp_path / "prompts.txt"
    source.write_text("write a story\n\nfix my python code\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    assert main(["batch", str(source), "-o", str(out_dir), "--engine", "heuristic"]) == 0
    assert (out_dir / "optimized_001.txt").is_file()
    assert (out_dir / "optimized_002.txt").is_file()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["runs"] == 2
    assert "optimized 2 prompt(s)" in capsys.readouterr().out


def test_batch_over_a_directory(tmp_path):
    src = tmp_path / "prompts"
    src.mkdir()
    (src / "a.txt").write_text("write a story", encoding="utf-8")
    (src / "b.txt").write_text("analyze the sales data", encoding="utf-8")
    out_dir = tmp_path / "out"
    assert main(["batch", str(src), "-o", str(out_dir), "--engine", "heuristic", "--json"]) == 0
    assert (out_dir / "optimized_002.txt").is_file()


def test_batch_missing_input(capsys):
    assert main(["batch", "no-such-place"]) == 2
    assert "not found" in capsys.readouterr().err


def test_config_command(capsys):
    assert main(["config"]) == 0
    out = capsys.readouterr().out
    assert "engine" in out
    assert "not set" in out


def test_unknown_template_is_rejected():
    with pytest.raises(SystemExit):
        main(["optimize", "--template", "nope", "write a story"])
