# Prompt Optimizer

Turn a rough one-liner into a fully specified prompt, and show you the numbers that say it got better.

`prompt-optimizer` statically analyses a prompt (intent, readability, ambiguity, five weighted quality scores), then rewrites it using a template chosen from the detected intent. Rewriting runs either **offline** with a deterministic rule engine or **through Gemini** when an API key is available. The offline engine is not a stub: it is the fallback whenever the network or your quota says no, so the tool always produces something usable.

## What it actually does

| Capability | Where |
| --- | --- |
| Intent detection, readability (Flesch), ambiguity findings | `prompt_optimizer/analysis.py` |
| Five scores: clarity, specificity, context, structure, effectiveness | `prompt_optimizer/analysis.py` |
| Nine-template library, auto-selected per intent | `prompt_optimizer/templates.py` |
| Offline rule-based rewriting | `prompt_optimizer/heuristic.py` |
| Gemini-backed rewriting with graceful fallback | `prompt_optimizer/gemini.py` |
| A/B comparison with per-metric deltas | `prompt_optimizer/compare.py` |
| Batch optimization with a JSON summary | `prompt_optimizer/cli.py` |

## Install

```bash
git clone https://github.com/SdSarthak/Prompt-Optimizer.git
cd Prompt-Optimizer
pip install -r requirements.txt
```

Or install it as a package, which also puts a `prompt-optimizer` command on your PATH:

```bash
pip install -e .
```

Requires Python 3.8+.

### Configuration

Everything works without an API key. To enable Gemini-backed rewriting:

```bash
cp .env.example .env      # then paste your key into .env
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `GEMINI_API_KEY` | *(unset)* | Google AI Studio key. Without it the tool runs offline. |
| `PROMPT_OPTIMIZER_MODEL` | `gemini-2.5-flash` | Model used for rewriting. |
| `PROMPT_OPTIMIZER_TEMPERATURE` | `0.4` | Sampling temperature, 0.0-2.0. |
| `PROMPT_OPTIMIZER_ENGINE` | `auto` | `auto`, `heuristic`, or `llm`. |
| `PROMPT_OPTIMIZER_TIMEOUT` | `60` | Seconds to wait for the provider. |
| `PROMPT_OPTIMIZER_MAX_RETRIES` | `2` | Retries on a transient failure (429, 5xx, timeout). |

Every value is validated on load, and `--model`, `--temperature`, `--timeout`
and `--retries` override the environment per command.

`.env` is gitignored. Never commit a real key.

Check what got resolved:

```bash
python -m prompt_optimizer config
```

## Usage

```bash
python -m prompt_optimizer <command>   # or: prompt-optimizer <command>
python main.py                         # demo run on the built-in samples
```

### Analyse a prompt

```bash
python -m prompt_optimizer analyze "write a story"
```

```
PROMPT ANALYSIS
------------------------------------------------------------
  intent            : creative
  words / sentences : 3 / 1
  avg sentence len  : 3.0
  reading ease      : 91.0

SCORES (0-1)
------------------------------------------------------------
  clarity               0.61  ############
  specificity           0.10  ##
  context_completeness  0.05  #
  structure_quality     0.15  ###
  effectiveness         0.25  #####
```

The full report also lists which signals are missing (role, context, constraints, examples, output format, audience, tone, reasoning steps), the ambiguities it found, and a suggestion per gap. Add `--json` for machine-readable output.

### Optimize a prompt

```bash
python -m prompt_optimizer optimize "write a story" --target claude --show-analysis
```

```
# Role
You are an accomplished writer with a distinctive, controlled voice.

# Task
Write a story.

# Context
The piece will be read on its own, so it has to work without extra explanation.

# Requirements
- Show the reader the scene rather than summarising it
- Keep one consistent point of view and tense throughout
...

# Output format
Prose only. No preamble, no commentary on your own writing.
```

```
engine=heuristic  template=creative_writing  target=claude  effectiveness 0.25 -> 0.89 (+0.64)

PROMPT COMPARISON
------------------------------------------------------------
  metric                  before     after     delta
  clarity                   0.61      0.80     +0.19
  specificity               0.10      1.00     +0.90
  context_completeness      0.05      1.00     +0.95
  structure_quality         0.15      0.75     +0.60
  effectiveness             0.25      0.89     +0.64
```

Useful flags: `-f/--file` to read the prompt from a file, `-q/--quiet` to print only the rewritten prompt (pipe-friendly), `-o/--out` to save a full report, `--engine`, `--template`, `--target`, `--model`, `--temperature`, `--json`.

Prompts can also arrive on stdin:

```bash
echo "explain machine learning" | python -m prompt_optimizer optimize -q
```

### Compare two prompts (A/B)

```bash
python -m prompt_optimizer compare --file-a before.txt --file-b after.txt
```

Prints the metric table above plus a winner. `--json` gives you the raw deltas.

### Batch

```bash
python -m prompt_optimizer batch examples/rough_prompts.txt -o out
```

Reads a `.txt` file (prompts separated by blank lines) or a directory of `.txt` files, writes one report per prompt plus `summary.json` with the aggregate effectiveness gain.

### Templates

```bash
python -m prompt_optimizer templates
```

| Template | For |
| --- | --- |
| `general` | Anything not covered below |
| `creative_writing` | Stories, scripts, copy |
| `technical_writing` | Docs and explainers |
| `code_generation` | Writing, reviewing, refactoring code |
| `data_analysis` | Analysis, comparison, evaluation |
| `extraction` | Structured extraction and classification |
| `problem_solving` | Reasoning to a recommendation |
| `conversational` | Chatbot and support personas |
| `image_generation` | Text-to-image prompts |

One is picked automatically from the detected intent; `--template` overrides it.

### File-in / file-out

```bash
python optimize_prompt.py rough_prompt.txt improved_prompt.txt
```

## Python API

```python
from prompt_optimizer import PromptOptimizer, analyze, compare

report = analyze("write a story")
print(report.intent, report.effectiveness, report.missing)

optimizer = PromptOptimizer()
result = optimizer.optimize("write a story", target_model="claude")
print(result.optimized, result.engine, result.improvement)

print(compare(result.original, result.optimized).render())
print(optimizer.summary())
```

`PromptOptimizer` accepts a `Config` and an injectable `rewriter`, which is how the LLM path is tested without a network call.

## How the scores work

All five scores are deterministic functions of the prompt text, clamped to `0.0-1.0`.

- **clarity** - action verb up front, sentence length, Flesch reading ease in the 30-80 band, minus a penalty per ambiguity finding.
- **specificity** - explicit constraints, output format, audience, tone, numeric limits; minus a penalty per vague term.
- **context_completeness** - role, background, examples, audience.
- **structure_quality** - line breaks, bullets, headings, delimiters, explicit reasoning steps.
- **effectiveness** - `0.30 * clarity + 0.30 * specificity + 0.20 * context + 0.20 * structure`.

They are a heuristic proxy for prompt quality, not a measurement of model output quality. They are useful for catching a prompt that forgot to say what it wants; they will not tell you which of two well-formed prompts produces better answers.

## Engine selection

| `--engine` | Behaviour |
| --- | --- |
| `auto` (default) | Use Gemini when a key is present; fall back to the rule engine on any provider error, and report why. |
| `heuristic` | Never touch the network. |
| `llm` | Require Gemini; fail loudly if it is unavailable. |

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

81 tests cover scoring bounds and determinism, intent and ambiguity detection, the template library, config precedence, both engines, the fallback path, comparison, and every CLI command. The Gemini provider is tested with a fake client, so the suite runs offline.

## Project layout

```
prompt_optimizer/
  analysis.py     static analysis and scoring
  templates.py    the template library
  heuristic.py    offline rewriting
  gemini.py       Gemini-backed rewriting
  optimizer.py    engine selection, results, history
  compare.py      A/B comparison
  cli.py          argparse CLI
main.py           demo + CLI passthrough
optimize_prompt.py  file-in / file-out helper
examples/         sample rough prompts for batch mode
tests/            pytest suite
```

## Limitations

- Scores are lexical heuristics. They reward prompts that state their requirements; they cannot judge factual quality.
- Intent detection is keyword-based and will misfire on prompts that mix domains. Use `--template` when it does.
- The Gemini engine depends on your key's quota. When the API refuses, `auto` silently produces a heuristic rewrite and prints the reason.
- English only.

## License

MIT
