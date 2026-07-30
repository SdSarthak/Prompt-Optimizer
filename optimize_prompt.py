"""File-in / file-out convenience script.

Reads rough_prompt.txt, optimizes it, writes improved_prompt.txt. Uses the LLM
engine when GEMINI_API_KEY is set and falls back to the heuristic engine
otherwise, so it always produces a result.

    python optimize_prompt.py [input.txt] [output.txt]
"""

import sys
from pathlib import Path

from prompt_optimizer import PromptOptimizer
from prompt_optimizer.errors import PromptOptimizerError

DEFAULT_INPUT = "rough_prompt.txt"
DEFAULT_OUTPUT = "improved_prompt.txt"


def main() -> int:
    input_path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT)
    output_path = Path(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT)

    if not input_path.is_file():
        print(f"error: {input_path} not found", file=sys.stderr)
        return 2

    rough_prompt = input_path.read_text(encoding="utf-8").strip()
    if not rough_prompt:
        print(f"error: {input_path} is empty", file=sys.stderr)
        return 2

    try:
        result = PromptOptimizer().optimize(rough_prompt)
    except PromptOptimizerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_path.write_text(result.optimized + "\n", encoding="utf-8")

    print(f"engine        : {result.engine}")
    print(f"template      : {result.template}")
    print(
        f"effectiveness : {result.analysis_before.effectiveness:.2f} -> "
        f"{result.analysis_after.effectiveness:.2f} ({result.improvement:+.2f})"
    )
    if result.fallback_reason:
        reason = " ".join(result.fallback_reason.split())
        print(f"note          : {reason[:160]}")
    print(f"written to    : {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
