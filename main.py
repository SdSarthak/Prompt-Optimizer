"""Prompt Optimizer entry point.

    python main.py                      # short demo on the built-in samples
    python main.py optimize "..."       # same CLI as `python -m prompt_optimizer`
"""

import sys

from prompt_optimizer import PromptOptimizer, load_config
from prompt_optimizer.cli import main as cli_main

DEMO_PROMPTS = [
    "write a story",
    "explain machine learning",
    "help me code a website",
]


def demo() -> int:
    """Run the heuristic engine over a few sample prompts."""
    config = load_config(engine="heuristic")
    optimizer = PromptOptimizer(config)

    print("=" * 70)
    print("Prompt Optimizer - demo (heuristic engine, no API calls)")
    print("=" * 70)

    for index, rough in enumerate(DEMO_PROMPTS, 1):
        result = optimizer.optimize(rough)
        print(f"\n[{index}/{len(DEMO_PROMPTS)}] {rough!r}")
        print(
            f"  intent={result.analysis_before.intent}  template={result.template}  "
            f"effectiveness {result.analysis_before.effectiveness:.2f} -> "
            f"{result.analysis_after.effectiveness:.2f} ({result.improvement:+.2f})"
        )
        preview = result.optimized.splitlines()
        for line in preview[:6]:
            print(f"  | {line}")
        if len(preview) > 6:
            print(f"  | ... {len(preview) - 6} more lines")

    summary = optimizer.summary()
    print("\n" + "=" * 70)
    print(
        f"{summary['runs']} prompts optimized, "
        f"average effectiveness gain {summary['average_improvement']:+.3f}"
    )
    print("Run `python main.py --help` for the full CLI.")
    print("=" * 70)
    return 0


def main() -> int:
    if len(sys.argv) > 1:
        return cli_main(sys.argv[1:])
    return demo()


if __name__ == "__main__":
    sys.exit(main())
