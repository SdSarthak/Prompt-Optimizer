"""Command line interface for Prompt Optimizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from .analysis import analyze
from .compare import compare
from .config import load_config
from .errors import PromptOptimizerError
from .optimizer import PromptOptimizer
from .templates import list_templates, template_names

MODEL_FAMILIES = ("general", "gpt", "openai", "claude", "anthropic", "gemini", "google",
                  "llama", "mistral")


def _read_text(path: str) -> str:
    file_path = Path(path)
    if file_path.is_dir():
        raise PromptOptimizerError(f"{path} is a directory, not a file")
    if not file_path.is_file():
        raise PromptOptimizerError(f"file not found: {path}")
    return _decode(file_path)


def _decode(file_path: Path) -> str:
    """Read a text file, turning every failure into an actionable message."""
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PromptOptimizerError(
            f"{file_path} is not valid UTF-8 text (byte {exc.start}); "
            "prompts must be plain UTF-8 files"
        ) from exc
    except OSError as exc:
        raise PromptOptimizerError(f"could not read {file_path}: {exc}") from exc


def _resolve_prompt(text: Optional[Sequence[str]], file: Optional[str]) -> str:
    """Prompt comes from arguments, a file, or piped stdin - in that order."""
    if text:
        joined = " ".join(text).strip()
        if joined:
            return joined
    if file:
        content = _read_text(file).strip()
        if not content:
            raise PromptOptimizerError(f"{file} is empty")
        return content
    if sys.stdin is not None and not sys.stdin.isatty():
        try:
            piped = sys.stdin.read().strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise PromptOptimizerError(f"could not read the prompt from stdin: {exc}") from exc
        if piped:
            return piped
    raise PromptOptimizerError(
        "no prompt given; pass it as an argument, use --file, or pipe it on stdin"
    )


def _collect_batch_inputs(source: str) -> List[str]:
    path = Path(source)
    if path.is_dir():
        prompts = []
        for child in sorted(path.glob("*.txt")):
            content = _decode(child).strip()
            if content:
                prompts.append(content)
        if not prompts:
            raise PromptOptimizerError(f"no non-empty .txt files in {source}")
        return prompts
    if path.is_file():
        # Prompts are separated by a blank line; \r\n files must split too.
        text = _decode(path).replace("\r\n", "\n").replace("\r", "\n")
        blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
        if not blocks:
            raise PromptOptimizerError(f"{source} contains no prompts")
        return blocks
    raise PromptOptimizerError(f"batch input not found: {source}")


def _make_optimizer(args: argparse.Namespace) -> PromptOptimizer:
    config = load_config(
        engine=getattr(args, "engine", None),
        model=getattr(args, "model", None),
        temperature=getattr(args, "temperature", None),
        timeout=getattr(args, "timeout", None),
        max_retries=getattr(args, "retries", None),
    )
    return PromptOptimizer(config)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_analyze(args: argparse.Namespace) -> int:
    report = analyze(_resolve_prompt(args.prompt, args.file))
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.render())
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    prompt = _resolve_prompt(args.prompt, args.file)
    optimizer = _make_optimizer(args)
    result = optimizer.optimize(
        prompt,
        template=args.template,
        target_model=args.target,
        engine=args.engine,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.quiet:
        print(result.optimized)
    else:
        print(result.render())
        if args.show_analysis:
            print()
            print(result.comparison.render())

    if args.out:
        path = result.save(args.out)
        if not args.quiet and not args.json:
            print(f"\nsaved to {path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    prompt_a = _read_text(args.file_a).strip() if args.file_a else (args.prompt_a or "")
    prompt_b = _read_text(args.file_b).strip() if args.file_b else (args.prompt_b or "")
    if not prompt_a or not prompt_b:
        raise PromptOptimizerError(
            "compare needs two prompts: pass them positionally or use --file-a/--file-b"
        )
    result = compare(prompt_a, prompt_b, args.label_a, args.label_b)
    print(json.dumps(result.to_dict(), indent=2) if args.json else result.render())
    return 0


def cmd_templates(args: argparse.Namespace) -> int:
    templates = list_templates()
    if args.json:
        print(json.dumps(
            [
                {
                    "name": t.name,
                    "description": t.description,
                    "role": t.role,
                    "requirements": t.requirements,
                    "output_format": t.output_format,
                }
                for t in templates
            ],
            indent=2,
        ))
        return 0
    print("TEMPLATE LIBRARY")
    print("-" * 60)
    for tpl in templates:
        print(f"  {tpl.name:<20} {tpl.description}")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    prompts = _collect_batch_inputs(args.input)
    optimizer = _make_optimizer(args)
    results = optimizer.optimize_batch(
        prompts, template=args.template, target_model=args.target, engine=args.engine
    )

    out_dir = Path(args.out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise PromptOptimizerError(f"--out-dir {args.out_dir} exists and is not a directory")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PromptOptimizerError(f"could not create {args.out_dir}: {exc}") from exc

    for index, result in enumerate(results, 1):
        result.save(str(out_dir / f"optimized_{index:03d}.txt"))

    summary = optimizer.summary()
    failures = [f.to_dict() for f in optimizer.failures]
    summary_path = out_dir / "summary.json"
    try:
        summary_path.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "failures": failures,
                    "results": [r.to_dict() for r in results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise PromptOptimizerError(f"could not write {summary_path}: {exc}") from exc

    if args.json:
        print(json.dumps({**summary, "failures": failures}, indent=2))
    else:
        print(f"optimized {summary['runs']} prompt(s) into {out_dir}")
        print(f"average effectiveness gain: {summary['average_improvement']:+.3f}")
        print(f"engines used: {summary['engines']}")
        if failures:
            print(f"failed: {len(failures)} prompt(s); see {summary_path}")
        print(f"summary written to {summary_path}")
    return 1 if failures and not results else 0


def cmd_config(args: argparse.Namespace) -> int:
    config = load_config()
    print("CONFIGURATION")
    print("-" * 60)
    print(f"  api key       : {'set' if config.has_api_key else 'not set'}")
    print(f"  model         : {config.model}")
    print(f"  temperature   : {config.temperature}")
    print(f"  timeout       : {config.timeout}s")
    print(f"  max retries   : {config.max_retries}")
    print(f"  engine        : {config.engine} (resolves to {config.resolve_engine()})")
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt-optimizer",
        description="Analyse and rewrite rough prompts into effective ones.",
    )
    parser.add_argument("--version", action="version", version=f"prompt-optimizer {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    def add_engine_flags(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--engine", choices=("auto", "heuristic", "llm"),
                         help="rewriting engine (default: auto)")
        sub.add_argument("--model", help="Gemini model id for the llm engine")
        sub.add_argument("--temperature", type=float, help="sampling temperature (0.0-2.0)")
        sub.add_argument("--timeout", type=float,
                         help="seconds to wait for the provider (default: 60)")
        sub.add_argument("--retries", type=int,
                         help="retries on a transient provider failure (default: 2)")
        sub.add_argument("--template", choices=template_names(),
                         help="force a template instead of picking one from the intent")
        sub.add_argument("--target", default="general", choices=MODEL_FAMILIES,
                         help="model family the prompt is aimed at")

    p_analyze = subparsers.add_parser("analyze", help="score a prompt without rewriting it")
    p_analyze.add_argument("prompt", nargs="*", help="prompt text")
    p_analyze.add_argument("-f", "--file", help="read the prompt from a file")
    p_analyze.add_argument("--json", action="store_true", help="emit JSON")
    p_analyze.set_defaults(func=cmd_analyze)

    p_optimize = subparsers.add_parser("optimize", help="rewrite a prompt")
    p_optimize.add_argument("prompt", nargs="*", help="prompt text")
    p_optimize.add_argument("-f", "--file", help="read the prompt from a file")
    p_optimize.add_argument("-o", "--out", help="write a full report to this file")
    p_optimize.add_argument("--json", action="store_true", help="emit JSON")
    p_optimize.add_argument("-q", "--quiet", action="store_true",
                            help="print only the optimized prompt")
    p_optimize.add_argument("--show-analysis", action="store_true",
                            help="also print the before/after metric table")
    add_engine_flags(p_optimize)
    p_optimize.set_defaults(func=cmd_optimize)

    p_compare = subparsers.add_parser("compare", help="A/B compare two prompts")
    p_compare.add_argument("prompt_a", nargs="?", help="first prompt")
    p_compare.add_argument("prompt_b", nargs="?", help="second prompt")
    p_compare.add_argument("--file-a", dest="file_a", help="read the first prompt from a file")
    p_compare.add_argument("--file-b", dest="file_b", help="read the second prompt from a file")
    p_compare.add_argument("--label-a", dest="label_a", default="A")
    p_compare.add_argument("--label-b", dest="label_b", default="B")
    p_compare.add_argument("--json", action="store_true", help="emit JSON")
    p_compare.set_defaults(func=cmd_compare)

    p_templates = subparsers.add_parser("templates", help="list the template library")
    p_templates.add_argument("--json", action="store_true", help="emit JSON")
    p_templates.set_defaults(func=cmd_templates)

    p_batch = subparsers.add_parser("batch", help="optimize many prompts at once")
    p_batch.add_argument("input", help="a .txt file (prompts separated by blank lines) or a directory")
    p_batch.add_argument("-o", "--out-dir", default="out", help="output directory (default: out)")
    p_batch.add_argument("--json", action="store_true", help="emit JSON")
    add_engine_flags(p_batch)
    p_batch.set_defaults(func=cmd_batch)

    p_config = subparsers.add_parser("config", help="show the resolved configuration")
    p_config.set_defaults(func=cmd_config)

    return parser


def use_utf8_stdio() -> None:
    """Make the streams UTF-8 so a non-ASCII prompt cannot kill the process.

    On Windows the console and any redirected pipe default to cp1252, and
    printing an optimized prompt containing an em dash or CJK text raises
    UnicodeEncodeError. Errors are replaced rather than raised: mangled output
    beats a traceback.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - detached/odd streams
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    use_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except PromptOptimizerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # pragma: no cover - depends on the consumer
        return 141
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
