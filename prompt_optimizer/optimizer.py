"""The optimizer: picks an engine, rewrites the prompt, measures the change."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import heuristic
from .analysis import PromptAnalysis, analyze
from .compare import Comparison, compare
from .config import Config, load_config
from .errors import ProviderError, PromptOptimizerError
from .gemini import GeminiRewriter
from .templates import Template, get_template, template_for_intent


def _one_line(text: str, limit: int = 160) -> str:
    """Squash a provider error onto one readable line for terminal output."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


@dataclass
class OptimizationResult:
    """Everything one optimization run produced."""

    original: str
    optimized: str
    engine: str
    template: str
    target_model: str
    analysis_before: PromptAnalysis
    analysis_after: PromptAnalysis
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    fallback_reason: Optional[str] = None

    @property
    def improvement(self) -> float:
        """Change in the overall effectiveness score."""
        return round(
            self.analysis_after.effectiveness - self.analysis_before.effectiveness, 3
        )

    @property
    def comparison(self) -> Comparison:
        return compare(self.original, self.optimized, "before", "after")

    def to_dict(self) -> Dict[str, object]:
        return {
            "original": self.original,
            "optimized": self.optimized,
            "engine": self.engine,
            "template": self.template,
            "target_model": self.target_model,
            "timestamp": self.timestamp,
            "fallback_reason": self.fallback_reason,
            "improvement": self.improvement,
            "analysis_before": self.analysis_before.to_dict(),
            "analysis_after": self.analysis_after.to_dict(),
        }

    def render(self) -> str:
        header = (
            f"engine={self.engine}  template={self.template}  "
            f"target={self.target_model}  effectiveness "
            f"{self.analysis_before.effectiveness:.2f} -> "
            f"{self.analysis_after.effectiveness:.2f} ({self.improvement:+.2f})"
        )
        parts = ["=" * 70, "OPTIMIZED PROMPT", "=" * 70, self.optimized, "", "-" * 70, header]
        if self.fallback_reason:
            parts.append(
                "note: fell back to the heuristic engine - "
                f"{_one_line(self.fallback_reason)}"
            )
        return "\n".join(parts)

    def save(self, path: str) -> str:
        """Write a human-readable record of the run."""
        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        body = [
            "=" * 70,
            "PROMPT OPTIMIZATION RESULT",
            "=" * 70,
            f"timestamp     : {self.timestamp}",
            f"engine        : {self.engine}",
            f"template      : {self.template}",
            f"target model  : {self.target_model}",
            f"effectiveness : {self.analysis_before.effectiveness:.2f} -> "
            f"{self.analysis_after.effectiveness:.2f} ({self.improvement:+.2f})",
            "",
            "ORIGINAL PROMPT",
            "-" * 70,
            self.original,
            "",
            "OPTIMIZED PROMPT",
            "-" * 70,
            self.optimized,
            "",
            self.comparison.render(),
            "",
        ]
        target.write_text("\n".join(body), encoding="utf-8")
        return str(target)


class PromptOptimizer:
    """Optimize prompts with the heuristic engine, the LLM engine, or both."""

    def __init__(self, config: Optional[Config] = None, rewriter=None) -> None:
        self.config = config or load_config()
        self._rewriter = rewriter
        self.history: List[OptimizationResult] = []

    # -- engines ---------------------------------------------------------- #

    @property
    def rewriter(self) -> GeminiRewriter:
        if self._rewriter is None:
            self._rewriter = GeminiRewriter(self.config)
        return self._rewriter

    def _select_template(
        self, analysis: PromptAnalysis, template: Optional[str]
    ) -> Template:
        if template:
            return get_template(template)
        return template_for_intent(analysis.intent)

    # -- public API ------------------------------------------------------- #

    def analyze(self, prompt: str) -> PromptAnalysis:
        """Static analysis only - no rewriting."""
        return analyze(prompt)

    def optimize(
        self,
        rough_prompt: str,
        template: Optional[str] = None,
        target_model: str = "general",
        engine: Optional[str] = None,
    ) -> OptimizationResult:
        """Rewrite ``rough_prompt`` and report how much it improved."""
        if not isinstance(rough_prompt, str) or not rough_prompt.strip():
            raise PromptOptimizerError("prompt is empty; nothing to optimize")

        analysis_before = analyze(rough_prompt)
        tpl = self._select_template(analysis_before, template)

        requested = engine or self.config.engine
        if requested not in ("auto", "heuristic", "llm"):
            raise PromptOptimizerError(f"unknown engine {requested!r}")

        fallback_reason: Optional[str] = None
        used = "heuristic"
        optimized: Optional[str] = None

        if requested in ("llm", "auto") and (requested == "llm" or self.config.has_api_key):
            try:
                optimized = self.rewriter.rewrite(analysis_before, tpl, target_model)
                used = "llm"
            except ProviderError as exc:
                if requested == "llm":
                    raise
                fallback_reason = str(exc)
        elif requested == "auto":
            fallback_reason = "no API key configured"

        if optimized is None:
            optimized = heuristic.build_prompt(
                rough_prompt, analysis_before, tpl, target_model
            )
            used = "heuristic"

        result = OptimizationResult(
            original=rough_prompt.strip(),
            optimized=optimized,
            engine=used,
            template=tpl.name,
            target_model=target_model,
            analysis_before=analysis_before,
            analysis_after=analyze(optimized),
            fallback_reason=fallback_reason,
        )
        self.history.append(result)
        return result

    def optimize_batch(
        self,
        prompts: Sequence[str],
        template: Optional[str] = None,
        target_model: str = "general",
        engine: Optional[str] = None,
    ) -> List[OptimizationResult]:
        """Optimize many prompts. One bad prompt does not stop the batch."""
        results: List[OptimizationResult] = []
        for prompt in prompts:
            if not prompt.strip():
                continue
            results.append(self.optimize(prompt, template, target_model, engine))
        return results

    def compare(self, prompt_a: str, prompt_b: str, label_a: str = "A",
                label_b: str = "B") -> Comparison:
        return compare(prompt_a, prompt_b, label_a, label_b)

    def summary(self) -> Dict[str, object]:
        """Aggregate stats over everything this instance has optimized."""
        if not self.history:
            return {"runs": 0, "average_improvement": 0.0, "engines": {}}
        engines: Dict[str, int] = {}
        for item in self.history:
            engines[item.engine] = engines.get(item.engine, 0) + 1
        improvements = [item.improvement for item in self.history]
        return {
            "runs": len(self.history),
            "average_improvement": round(sum(improvements) / len(improvements), 3),
            "best_improvement": max(improvements),
            "worst_improvement": min(improvements),
            "engines": engines,
        }
