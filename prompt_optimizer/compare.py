"""Side-by-side comparison of two prompts (the A/B testing surface)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .analysis import PromptAnalysis, analyze

SCORE_NAMES = (
    "clarity",
    "specificity",
    "context_completeness",
    "structure_quality",
    "effectiveness",
)


@dataclass
class Comparison:
    """Metric-by-metric diff between two prompts."""

    label_a: str
    label_b: str
    analysis_a: PromptAnalysis
    analysis_b: PromptAnalysis

    @property
    def deltas(self) -> Dict[str, float]:
        a, b = self.analysis_a.scores, self.analysis_b.scores
        return {name: round(b[name] - a[name], 3) for name in SCORE_NAMES}

    @property
    def winner(self) -> Optional[str]:
        """Label of the stronger prompt, or None when effectiveness ties."""
        delta = self.deltas["effectiveness"]
        if delta > 0:
            return self.label_b
        if delta < 0:
            return self.label_a
        return None

    def to_dict(self) -> Dict[str, object]:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "scores_a": self.analysis_a.scores,
            "scores_b": self.analysis_b.scores,
            "deltas": self.deltas,
            "winner": self.winner,
        }

    def render(self) -> str:
        a, b = self.analysis_a.scores, self.analysis_b.scores
        deltas = self.deltas
        width = max(len(self.label_a), len(self.label_b), 8)
        lines = [
            "PROMPT COMPARISON",
            "-" * 60,
            f"  {'metric':<22}{self.label_a:>{width}}{self.label_b:>{width + 2}}{'delta':>10}",
        ]
        for name in SCORE_NAMES:
            delta = deltas[name]
            arrow = "+" if delta > 0 else ("-" if delta < 0 else " ")
            lines.append(
                f"  {name:<22}{a[name]:>{width}.2f}{b[name]:>{width + 2}.2f}"
                f"{arrow + format(abs(delta), '.2f'):>10}"
            )
        lines.append("")
        lines.append(f"  words{'':<17}{self.analysis_a.word_count:>{width}}"
                     f"{self.analysis_b.word_count:>{width + 2}}")
        lines.append("")
        winner = self.winner
        lines.append(f"  winner: {winner}" if winner else "  winner: tie")
        return "\n".join(lines)


def compare(
    prompt_a: str,
    prompt_b: str,
    label_a: str = "A",
    label_b: str = "B",
) -> Comparison:
    """Analyse both prompts and return their metric diff."""
    return Comparison(
        label_a=label_a,
        label_b=label_b,
        analysis_a=analyze(prompt_a),
        analysis_b=analyze(prompt_b),
    )
