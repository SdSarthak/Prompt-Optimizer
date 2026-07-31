"""The offline rewriter: builds a structured prompt from analysis + template.

This engine never touches the network, so the tool is fully usable without an
API key and the LLM engine always has something to fall back to.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .analysis import PromptAnalysis, analyze
from .templates import Template, template_for_intent

# Guidance that is genuinely model-specific rather than decorative.
MODEL_GUIDANCE: Dict[str, str] = {
    "gpt": "Return the final answer directly; do not restate the instructions back to me.",
    "openai": "Return the final answer directly; do not restate the instructions back to me.",
    "claude": "Think through the problem before answering, then give the final answer "
              "under a clear heading.",
    "anthropic": "Think through the problem before answering, then give the final answer "
                 "under a clear heading.",
    "gemini": "Ground every factual claim in the supplied context and flag anything you "
              "are unsure about.",
    "google": "Ground every factual claim in the supplied context and flag anything you "
              "are unsure about.",
    "llama": "Follow the sections above literally and do not add sections of your own.",
    "mistral": "Follow the sections above literally and do not add sections of your own.",
    "general": "",
}

_VAGUE_REPLACEMENTS = (
    (r"\bmake it good\b", "meet the requirements listed below"),
    (r"\bsome stuff\b", "the specific items listed below"),
    (r"\bnice\b", "well-structured"),
    (r"\bstuff\b", "content"),
    (r"\bthings\b", "items"),
    (r"\betc\.?\b", "and the remaining cases listed below"),
)

_FENCE_LINE_RE = re.compile(r"^\s*```")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
# Trailing characters after which a full stop would be wrong or redundant.
_TERMINAL_CHARS = ".!?:;,"


def _tidy_line(line: str) -> str:
    """Collapse runs of whitespace inside one line and de-vague its wording."""
    text = " ".join(line.split())
    for pattern, replacement in _VAGUE_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _normalize_task(prompt: str) -> str:
    """Tidy the user's own words without changing what they asked for.

    Line structure is load-bearing in a prompt: bullet lists, numbered steps and
    fenced code blocks all mean something. Only whitespace *within* a line is
    collapsed, code fences are copied through byte-for-byte, and runs of blank
    lines are squashed to one.
    """
    lines: List[str] = []
    in_fence = False
    blank_run = 0
    for raw in prompt.splitlines():
        if _FENCE_LINE_RE.match(raw):
            in_fence = not in_fence
            blank_run = 0
            lines.append(raw.rstrip())
            continue
        if in_fence:
            # Indentation is part of the code; keep it exactly as written.
            blank_run = 0
            lines.append(raw.rstrip())
            continue
        tidied = _tidy_line(raw)
        if not tidied:
            blank_run += 1
            if blank_run > 1 or not lines:
                continue
            lines.append("")
            continue
        blank_run = 0
        lines.append(tidied)

    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""

    first = lines[0]
    if first and first[0].islower() and not _LIST_ITEM_RE.match(first):
        lines[0] = first[0].upper() + first[1:]

    last = lines[-1]
    # Never punctuate a closing code fence, and never double up punctuation.
    if last and not _FENCE_LINE_RE.match(last) and last[-1] not in _TERMINAL_CHARS:
        lines[-1] = last + "."
    return "\n".join(lines)


def _intent_requirements(intent: str) -> List[str]:
    extras = {
        "code": ["Target the language and version implied by the task"],
        "creative": ["Aim for roughly 400-600 words unless the task says otherwise"],
        "image": ["Describe only what is visible in the frame"],
        "analysis": ["Show the numbers behind each conclusion"],
        "extraction": ["Emit nothing except the structured output"],
        "explanation": ["Start from what the reader already knows"],
        "conversation": ["Keep the reply under 120 words"],
    }
    return extras.get(intent, [])


def build_prompt(
    rough_prompt: str,
    analysis: Optional[PromptAnalysis] = None,
    template: Optional[Template] = None,
    target_model: str = "general",
) -> str:
    """Assemble an optimized prompt from a rough one. Pure string work."""
    report = analysis or analyze(rough_prompt)
    tpl = template or template_for_intent(report.intent)
    task = _normalize_task(report.prompt)

    sections: List[str] = []

    if not report.signals.get("role"):
        sections.append(f"# Role\n{tpl.role}")

    sections.append(f"# Task\n{task}")

    if not report.signals.get("context") and tpl.context:
        sections.append(f"# Context\n{tpl.context}")

    requirements: List[str] = list(tpl.requirements)
    requirements.extend(_intent_requirements(report.intent))
    if not report.signals.get("audience") and report.intent not in ("creative", "image"):
        requirements.append("Write for a competent reader who is new to this specific topic")
    if not report.signals.get("tone") and report.intent != "creative":
        requirements.append("Use a clear, professional tone without filler")
    if not report.signals.get("examples") and report.intent in (
        "code",
        "analysis",
        "explanation",
        "general",
    ):
        requirements.append("Include one concrete example that illustrates the answer")
    if report.ambiguities:
        requirements.append(
            "If any part of the task is ambiguous, state the interpretation you chose "
            "before answering"
        )

    # Preserve order while removing duplicates.
    seen = set()
    deduped = [r for r in requirements if not (r.lower() in seen or seen.add(r.lower()))]
    sections.append("# Requirements\n" + "\n".join(f"- {r}" for r in deduped))

    if not report.signals.get("output_format"):
        sections.append(f"# Output format\n{tpl.output_format}")

    if report.intent not in ("extraction", "image") and not report.signals.get("constraints"):
        sections.append(
            "# Constraints\n"
            "- Do not pad the answer with restatements of the task\n"
            "- Do not invent facts; mark uncertainty explicitly\n"
            "- Stop once the requirements are met"
        )

    guidance = MODEL_GUIDANCE.get(target_model.strip().lower(), "")
    if guidance:
        sections.append(f"# Model note\n{guidance}")

    return "\n\n".join(sections).strip()
