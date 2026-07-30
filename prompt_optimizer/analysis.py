"""Static analysis of a prompt: readability, intent, ambiguity and scoring.

Everything here is deterministic and offline. No network, no model calls, so the
same prompt always produces the same report and the scores can be unit tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Signal vocabularies
# --------------------------------------------------------------------------- #

ROLE_MARKERS = ("you are", "act as", "acting as", "as an expert", "your role", "you're a")

CONTEXT_MARKERS = (
    "context",
    "background",
    "scenario",
    "given that",
    "the audience",
    "for a",
    "our team",
    "the user is",
)

CONSTRAINT_MARKERS = (
    "must",
    "must not",
    "should",
    "do not",
    "don't",
    "never",
    "always",
    "at most",
    "at least",
    "no more than",
    "limit",
    "avoid",
    "only",
    "required",
    "constraint",
)

EXAMPLE_MARKERS = ("example", "for instance", "e.g.", "such as", "sample input", "few-shot")

FORMAT_MARKERS = (
    "format",
    "json",
    "yaml",
    "csv",
    "markdown",
    "table",
    "bullet",
    "bullet points",
    "numbered list",
    "schema",
    "return a",
    "respond with",
    "output:",
    "structure your",
)

AUDIENCE_MARKERS = (
    "audience",
    "for beginners",
    "for a beginner",
    "for experts",
    "non-technical",
    "for developers",
    "for students",
    "reader",
    "five-year-old",
)

TONE_MARKERS = (
    "tone",
    "formal",
    "informal",
    "casual",
    "professional",
    "friendly",
    "concise",
    "playful",
    "academic",
    "voice",
)

STEP_MARKERS = ("step by step", "step-by-step", "first,", "then,", "finally,", "reasoning")

VAGUE_TERMS = (
    "good",
    "nice",
    "better",
    "stuff",
    "things",
    "something",
    "some",
    "a few",
    "several",
    "etc",
    "and so on",
    "appropriate",
    "proper",
    "as needed",
    "if possible",
    "kind of",
    "sort of",
    "very",
    "really",
    "interesting",
    "great",
)

CONFLICT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("brief", "comprehensive"),
    ("brief", "detailed"),
    ("short", "in depth"),
    ("concise", "exhaustive"),
    ("simple", "advanced"),
    ("summarize", "explain in detail"),
)

INTENT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "code": (
        "code",
        "function",
        "script",
        "api",
        "bug",
        "refactor",
        "debug",
        "python",
        "javascript",
        "sql",
        "website",
        "app",
        "class",
        "test",
    ),
    "creative": (
        "story",
        "poem",
        "song",
        "novel",
        "character",
        "screenplay",
        "creative",
        "fiction",
        "tagline",
        "slogan",
    ),
    "image": ("image", "picture", "photo", "illustration", "render", "logo", "artwork", "drawing"),
    "analysis": (
        "analyze",
        "analyse",
        "compare",
        "evaluate",
        "assess",
        "review",
        "insight",
        "data",
        "metric",
        "trend",
        "report",
    ),
    "extraction": (
        "extract",
        "parse",
        "classify",
        "label",
        "categorize",
        "tag",
        "identify",
        "list all",
        "find all",
    ),
    "explanation": (
        "explain",
        "what is",
        "how does",
        "why",
        "describe",
        "teach",
        "summarize",
        "define",
        "overview",
    ),
    "conversation": (
        "chatbot",
        "assistant",
        "customer",
        "support agent",
        "reply to",
        "respond to the user",
        "persona",
    ),
}

ACTION_VERBS = (
    "write",
    "create",
    "generate",
    "explain",
    "summarize",
    "analyze",
    "analyse",
    "build",
    "design",
    "draft",
    "list",
    "compare",
    "translate",
    "review",
    "refactor",
    "extract",
    "classify",
    "convert",
    "rewrite",
    "help",
    "make",
    "describe",
    "plan",
)

PLACEHOLDER_RE = re.compile(r"(\{\{\s*\w+\s*\}\}|<[A-Z_]{3,}>|\[[A-Z_]{3,}\])")
_WORD_RE = re.compile(r"[A-Za-z']+")
_SENTENCE_RE = re.compile(r"[.!?\n]+")
_VOWEL_GROUPS_RE = re.compile(r"[aeiouy]+")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round(value: float) -> float:
    return round(_clamp(value), 3)


def count_syllables(word: str) -> int:
    """Approximate syllable count for a single English word."""
    word = word.lower().strip("'")
    if not word:
        return 0
    groups = _VOWEL_GROUPS_RE.findall(word)
    count = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ee", "ye")) and count > 1:
        count -= 1
    return max(count, 1)


def flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease (0 = very hard, 100 = very easy). Clamped to 0-100."""
    words = _WORD_RE.findall(text)
    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    if not words or not sentences:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    score = (
        206.835
        - 1.015 * (len(words) / len(sentences))
        - 84.6 * (syllables / len(words))
    )
    return round(max(0.0, min(100.0, score)), 1)


def _contains_any(haystack: str, needles) -> bool:
    return any(needle in haystack for needle in needles)


def detect_intent(prompt: str) -> str:
    """Best-guess of what the prompt is asking for."""
    lowered = prompt.lower()
    scores: Dict[str, int] = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lowered)
        if hits:
            scores[intent] = hits
    if not scores:
        return "general"
    best = max(scores.values())
    # Deterministic tie-break: declaration order of INTENT_KEYWORDS.
    for intent in INTENT_KEYWORDS:
        if scores.get(intent) == best:
            return intent
    return "general"


def find_ambiguities(prompt: str) -> List[str]:
    """Concrete, actionable ambiguity findings - not a generic warning list."""
    lowered = prompt.lower()
    words = _WORD_RE.findall(lowered)
    findings: List[str] = []

    vague_hits = sorted({term for term in VAGUE_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered)})
    if vague_hits:
        findings.append("vague wording: " + ", ".join(f"'{t}'" for t in vague_hits[:6]))

    for left, right in CONFLICT_PAIRS:
        if left in lowered and right in lowered:
            findings.append(f"conflicting instructions: '{left}' vs '{right}'")

    placeholders = PLACEHOLDER_RE.findall(prompt)
    if placeholders:
        findings.append("unfilled placeholder(s): " + ", ".join(sorted(set(placeholders))[:5]))

    first_words = words[:2]
    if first_words and first_words[0] in ("it", "this", "that", "they", "those"):
        findings.append(f"opens with an unresolved pronoun: '{first_words[0]}'")

    for sentence in (s.strip() for s in _SENTENCE_RE.split(prompt)):
        sentence_words = _WORD_RE.findall(sentence)
        if len(sentence_words) > 45:
            findings.append(f"run-on sentence ({len(sentence_words)} words) is hard to follow")
            break

    if len(words) < 6:
        findings.append("prompt is too short to constrain the model's output")

    if "?" not in prompt and not any(
        re.match(rf"^{verb}\b", lowered) for verb in ACTION_VERBS
    ) and not _contains_any(lowered, ROLE_MARKERS):
        findings.append("no explicit request: state the task with an action verb")

    return findings


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


@dataclass
class PromptAnalysis:
    """A full static report on a single prompt."""

    prompt: str
    char_count: int
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    reading_ease: float
    intent: str
    signals: Dict[str, bool] = field(default_factory=dict)
    ambiguities: List[str] = field(default_factory=list)
    clarity: float = 0.0
    specificity: float = 0.0
    context_completeness: float = 0.0
    structure_quality: float = 0.0
    effectiveness: float = 0.0
    suggestions: List[str] = field(default_factory=list)

    @property
    def scores(self) -> Dict[str, float]:
        return {
            "clarity": self.clarity,
            "specificity": self.specificity,
            "context_completeness": self.context_completeness,
            "structure_quality": self.structure_quality,
            "effectiveness": self.effectiveness,
        }

    @property
    def missing(self) -> List[str]:
        """Signals the prompt does not carry, in report order."""
        return [name for name, present in self.signals.items() if not present]

    def to_dict(self) -> Dict[str, object]:
        return {
            "char_count": self.char_count,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "avg_sentence_length": self.avg_sentence_length,
            "reading_ease": self.reading_ease,
            "intent": self.intent,
            "signals": dict(self.signals),
            "ambiguities": list(self.ambiguities),
            "scores": self.scores,
            "suggestions": list(self.suggestions),
        }

    def render(self) -> str:
        lines = [
            "PROMPT ANALYSIS",
            "-" * 60,
            f"  intent            : {self.intent}",
            f"  words / sentences : {self.word_count} / {self.sentence_count}",
            f"  avg sentence len  : {self.avg_sentence_length}",
            f"  reading ease      : {self.reading_ease}",
            "",
            "SCORES (0-1)",
            "-" * 60,
        ]
        for name, value in self.scores.items():
            bar = "#" * int(round(value * 20))
            lines.append(f"  {name:<20} {value:>5.2f}  {bar}")

        lines += ["", "SIGNALS", "-" * 60]
        for name, present in self.signals.items():
            lines.append(f"  [{'x' if present else ' '}] {name}")

        lines += ["", "AMBIGUITIES", "-" * 60]
        lines += [f"  - {item}" for item in self.ambiguities] or ["  none detected"]

        lines += ["", "SUGGESTIONS", "-" * 60]
        lines += [f"  - {item}" for item in self.suggestions] or ["  none - prompt looks solid"]
        return "\n".join(lines)


def _collect_signals(prompt: str) -> Dict[str, bool]:
    lowered = prompt.lower()
    return {
        "role": _contains_any(lowered, ROLE_MARKERS),
        "context": _contains_any(lowered, CONTEXT_MARKERS),
        "constraints": _contains_any(lowered, CONSTRAINT_MARKERS),
        "examples": _contains_any(lowered, EXAMPLE_MARKERS),
        "output_format": _contains_any(lowered, FORMAT_MARKERS),
        "audience": _contains_any(lowered, AUDIENCE_MARKERS),
        "tone": _contains_any(lowered, TONE_MARKERS),
        "reasoning_steps": _contains_any(lowered, STEP_MARKERS),
    }


def _clarity_score(prompt: str, words: int, avg_sentence_length: float, ease: float,
                   ambiguities: List[str]) -> float:
    lowered = prompt.lower().lstrip()
    score = 0.2
    if any(lowered.startswith(verb) for verb in ACTION_VERBS) or "?" in prompt:
        score += 0.2
    if words >= 12:
        score += 0.15
    if words >= 40:
        score += 0.05
    if 0 < avg_sentence_length <= 25:
        score += 0.2
    elif avg_sentence_length <= 35:
        score += 0.1
    if 30.0 <= ease <= 80.0:
        score += 0.2
    elif ease > 0:
        score += 0.1
    score -= 0.09 * len(ambiguities)
    return _round(score)


def _specificity_score(prompt: str, signals: Dict[str, bool], words: int) -> float:
    lowered = prompt.lower()
    score = 0.1
    if signals["constraints"]:
        score += 0.2
    if signals["output_format"]:
        score += 0.2
    if signals["audience"]:
        score += 0.1
    if signals["tone"]:
        score += 0.1
    if re.search(r"\b\d+\b", prompt):
        score += 0.15
    if re.search(r"\b(word|words|sentence|sentences|paragraph|paragraphs|bullet|line|lines)\b", lowered):
        score += 0.1
    if words >= 25:
        score += 0.1
    vague_count = sum(1 for term in VAGUE_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered))
    score -= 0.05 * vague_count
    return _round(score)


def _context_score(signals: Dict[str, bool], words: int) -> float:
    score = 0.05
    if signals["role"]:
        score += 0.25
    if signals["context"]:
        score += 0.25
    if signals["examples"]:
        score += 0.2
    if signals["audience"]:
        score += 0.15
    if words >= 30:
        score += 0.1
    return _round(score)


def _structure_score(prompt: str, signals: Dict[str, bool]) -> float:
    score = 0.15
    lines = [line for line in prompt.splitlines() if line.strip()]
    if len(lines) > 1:
        score += 0.15
    if re.search(r"^\s*(?:[-*•]|\d+[.)])\s+", prompt, re.MULTILINE):
        score += 0.25
    if re.search(r"^\s*#{1,6}\s+\S|^[A-Z][A-Za-z ]{2,30}:\s*$", prompt, re.MULTILINE):
        score += 0.2
    if "```" in prompt or re.search(r"</?[a-z_]+>", prompt) or "---" in prompt:
        score += 0.15
    if signals["reasoning_steps"]:
        score += 0.1
    return _round(score)


def _build_suggestions(signals: Dict[str, bool], ambiguities: List[str], words: int,
                       avg_sentence_length: float) -> List[str]:
    suggestions: List[str] = []
    if not signals["role"]:
        suggestions.append("Open with a role so the model knows which expertise to apply.")
    if not signals["context"]:
        suggestions.append("Add background: who the output is for and what it will be used for.")
    if not signals["output_format"]:
        suggestions.append("State the output format explicitly (markdown, JSON schema, table, ...).")
    if not signals["constraints"]:
        suggestions.append("Add constraints: length, what to include, what to avoid.")
    if not signals["examples"]:
        suggestions.append("Include one worked example to anchor style and depth.")
    if not signals["audience"]:
        suggestions.append("Name the audience and their level of expertise.")
    if not signals["tone"]:
        suggestions.append("Specify the tone or voice you expect.")
    if words < 12:
        suggestions.append("Expand the prompt - under a dozen words leaves too much to chance.")
    if avg_sentence_length > 35:
        suggestions.append("Split long sentences into short, single-instruction lines.")
    if ambiguities:
        suggestions.append("Replace vague wording with measurable requirements.")
    return suggestions


def analyze(prompt: str) -> PromptAnalysis:
    """Produce a full static analysis of ``prompt``."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    text = prompt.strip()
    words = _WORD_RE.findall(text)
    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_length = round(word_count / sentence_count, 2) if sentence_count else 0.0
    ease = flesch_reading_ease(text)

    signals = _collect_signals(text)
    ambiguities = find_ambiguities(text) if text else ["prompt is empty"]

    clarity = _clarity_score(text, word_count, avg_sentence_length, ease, ambiguities)
    specificity = _specificity_score(text, signals, word_count)
    context = _context_score(signals, word_count)
    structure = _structure_score(text, signals)
    effectiveness = _round(
        0.30 * clarity + 0.30 * specificity + 0.20 * context + 0.20 * structure
    )

    return PromptAnalysis(
        prompt=text,
        char_count=len(text),
        word_count=word_count,
        sentence_count=sentence_count,
        avg_sentence_length=avg_sentence_length,
        reading_ease=ease,
        intent=detect_intent(text) if text else "general",
        signals=signals,
        ambiguities=ambiguities,
        clarity=clarity,
        specificity=specificity,
        context_completeness=context,
        structure_quality=structure,
        effectiveness=effectiveness,
        suggestions=_build_suggestions(signals, ambiguities, word_count, avg_sentence_length),
    )
