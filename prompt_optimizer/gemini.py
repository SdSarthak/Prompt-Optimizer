"""Gemini-backed rewriting via the google-genai SDK."""

from __future__ import annotations

import re
import time
from typing import Callable, List, Optional

from .analysis import PromptAnalysis
from .config import Config
from .errors import ProviderError
from .templates import Template

SYSTEM_INSTRUCTION = (
    "You are a prompt engineer. You rewrite rough prompts into precise, "
    "self-contained instructions for large language models.\n"
    "Rules you never break:\n"
    "1. Preserve the user's original intent exactly. Never answer the prompt yourself.\n"
    "2. Output only the rewritten prompt. No preamble, no explanation, no code fences.\n"
    "3. Keep the rewritten prompt directly usable as-is."
)

_LEAD_IN_RE = re.compile(
    r"^\s*(?:here(?:'s| is)[^\n:]*:|rewritten prompt:|optimized prompt:|refined prompt:)\s*",
    re.IGNORECASE,
)

# Trailing chatter after the closing fence ("Hope that helps!") is dropped, but
# only when it is short - anything longer is probably part of the answer.
_MAX_TRAILING_CHATTER = 200

# Substrings that mark an error worth retrying rather than failing on.
_TRANSIENT_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "resource_exhausted",
    "resource exhausted",
    "unavailable",
    "deadline_exceeded",
    "deadline exceeded",
    "timed out",
    "timeout",
    "rate limit",
    "internal error",
    "connection reset",
    "connection aborted",
    "temporarily",
)
_RETRY_BASE_DELAY = 1.0


def build_meta_prompt(
    analysis: PromptAnalysis,
    template: Template,
    target_model: str = "general",
) -> str:
    """The instruction sent to Gemini, grounded in the static analysis."""
    weaknesses: List[str] = []
    for name in analysis.missing:
        weaknesses.append(f"missing {name.replace('_', ' ')}")
    weaknesses.extend(analysis.ambiguities)
    weakness_block = "\n".join(f"- {w}" for w in weaknesses) or "- none detected"
    requirement_block = "\n".join(f"- {r}" for r in template.requirements)

    return (
        "Rewrite the prompt below so it reliably produces a high quality answer.\n\n"
        f"Target model family: {target_model}\n"
        f"Detected intent: {analysis.intent}\n"
        f"Template to follow: {template.name} - {template.description}\n\n"
        "Weaknesses found by static analysis, fix each one that applies:\n"
        f"{weakness_block}\n\n"
        "The rewritten prompt must carry, in this order and only where they add value:\n"
        "a role, the task, the context, explicit requirements, the output format, "
        "and hard constraints.\n\n"
        "Baseline requirements for this kind of task:\n"
        f"{requirement_block}\n\n"
        f"Preferred output format: {template.output_format}\n\n"
        "--- ORIGINAL PROMPT ---\n"
        f"{analysis.prompt}\n"
        "--- END ORIGINAL PROMPT ---\n\n"
        "Return only the rewritten prompt."
    )


def _strip_wrapping_fence(text: str) -> str:
    """Remove a code fence that wraps the whole answer, keeping nested fences.

    The closing fence is looked for from the end, so a rewritten prompt that
    itself contains a fenced example survives intact.
    """
    lines = text.splitlines()
    if not lines or not lines[0].lstrip().startswith("```"):
        return text
    for index in range(len(lines) - 1, 0, -1):
        if lines[index].strip().startswith("```"):
            inner = "\n".join(lines[1:index]).strip()
            trailing = "\n".join(lines[index + 1:]).strip()
            if not inner:
                return "" if not trailing else text
            if len(trailing) <= _MAX_TRAILING_CHATTER:
                return inner
            return text
    return text


def clean_response(text: str) -> str:
    """Strip fences and lead-ins that models add despite being told not to.

    Lead-ins are removed before fences, because "Here is the prompt:" followed
    by a fenced block is the single most common shape of a disobedient reply.
    """
    cleaned = (text or "").strip()
    for _ in range(3):  # bounded: lead-in, fence, and one more of each
        previous = cleaned
        cleaned = _LEAD_IN_RE.sub("", cleaned, count=1).strip()
        cleaned = _strip_wrapping_fence(cleaned).strip()
        if cleaned == previous:
            break
    return cleaned.strip()


def is_transient(exc: BaseException) -> bool:
    """True when retrying the same request has a real chance of succeeding."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    haystack = f"{type(exc).__name__} {exc}".lower()
    return any(marker in haystack for marker in _TRANSIENT_MARKERS)


def _blocked_reason(response: object) -> Optional[str]:
    """Why the model returned no text, when the SDK can tell us."""
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None)
    if block_reason:
        return f"the prompt was blocked ({block_reason})"
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        finish = getattr(candidate, "finish_reason", None)
        if finish and str(finish).upper().split(".")[-1] not in ("STOP", "FINISH_REASON_STOP"):
            return f"generation stopped early ({finish})"
    return None


def response_text(response: object) -> str:
    """Read ``response.text`` without letting the SDK's accessor escape.

    ``google-genai`` exposes ``text`` as a property that raises when the reply
    carries no usable part - a blocked prompt or a truncated candidate. Left
    unguarded that escapes as a bare ValueError and defeats the fallback path.
    """
    try:
        text = getattr(response, "text", None)
    except Exception as exc:  # noqa: BLE001 - the SDK raises several types here
        reason = _blocked_reason(response) or str(exc)
        raise ProviderError(f"Gemini returned no usable text: {reason}") from exc
    if text is None:
        reason = _blocked_reason(response)
        if reason:
            raise ProviderError(f"Gemini returned no usable text: {reason}")
        return ""
    if not isinstance(text, str):
        raise ProviderError(f"Gemini returned {type(text).__name__}, expected text")
    return text


class GeminiRewriter:
    """Thin wrapper over google-genai. The SDK is imported lazily."""

    def __init__(
        self,
        config: Config,
        client: Optional[object] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.config = config
        self._client = client
        self._sleep = sleep or time.sleep

    @property
    def client(self):
        if self._client is None:
            if not self.config.has_api_key:
                raise ProviderError(
                    "GEMINI_API_KEY is not set. Copy .env.example to .env and add a key, "
                    "or run with --engine heuristic."
                )
            try:
                from google import genai
            except ImportError as exc:
                raise ProviderError(
                    "google-genai is not installed. Run: pip install -r requirements.txt"
                ) from exc
            try:
                self._client = genai.Client(
                    api_key=self.config.api_key,
                    # google-genai takes the request timeout in milliseconds.
                    http_options={"timeout": int(self.config.timeout * 1000)},
                )
            except Exception as exc:  # noqa: BLE001 - SDK raises many types
                raise ProviderError(f"could not create the Gemini client: {exc}") from exc
        return self._client

    def _generate(self, meta_prompt: str):
        return self.client.models.generate_content(
            model=self.config.model,
            contents=meta_prompt,
            config={
                "system_instruction": SYSTEM_INSTRUCTION,
                "temperature": self.config.temperature,
            },
        )

    def rewrite(
        self,
        analysis: PromptAnalysis,
        template: Template,
        target_model: str = "general",
    ) -> str:
        """Send the meta-prompt to Gemini and return the cleaned rewrite.

        Transient failures (429, 5xx, timeouts) are retried with an exponential
        backoff; anything else fails immediately so the caller can fall back.
        """
        meta_prompt = build_meta_prompt(analysis, template, target_model)
        attempts = max(1, self.config.max_retries + 1)

        for attempt in range(attempts):
            try:
                response = self._generate(meta_prompt)
            except ProviderError:
                raise
            except Exception as exc:  # noqa: BLE001 - network/SDK errors vary widely
                if attempt + 1 < attempts and is_transient(exc):
                    self._sleep(_RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                suffix = f" after {attempt + 1} attempts" if attempt else ""
                raise ProviderError(f"Gemini request failed{suffix}: {exc}") from exc

            # An empty reply is a decision by the model, not a transport fault,
            # so it is reported rather than retried.
            text = clean_response(response_text(response))
            if not text:
                raise ProviderError("Gemini returned an empty response")
            return text

        raise ProviderError("Gemini request failed: no attempt was made")  # pragma: no cover
