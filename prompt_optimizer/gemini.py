"""Gemini-backed rewriting via the google-genai SDK."""

from __future__ import annotations

import re
from typing import List, Optional

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

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)
_LEAD_IN_RE = re.compile(
    r"^\s*(?:here(?:'s| is)[^\n:]*:|rewritten prompt:|optimized prompt:|refined prompt:)\s*",
    re.IGNORECASE,
)


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


def clean_response(text: str) -> str:
    """Strip fences and lead-ins that models add despite being told not to."""
    cleaned = (text or "").strip()
    match = _FENCE_RE.match(cleaned)
    if match:
        cleaned = match.group(1).strip()
    cleaned = _LEAD_IN_RE.sub("", cleaned, count=1)
    return cleaned.strip()


class GeminiRewriter:
    """Thin wrapper over google-genai. The SDK is imported lazily."""

    def __init__(self, config: Config, client: Optional[object] = None) -> None:
        self.config = config
        self._client = client

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
                self._client = genai.Client(api_key=self.config.api_key)
            except Exception as exc:  # noqa: BLE001 - SDK raises many types
                raise ProviderError(f"could not create the Gemini client: {exc}") from exc
        return self._client

    def rewrite(
        self,
        analysis: PromptAnalysis,
        template: Template,
        target_model: str = "general",
    ) -> str:
        """Send the meta-prompt to Gemini and return the cleaned rewrite."""
        meta_prompt = build_meta_prompt(analysis, template, target_model)
        try:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=meta_prompt,
                config={
                    "system_instruction": SYSTEM_INSTRUCTION,
                    "temperature": self.config.temperature,
                },
            )
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - network/SDK errors vary widely
            raise ProviderError(f"Gemini request failed: {exc}") from exc

        text = clean_response(getattr(response, "text", "") or "")
        if not text:
            raise ProviderError("Gemini returned an empty response")
        return text
