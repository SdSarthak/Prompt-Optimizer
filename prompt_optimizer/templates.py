"""The template library: reusable scaffolds for common prompt shapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .errors import TemplateNotFoundError


@dataclass(frozen=True)
class Template:
    """A scaffold that turns a bare task into a fully specified prompt."""

    name: str
    description: str
    role: str
    intents: tuple = ()
    context: str = ""
    requirements: List[str] = field(default_factory=list)
    output_format: str = ""


_TEMPLATES: Dict[str, Template] = {
    "general": Template(
        name="general",
        description="Balanced default for any task.",
        role="You are an expert assistant with deep knowledge of the subject at hand.",
        intents=("general",),
        context="The reader wants a usable answer, not a survey of the topic.",
        requirements=[
            "Answer the request directly before adding any elaboration",
            "Prefer concrete detail over generalities",
            "State assumptions explicitly when the request is under-specified",
            "Say so plainly if something cannot be determined",
        ],
        output_format="Markdown, with short paragraphs and headings only where they earn their place.",
    ),
    "creative_writing": Template(
        name="creative_writing",
        description="Stories, scripts, copy and other creative output.",
        role="You are an accomplished writer with a distinctive, controlled voice.",
        intents=("creative",),
        context="The piece will be read on its own, so it has to work without extra explanation.",
        requirements=[
            "Show the reader the scene rather than summarising it",
            "Keep one consistent point of view and tense throughout",
            "Avoid cliche phrasing and filler adverbs",
            "End on a deliberate beat rather than trailing off",
        ],
        output_format="Prose only. No preamble, no commentary on your own writing.",
    ),
    "technical_writing": Template(
        name="technical_writing",
        description="Documentation, explainers and technical reference material.",
        role="You are a senior technical writer who has shipped developer documentation.",
        intents=("explanation",),
        context="The reader is competent but new to this specific topic.",
        requirements=[
            "Define each term the first time it appears",
            "Include a minimal, runnable example where code is involved",
            "Call out the common failure mode and how to recover from it",
            "Keep sentences under 25 words",
        ],
        output_format="Markdown with headings, short paragraphs and fenced code blocks.",
    ),
    "code_generation": Template(
        name="code_generation",
        description="Writing, reviewing or refactoring code.",
        role="You are a senior software engineer who values correctness over cleverness.",
        intents=("code",),
        context="The code will be pasted into a real project and has to run as written.",
        requirements=[
            "Produce complete, runnable code with every import included",
            "Handle the error and edge cases the task implies",
            "Follow the idioms of the target language",
            "Explain any non-obvious decision in one line after the code",
        ],
        output_format="A single fenced code block, followed by a short note on how to run it.",
    ),
    "data_analysis": Template(
        name="data_analysis",
        description="Analysis, comparison and evaluation tasks.",
        role="You are a data analyst who reasons from evidence and quantifies claims.",
        intents=("analysis",),
        context="The output feeds a decision, so unsupported conclusions are worse than none.",
        requirements=[
            "State the method before the conclusion",
            "Quantify every claim that can be quantified",
            "Separate what the data shows from what you infer",
            "List the assumptions and the limitations of the analysis",
        ],
        output_format="Markdown: Findings, Method, Assumptions, Limitations - in that order.",
    ),
    "extraction": Template(
        name="extraction",
        description="Structured extraction and classification.",
        role="You are a precise information extraction engine.",
        intents=("extraction",),
        context="The output is parsed by a program, so format compliance matters more than prose.",
        requirements=[
            "Return only the requested structure - no commentary, no code fences",
            "Use null for fields the source does not support",
            "Never invent values that are absent from the input",
            "Preserve the source spelling of extracted values",
        ],
        output_format="A single valid JSON object matching the schema given in the task.",
    ),
    "problem_solving": Template(
        name="problem_solving",
        description="Reasoning through a problem to a recommendation.",
        role="You are a rigorous problem solver who shows the path to the answer.",
        intents=("general",),
        context="The reader needs to be able to check the reasoning, not just trust the result.",
        requirements=[
            "Restate the problem and the constraints in your own words first",
            "Work through the reasoning step by step",
            "Consider at least one alternative and say why it loses",
            "Finish with a single, clearly marked recommendation",
        ],
        output_format="Numbered reasoning steps, then a 'Recommendation' section.",
    ),
    "conversational": Template(
        name="conversational",
        description="Chatbot, assistant and support personas.",
        role="You are a support assistant representing the product with care and accuracy.",
        intents=("conversation",),
        context="Replies are shown directly to an end user in a chat window.",
        requirements=[
            "Stay in persona for the whole reply",
            "Acknowledge the user's situation before solving it",
            "Never invent policy, pricing or capabilities",
            "Escalate to a human when the request is outside your scope",
        ],
        output_format="Two to five short conversational sentences. No headings, no bullet lists.",
    ),
    "image_generation": Template(
        name="image_generation",
        description="Prompts for text-to-image models.",
        role="You are a prompt engineer for text-to-image models.",
        intents=("image",),
        context="The prompt is consumed by an image model that has no memory of this conversation.",
        requirements=[
            "Name the subject, then the setting, then the lighting, then the style",
            "Specify composition, camera angle and aspect ratio",
            "Add an explicit list of things to exclude",
            "Keep it to one dense paragraph of comma-separated phrases",
        ],
        output_format="One paragraph of the positive prompt, then a line beginning 'Negative:'.",
    ),
}

# Intent -> template used when the caller does not name one.
_INTENT_DEFAULTS: Dict[str, str] = {
    "code": "code_generation",
    "creative": "creative_writing",
    "image": "image_generation",
    "analysis": "data_analysis",
    "extraction": "extraction",
    "explanation": "technical_writing",
    "conversation": "conversational",
    "general": "general",
}


def list_templates() -> List[Template]:
    """Every template in the library, in declaration order."""
    return list(_TEMPLATES.values())


def template_names() -> List[str]:
    return list(_TEMPLATES)


def get_template(name: str) -> Template:
    """Look up a template by name, case-insensitively."""
    try:
        return _TEMPLATES[name.strip().lower()]
    except KeyError:
        raise TemplateNotFoundError(
            f"unknown template {name!r}; available: {', '.join(_TEMPLATES)}"
        ) from None


def template_for_intent(intent: str) -> Template:
    """The template that best fits a detected intent."""
    return _TEMPLATES[_INTENT_DEFAULTS.get(intent, "general")]
