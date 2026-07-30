"""Prompt Optimizer - analyse and rewrite rough prompts into effective ones."""

from .analysis import PromptAnalysis, analyze
from .compare import Comparison, compare
from .config import Config, load_config
from .errors import ProviderError, PromptOptimizerError, TemplateNotFoundError
from .optimizer import OptimizationResult, PromptOptimizer
from .templates import Template, get_template, list_templates

__version__ = "1.0.0"

__all__ = [
    "Comparison",
    "Config",
    "OptimizationResult",
    "PromptAnalysis",
    "PromptOptimizer",
    "PromptOptimizerError",
    "ProviderError",
    "Template",
    "TemplateNotFoundError",
    "analyze",
    "compare",
    "get_template",
    "list_templates",
    "load_config",
    "__version__",
]
