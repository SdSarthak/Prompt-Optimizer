"""Exception types raised by the package."""


class PromptOptimizerError(Exception):
    """Base class for every error raised by prompt_optimizer."""


class ConfigError(PromptOptimizerError):
    """Configuration is missing or invalid."""


class ProviderError(PromptOptimizerError):
    """An LLM provider could not be reached or returned an unusable answer."""


class TemplateNotFoundError(PromptOptimizerError):
    """The requested template name is not in the library."""
