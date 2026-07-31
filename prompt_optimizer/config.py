"""Runtime configuration, read from the environment (and an optional .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .errors import ConfigError

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_ENGINE = "auto"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2
VALID_ENGINES = ("auto", "heuristic", "llm")

_ENV_LOADED = False


def _load_dotenv_once() -> None:
    """Load a .env from the project root, once per process. Never fatal."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a declared dep
        return
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number, got {raw!r}") from None


@dataclass
class Config:
    """Everything the optimizer needs to know about its environment."""

    api_key: Optional[str] = None
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    engine: str = DEFAULT_ENGINE
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES

    def __post_init__(self) -> None:
        if self.engine not in VALID_ENGINES:
            raise ConfigError(
                f"engine must be one of {', '.join(VALID_ENGINES)}, got {self.engine!r}"
            )
        try:
            self.temperature = float(self.temperature)
        except (TypeError, ValueError):
            raise ConfigError(f"temperature must be a number, got {self.temperature!r}") from None
        # NaN fails every comparison, so it would slip past a plain range check.
        if not self.temperature == self.temperature or not 0.0 <= self.temperature <= 2.0:
            raise ConfigError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ConfigError("model must be a non-empty string")
        self.model = self.model.strip()
        try:
            self.timeout = float(self.timeout)
        except (TypeError, ValueError):
            raise ConfigError(f"timeout must be a number, got {self.timeout!r}") from None
        if not self.timeout > 0 or self.timeout == float("inf"):
            raise ConfigError(f"timeout must be a positive number of seconds, got {self.timeout}")
        try:
            self.max_retries = int(self.max_retries)
        except (TypeError, ValueError):
            raise ConfigError(f"max_retries must be a whole number, got {self.max_retries!r}") from None
        if self.max_retries < 0:
            raise ConfigError(f"max_retries cannot be negative, got {self.max_retries}")

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def resolve_engine(self) -> str:
        """Turn 'auto' into the engine that will actually be used."""
        if self.engine != "auto":
            return self.engine
        return "llm" if self.has_api_key else "heuristic"


def load_config(
    engine: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> Config:
    """Build a Config from the environment, with explicit overrides winning."""
    _load_dotenv_once()

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key is not None and not key.strip():
        key = None

    return Config(
        api_key=key,
        model=model or os.environ.get("PROMPT_OPTIMIZER_MODEL") or DEFAULT_MODEL,
        temperature=(
            temperature
            if temperature is not None
            else _env_float("PROMPT_OPTIMIZER_TEMPERATURE", DEFAULT_TEMPERATURE)
        ),
        engine=engine or os.environ.get("PROMPT_OPTIMIZER_ENGINE") or DEFAULT_ENGINE,
        timeout=(
            timeout
            if timeout is not None
            else _env_float("PROMPT_OPTIMIZER_TIMEOUT", DEFAULT_TIMEOUT)
        ),
        max_retries=(
            max_retries
            if max_retries is not None
            else _env_int("PROMPT_OPTIMIZER_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        ),
    )
