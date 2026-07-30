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
        raise ConfigError(f"{name} must be a number, got {raw!r}")


@dataclass
class Config:
    """Everything the optimizer needs to know about its environment."""

    api_key: Optional[str] = None
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    engine: str = DEFAULT_ENGINE

    def __post_init__(self) -> None:
        if self.engine not in VALID_ENGINES:
            raise ConfigError(
                f"engine must be one of {', '.join(VALID_ENGINES)}, got {self.engine!r}"
            )
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")

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
    )
