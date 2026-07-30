"""Test-session setup.

Living at the project root, this file also puts the root on sys.path so the
tests import ``prompt_optimizer`` without an editable install.
"""

import pytest

from prompt_optimizer import config as config_module


@pytest.fixture(autouse=True)
def never_read_the_real_dotenv(monkeypatch):
    """Keep a developer's real .env out of the test environment."""
    monkeypatch.setattr(config_module, "_load_dotenv_once", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("PROMPT_OPTIMIZER_MODEL", raising=False)
    monkeypatch.delenv("PROMPT_OPTIMIZER_TEMPERATURE", raising=False)
    monkeypatch.delenv("PROMPT_OPTIMIZER_ENGINE", raising=False)
