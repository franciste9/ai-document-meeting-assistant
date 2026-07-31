"""Environment loading and constants.

The model name is read here rather than hardcoded per call, so every request
goes through one place.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from assistant.errors import AssistantError

load_dotenv()

# Model used for all completions. Overridable via CLAUDE_MODEL in .env.
DEFAULT_MODEL = "claude-sonnet-5"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)

# Documents above this token estimate get chunked; below it, passed whole.
DEFAULT_CHUNK_TOKEN_THRESHOLD = 150_000

# Retry policy for transient API failures (spec: max 3 retries, exp. backoff).
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0
MAX_BACKOFF_SECONDS = 30.0

# Default output cap. Streaming is used above this to avoid HTTP timeouts.
MAX_TOKENS = 8_192

# Path to the local SQLite database.
DB_PATH = os.getenv("ASSISTANT_DB_PATH", "assistant.db")


def _int_from_env(name: str, default: int) -> int:
    """Read an int env var, falling back to `default` when unset or blank."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise AssistantError(
            f"{name} must be an integer, got {raw!r}"
        ) from exc


def get_chunk_token_threshold() -> int:
    """The token estimate above which a document is chunked."""
    return _int_from_env("CHUNK_TOKEN_THRESHOLD", DEFAULT_CHUNK_TOKEN_THRESHOLD)


def get_model() -> str:
    """The model name for all completions."""
    return os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)


def get_api_token() -> str | None:
    """The inbound API token, or None if auth is disabled (unset).

    Distinct from `get_api_key()`: that is the outbound Anthropic credential,
    this gates inbound calls to our own HTTP routes. Returning None when unset
    is load-bearing — it keeps local dev and the existing test suite running
    without a token, and the gate activates the moment the env var is set.
    """
    token = os.getenv("ASSISTANT_API_TOKEN")
    return token.strip() if token and token.strip() else None


def get_api_key() -> str:
    """The Anthropic API key, or raise if it isn't configured."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key or not key.strip():
        raise AssistantError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key
