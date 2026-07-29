"""Anthropic SDK wrapper.

Owns the single client instance, retry policy, and prompt caching. Raw SDK
exceptions are wrapped in `AssistantError` so they don't leak to the CLI.
"""

from __future__ import annotations

import random
import time
from typing import Any

import anthropic

from assistant import config
from assistant.errors import AssistantError

# Instantiate once at module load. The SDK reads ANTHROPIC_API_KEY from the
# environment; `config` has already loaded .env by this point.
_client = anthropic.Anthropic()

# Errors worth retrying with exponential backoff.
_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APIStatusError,
    anthropic.APIConnectionError,
)


class ClaudeClient:
    """Thin wrapper over the Messages API."""

    def __init__(
        self,
        model: str | None = None,
        max_retries: int = config.MAX_RETRIES,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model or config.get_model()
        self.max_retries = max_retries
        self._client = client or _client

    # -- public API ----------------------------------------------------------

    def complete(
        self,
        messages: list[dict],
        system: str | list[dict] | None = None,
        stream: bool = False,
        max_tokens: int = config.MAX_TOKENS,
        cache_system: bool = True,
    ) -> str:
        """Send `messages` and return the concatenated text response.

        `system` is sent with a `cache_control` breakpoint by default so
        repeated calls against the same prompt aren't re-billed for those
        tokens. Retries transient failures with exponential backoff.
        """
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system is not None:
            request["system"] = _build_system(system, cache=cache_system)

        response = self._send_with_retry(request, stream=stream)
        return _extract_text(response)

    def complete_raw(
        self,
        messages: list[dict],
        system: str | list[dict] | None = None,
        stream: bool = False,
        max_tokens: int = config.MAX_TOKENS,
        cache_system: bool = True,
    ) -> anthropic.types.Message:
        """As `complete`, but returns the full Message (for inspecting `usage`).

        Used to verify prompt-cache hits via `usage.cache_read_input_tokens`.
        """
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system is not None:
            request["system"] = _build_system(system, cache=cache_system)

        return self._send_with_retry(request, stream=stream)

    # -- internals -----------------------------------------------------------

    def _send_with_retry(
        self, request: dict[str, Any], stream: bool
    ) -> anthropic.types.Message:
        """Call the API, retrying transient failures up to `max_retries` times."""
        last_error: Exception | None = None
        delay = config.INITIAL_BACKOFF_SECONDS

        # One initial attempt plus `max_retries` retries.
        for attempt in range(self.max_retries + 1):
            try:
                return self._send(request, stream=stream)
            except anthropic.APIStatusError as exc:
                # 4xx other than 429 won't succeed on retry.
                if not _is_retryable_status(exc):
                    raise AssistantError(
                        f"Anthropic API error ({exc.status_code}): {exc}"
                    ) from exc
                last_error = exc
            except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
                last_error = exc
            except anthropic.AnthropicError as exc:
                raise AssistantError(f"Anthropic SDK error: {exc}") from exc

            if attempt < self.max_retries:
                # Jitter avoids synchronized retries across concurrent callers.
                time.sleep(min(delay, config.MAX_BACKOFF_SECONDS) + random.uniform(0, 0.5))
                delay *= config.BACKOFF_MULTIPLIER

        raise AssistantError(
            f"Anthropic API call failed after {self.max_retries} retries: {last_error}"
        ) from last_error

    def _send(
        self, request: dict[str, Any], stream: bool
    ) -> anthropic.types.Message:
        if stream:
            with self._client.messages.stream(**request) as response_stream:
                return response_stream.get_final_message()
        return self._client.messages.create(**request)


# -- helpers -----------------------------------------------------------------


def _is_retryable_status(exc: anthropic.APIStatusError) -> bool:
    """Retry on rate limits and server errors only."""
    status = getattr(exc, "status_code", None)
    if status is None:
        return True
    return status == 429 or status >= 500


def _build_system(system: str | list[dict], cache: bool) -> list[dict]:
    """Render the system prompt as blocks, with a cache breakpoint on the last.

    Static document content passed in as system blocks is cached alongside the
    prompt, so repeated calls don't re-bill the full document tokens.
    """
    blocks = [{"type": "text", "text": system}] if isinstance(system, str) else [
        dict(block) for block in system
    ]

    if cache and blocks:
        blocks[-1]["cache_control"] = {"type": "ephemeral"}

    return blocks


def _extract_text(response: anthropic.types.Message) -> str:
    """Concatenate the text blocks of a response.

    Checks `stop_reason` first: a refusal carries no usable content.
    """
    if getattr(response, "stop_reason", None) == "refusal":
        raise AssistantError(
            "The model declined to respond to this request."
        )

    parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ]
    return "".join(parts).strip()
