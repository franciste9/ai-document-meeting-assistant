"""Tests for the Anthropic SDK wrapper.

All tests run against a fake client — no network calls, no API key required.
"""

from types import SimpleNamespace

import anthropic
import httpx
import pytest

from assistant.client import ClaudeClient, _build_system, _extract_text
from assistant.errors import AssistantError


# --- fakes -------------------------------------------------------------------


def make_message(text="hello", stop_reason="end_turn", **usage):
    """A stand-in for anthropic.types.Message."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=usage.get("input_tokens", 10),
            output_tokens=usage.get("output_tokens", 5),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        ),
    )


def make_status_error(status_code):
    """A real APIStatusError with the given status code."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    return anthropic.APIStatusError(
        f"error {status_code}", response=response, body=None
    )


def make_rate_limit_error():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    return anthropic.RateLimitError("rate limited", response=response, body=None)


def make_connection_error():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(request=request)


class FakeMessages:
    """Records calls and replays a scripted sequence of results."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return FakeStream(result)


class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get_final_message(self):
        return self.message


class FakeAnthropic:
    def __init__(self, results):
        self.messages = FakeMessages(results)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Make backoff instant so retry tests don't wait."""
    monkeypatch.setattr("assistant.client.time.sleep", lambda _: None)


def client_with(results, **kwargs):
    fake = FakeAnthropic(results)
    return ClaudeClient(model="claude-sonnet-5", client=fake, **kwargs), fake


# --- basic round trip --------------------------------------------------------


class TestComplete:
    def test_returns_text(self):
        client, _ = client_with([make_message("the answer")])
        assert client.complete([{"role": "user", "content": "q"}]) == "the answer"

    def test_sends_model_from_config(self):
        client, fake = client_with([make_message()])
        client.complete([{"role": "user", "content": "q"}])
        assert fake.messages.calls[0]["model"] == "claude-sonnet-5"

    def test_model_is_not_hardcoded_per_call(self):
        fake = FakeAnthropic([make_message()])
        client = ClaudeClient(model="claude-opus-5", client=fake)
        client.complete([{"role": "user", "content": "q"}])
        assert fake.messages.calls[0]["model"] == "claude-opus-5"

    def test_sends_messages_through(self):
        client, fake = client_with([make_message()])
        messages = [{"role": "user", "content": "q"}]
        client.complete(messages)
        assert fake.messages.calls[0]["messages"] == messages

    def test_omits_system_when_not_given(self):
        client, fake = client_with([make_message()])
        client.complete([{"role": "user", "content": "q"}])
        assert "system" not in fake.messages.calls[0]

    def test_max_tokens_is_sent(self):
        client, fake = client_with([make_message()])
        client.complete([{"role": "user", "content": "q"}], max_tokens=512)
        assert fake.messages.calls[0]["max_tokens"] == 512

    def test_concatenates_multiple_text_blocks(self):
        message = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="part one "),
                SimpleNamespace(type="text", text="part two"),
            ],
            stop_reason="end_turn",
        )
        client, _ = client_with([message])
        assert client.complete([{"role": "user", "content": "q"}]) == "part one part two"

    def test_ignores_non_text_blocks(self):
        message = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="hmm"),
                SimpleNamespace(type="text", text="answer"),
            ],
            stop_reason="end_turn",
        )
        client, _ = client_with([message])
        assert client.complete([{"role": "user", "content": "q"}]) == "answer"


class TestStreaming:
    def test_stream_returns_final_message_text(self):
        client, fake = client_with([make_message("streamed")])
        result = client.complete([{"role": "user", "content": "q"}], stream=True)
        assert result == "streamed"

    def test_stream_uses_the_stream_endpoint(self):
        client, fake = client_with([make_message()])
        client.complete([{"role": "user", "content": "q"}], stream=True)
        assert len(fake.messages.calls) == 1


# --- prompt caching ----------------------------------------------------------


class TestPromptCaching:
    def test_system_string_becomes_a_text_block(self):
        client, fake = client_with([make_message()])
        client.complete([{"role": "user", "content": "q"}], system="be brief")

        system = fake.messages.calls[0]["system"]
        assert system[0]["type"] == "text"
        assert system[0]["text"] == "be brief"

    def test_cache_control_on_last_system_block(self):
        client, fake = client_with([make_message()])
        client.complete([{"role": "user", "content": "q"}], system="be brief")

        system = fake.messages.calls[0]["system"]
        assert system[-1]["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_only_on_last_block(self):
        client, fake = client_with([make_message()])
        blocks = [
            {"type": "text", "text": "instructions"},
            {"type": "text", "text": "a long document"},
        ]
        client.complete([{"role": "user", "content": "q"}], system=blocks)

        system = fake.messages.calls[0]["system"]
        assert "cache_control" not in system[0]
        assert system[1]["cache_control"] == {"type": "ephemeral"}

    def test_caching_can_be_disabled(self):
        client, fake = client_with([make_message()])
        client.complete(
            [{"role": "user", "content": "q"}], system="be brief", cache_system=False
        )

        assert "cache_control" not in fake.messages.calls[0]["system"][0]

    def test_does_not_mutate_caller_blocks(self):
        client, _ = client_with([make_message()])
        blocks = [{"type": "text", "text": "instructions"}]
        client.complete([{"role": "user", "content": "q"}], system=blocks)

        assert "cache_control" not in blocks[0]

    def test_build_system_from_string(self):
        assert _build_system("hi", cache=False) == [{"type": "text", "text": "hi"}]

    def test_build_system_empty_list_is_safe(self):
        assert _build_system([], cache=True) == []

    def test_complete_raw_exposes_cache_usage(self):
        client, _ = client_with([make_message(cache_read_input_tokens=1234)])
        response = client.complete_raw([{"role": "user", "content": "q"}], system="s")

        assert response.usage.cache_read_input_tokens == 1234


# --- retry behaviour ---------------------------------------------------------


class TestRetry:
    def test_retries_rate_limit_then_succeeds(self):
        client, fake = client_with([make_rate_limit_error(), make_message("ok")])
        assert client.complete([{"role": "user", "content": "q"}]) == "ok"
        assert len(fake.messages.calls) == 2

    def test_retries_connection_error_then_succeeds(self):
        client, fake = client_with([make_connection_error(), make_message("ok")])
        assert client.complete([{"role": "user", "content": "q"}]) == "ok"

    def test_retries_server_error_then_succeeds(self):
        client, fake = client_with([make_status_error(529), make_message("ok")])
        assert client.complete([{"role": "user", "content": "q"}]) == "ok"

    def test_gives_up_after_max_retries(self):
        client, fake = client_with(
            [make_rate_limit_error() for _ in range(4)], max_retries=3
        )
        with pytest.raises(AssistantError, match="after 3 retries"):
            client.complete([{"role": "user", "content": "q"}])

    def test_makes_exactly_max_retries_plus_one_attempts(self):
        client, fake = client_with(
            [make_rate_limit_error() for _ in range(4)], max_retries=3
        )
        with pytest.raises(AssistantError):
            client.complete([{"role": "user", "content": "q"}])

        assert len(fake.messages.calls) == 4

    def test_does_not_retry_client_errors(self):
        client, fake = client_with([make_status_error(400), make_message("never")])
        with pytest.raises(AssistantError, match="400"):
            client.complete([{"role": "user", "content": "q"}])

        assert len(fake.messages.calls) == 1

    def test_does_not_retry_auth_errors(self):
        client, fake = client_with([make_status_error(401)])
        with pytest.raises(AssistantError):
            client.complete([{"role": "user", "content": "q"}])

        assert len(fake.messages.calls) == 1

    def test_backoff_grows_exponentially(self, monkeypatch):
        delays = []
        monkeypatch.setattr("assistant.client.time.sleep", lambda d: delays.append(d))

        client, _ = client_with(
            [make_rate_limit_error() for _ in range(4)], max_retries=3
        )
        with pytest.raises(AssistantError):
            client.complete([{"role": "user", "content": "q"}])

        assert len(delays) == 3
        # Jitter is bounded, so each delay still exceeds the previous.
        assert delays[1] > delays[0]
        assert delays[2] > delays[1]


# --- error wrapping ----------------------------------------------------------


class TestErrorWrapping:
    def test_sdk_errors_do_not_leak(self):
        client, _ = client_with([make_status_error(400)])
        with pytest.raises(AssistantError):
            client.complete([{"role": "user", "content": "q"}])

    def test_raises_assistant_error_not_sdk_error(self):
        client, _ = client_with([make_status_error(400)])
        try:
            client.complete([{"role": "user", "content": "q"}])
        except AssistantError as exc:
            assert not isinstance(exc, anthropic.AnthropicError)

    def test_original_error_is_chained(self):
        client, _ = client_with([make_status_error(400)])
        try:
            client.complete([{"role": "user", "content": "q"}])
        except AssistantError as exc:
            assert exc.__cause__ is not None

    def test_refusal_raises_assistant_error(self):
        client, _ = client_with([make_message("", stop_reason="refusal")])
        with pytest.raises(AssistantError, match="declined"):
            client.complete([{"role": "user", "content": "q"}])


class TestExtractText:
    def test_strips_surrounding_whitespace(self):
        assert _extract_text(make_message("  padded  ")) == "padded"

    def test_empty_content_returns_empty_string(self):
        message = SimpleNamespace(content=[], stop_reason="end_turn")
        assert _extract_text(message) == ""
