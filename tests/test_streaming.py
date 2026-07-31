"""Tests for the streaming path: client, orchestration, and CLI.

Network-free — extends the FakeClient convention with `complete_stream`.
"""

import anthropic
import httpx
import pytest

from assistant.client import ClaudeClient
from assistant.errors import AssistantError
from assistant.main import summarize_document_stream
from assistant.models import Document, SpeakerTurn
from assistant.orchestration import summarize_via_graph_stream


class FakeClient:
    """Records prompts and replays canned responses.

    Same shape as the FakeClient used elsewhere in the suite, plus
    `complete_stream` so the streaming path can be exercised without a network.
    """

    def __init__(self, responses=None, stream_deltas=None):
        self.responses = list(responses or ['{"summary": "ok"}'])
        self.stream_deltas = list(stream_deltas or ["str", "eam", "ed"])
        self.calls = []
        self.stream_calls = []

    def complete(self, messages, system=None, **kwargs):
        self.calls.append({"messages": messages, "system": system, **kwargs})
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]

    def complete_stream(self, messages, system=None, **kwargs):
        self.stream_calls.append({"messages": messages, "system": system, **kwargs})
        yield from self.stream_deltas


def make_document(text="short document", tokens=10, turns=None):
    return Document(
        source_path="./a.txt",
        doc_type="transcript" if turns else "text",
        title="A",
        raw_text=text,
        speaker_turns=turns,
        token_estimate=tokens,
    )


def big_document(paragraphs=6):
    text = "\n\n".join(f"para{i} " + "x" * 200 for i in range(paragraphs))
    return make_document(text=text, tokens=len(text) // 4)


# -- ClaudeClient.complete_stream --------------------------------------------


class FakeStream:
    def __init__(self, deltas, error=None):
        self.deltas = deltas
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    @property
    def text_stream(self):
        for delta in self.deltas:
            yield delta
        if self.error is not None:
            raise self.error


class FakeMessages:
    def __init__(self, deltas=None, error=None, raise_on_open=None):
        self.deltas = deltas or []
        self.error = error
        self.raise_on_open = raise_on_open
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_on_open is not None:
            raise self.raise_on_open
        return FakeStream(self.deltas, self.error)


class FakeAnthropic:
    def __init__(self, **kwargs):
        self.messages = FakeMessages(**kwargs)


def make_status_error(status_code=500):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    return anthropic.APIStatusError("boom", response=response, body=None)


def streaming_client(**kwargs):
    fake = FakeAnthropic(**kwargs)
    return ClaudeClient(model="claude-sonnet-5", client=fake), fake


class TestCompleteStream:
    def test_yields_each_delta(self):
        client, _ = streaming_client(deltas=["a", "b", "c"])

        assert list(client.complete_stream([{"role": "user", "content": "q"}])) == [
            "a",
            "b",
            "c",
        ]

    def test_assembles_to_the_full_text(self):
        client, _ = streaming_client(deltas=["Hello", ", ", "world"])

        result = "".join(client.complete_stream([{"role": "user", "content": "q"}]))

        assert result == "Hello, world"

    def test_is_lazy(self):
        """Nothing should be requested until the generator is consumed."""
        client, fake = streaming_client(deltas=["a"])

        client.complete_stream([{"role": "user", "content": "q"}])

        assert fake.messages.calls == []

    def test_sends_the_model(self):
        client, fake = streaming_client(deltas=["a"])

        list(client.complete_stream([{"role": "user", "content": "q"}]))

        assert fake.messages.calls[0]["model"] == "claude-sonnet-5"

    def test_sends_the_messages(self):
        client, fake = streaming_client(deltas=["a"])
        messages = [{"role": "user", "content": "q"}]

        list(client.complete_stream(messages))

        assert fake.messages.calls[0]["messages"] == messages

    def test_applies_cache_control_to_the_system_prompt(self):
        client, fake = streaming_client(deltas=["a"])

        list(client.complete_stream([{"role": "user", "content": "q"}], system="s"))

        assert fake.messages.calls[0]["system"][-1]["cache_control"] == {
            "type": "ephemeral"
        }

    def test_caching_can_be_disabled(self):
        client, fake = streaming_client(deltas=["a"])

        list(
            client.complete_stream(
                [{"role": "user", "content": "q"}], system="s", cache_system=False
            )
        )

        assert "cache_control" not in fake.messages.calls[0]["system"][0]

    def test_omits_system_when_not_given(self):
        client, fake = streaming_client(deltas=["a"])

        list(client.complete_stream([{"role": "user", "content": "q"}]))

        assert "system" not in fake.messages.calls[0]

    def test_max_tokens_is_sent(self):
        client, fake = streaming_client(deltas=["a"])

        list(client.complete_stream([{"role": "user", "content": "q"}], max_tokens=512))

        assert fake.messages.calls[0]["max_tokens"] == 512


class TestCompleteStreamDoesNotRetry:
    """A stream that has begun emitting must not be silently restarted."""

    def test_failure_on_open_raises_assistant_error(self):
        client, _ = streaming_client(raise_on_open=make_status_error(500))

        with pytest.raises(AssistantError, match="Anthropic API error"):
            list(client.complete_stream([{"role": "user", "content": "q"}]))

    def test_failure_on_open_makes_a_single_attempt(self):
        client, fake = streaming_client(raise_on_open=make_status_error(500))

        with pytest.raises(AssistantError):
            list(client.complete_stream([{"role": "user", "content": "q"}]))

        assert len(fake.messages.calls) == 1

    def test_rate_limit_is_not_retried(self):
        """complete() retries a 429; complete_stream deliberately does not."""
        client, fake = streaming_client(raise_on_open=make_status_error(429))

        with pytest.raises(AssistantError):
            list(client.complete_stream([{"role": "user", "content": "q"}]))

        assert len(fake.messages.calls) == 1

    def test_failure_mid_stream_raises_after_partial_output(self):
        client, _ = streaming_client(deltas=["a", "b"], error=make_status_error(500))

        received = []
        with pytest.raises(AssistantError):
            for delta in client.complete_stream([{"role": "user", "content": "q"}]):
                received.append(delta)

        assert received == ["a", "b"]

    def test_failure_mid_stream_makes_a_single_attempt(self):
        client, fake = streaming_client(
            deltas=["a"], error=make_status_error(500)
        )

        with pytest.raises(AssistantError):
            list(client.complete_stream([{"role": "user", "content": "q"}]))

        assert len(fake.messages.calls) == 1

    def test_sdk_errors_do_not_leak(self):
        client, _ = streaming_client(raise_on_open=make_status_error(400))

        try:
            list(client.complete_stream([{"role": "user", "content": "q"}]))
        except AssistantError as exc:
            assert not isinstance(exc, anthropic.AnthropicError)


# -- summarize_via_graph_stream ----------------------------------------------


class TestStreamWholeDocumentPath:
    def test_yields_the_deltas(self):
        client = FakeClient(stream_deltas=["one ", "two ", "three"])

        result = list(
            summarize_via_graph_stream(
                make_document(tokens=10), client=client, threshold=1000
            )
        )

        assert result == ["one ", "two ", "three"]

    def test_uses_the_streaming_call(self):
        client = FakeClient()

        list(
            summarize_via_graph_stream(
                make_document(tokens=10), client=client, threshold=1000
            )
        )

        assert len(client.stream_calls) == 1
        assert client.calls == []

    def test_uses_the_summary_system_prompt(self):
        client = FakeClient()

        list(
            summarize_via_graph_stream(
                make_document(tokens=10), client=client, threshold=1000
            )
        )

        assert "meeting and document analyst" in client.stream_calls[0]["system"]

    def test_sends_the_document_body(self):
        client = FakeClient()

        list(
            summarize_via_graph_stream(
                make_document(text="the body text", tokens=10),
                client=client,
                threshold=1000,
            )
        )

        content = client.stream_calls[0]["messages"][0]["content"]
        assert "the body text" in content

    def test_emits_no_progress_lines(self):
        client = FakeClient(stream_deltas=["a"])

        result = "".join(
            summarize_via_graph_stream(
                make_document(tokens=10), client=client, threshold=1000
            )
        )

        assert "summarizing chunk" not in result

    def test_at_threshold_uses_the_whole_path(self):
        """Matches _route's `<=`: the boundary itself is not chunked."""
        client = FakeClient()

        list(
            summarize_via_graph_stream(
                make_document(tokens=100), client=client, threshold=100
            )
        )

        assert len(client.stream_calls) == 1
        assert client.calls == []


class TestStreamChunkAndMergePath:
    def test_emits_a_progress_line_per_chunk(self):
        client = FakeClient()

        result = "".join(
            summarize_via_graph_stream(big_document(), client=client, threshold=100)
        )

        assert "[summarizing chunk 1 of" in result

    def test_progress_lines_are_numbered(self):
        client = FakeClient()

        result = "".join(
            summarize_via_graph_stream(big_document(), client=client, threshold=100)
        )

        chunk_count = len(client.calls)
        for index in range(1, chunk_count + 1):
            assert f"[summarizing chunk {index} of {chunk_count}]" in result

    def test_chunk_calls_are_not_streamed(self):
        """Per-chunk calls stay synchronous — the resolved PRD trade-off."""
        client = FakeClient()

        list(summarize_via_graph_stream(big_document(), client=client, threshold=100))

        assert len(client.calls) > 1

    def test_only_the_merge_step_streams(self):
        client = FakeClient()

        list(summarize_via_graph_stream(big_document(), client=client, threshold=100))

        assert len(client.stream_calls) == 1

    def test_merge_uses_the_merge_prompt(self):
        client = FakeClient()

        list(summarize_via_graph_stream(big_document(), client=client, threshold=100))

        assert "merging partial analyses" in client.stream_calls[0]["system"]

    def test_chunk_calls_use_the_summary_prompt(self):
        client = FakeClient()

        list(summarize_via_graph_stream(big_document(), client=client, threshold=100))

        assert "meeting and document analyst" in client.calls[0]["system"]

    def test_merge_receives_the_partials(self):
        client = FakeClient(responses=['{"summary": "part"}'])

        list(summarize_via_graph_stream(big_document(), client=client, threshold=100))

        merge_content = client.stream_calls[0]["messages"][0]["content"]
        assert '{"summary": "part"}' in merge_content

    def test_fenced_partials_are_stripped_before_merging(self):
        client = FakeClient(responses=['```json\n{"summary": "part"}\n```'])

        list(summarize_via_graph_stream(big_document(), client=client, threshold=100))

        merge_content = client.stream_calls[0]["messages"][0]["content"]
        assert "```" not in merge_content

    def test_merge_deltas_are_yielded_last(self):
        client = FakeClient(stream_deltas=["MERGED"])

        result = "".join(
            summarize_via_graph_stream(big_document(), client=client, threshold=100)
        )

        assert result.endswith("MERGED")

    def test_streamed_output_is_not_fence_stripped(self):
        """Raw by design — you can't strip a fence you haven't finished getting."""
        client = FakeClient(stream_deltas=["```json\n", '{"a": 1}', "\n```"])

        result = "".join(
            summarize_via_graph_stream(
                make_document(tokens=10), client=client, threshold=1000
            )
        )

        assert result.startswith("```")


class TestStreamDefaults:
    def test_threshold_defaults_to_config(self, monkeypatch):
        monkeypatch.setenv("CHUNK_TOKEN_THRESHOLD", "50")
        client = FakeClient()

        list(summarize_via_graph_stream(big_document(), client=client))

        # A threshold of 50 forces the chunk path.
        assert "merging partial analyses" in client.stream_calls[0]["system"]

    def test_is_lazy(self):
        """No API calls until the generator is consumed."""
        client = FakeClient()

        summarize_via_graph_stream(
            make_document(tokens=10), client=client, threshold=1000
        )

        assert client.stream_calls == []
        assert client.calls == []


# -- CLI wrapper -------------------------------------------------------------


class TestMainStreamWrapper:
    def test_delegates_to_orchestration(self):
        client = FakeClient(stream_deltas=["a", "b"])

        result = list(
            summarize_document_stream(
                make_document(tokens=10), client=client, threshold=1000
            )
        )

        assert result == ["a", "b"]

    def test_matches_the_orchestration_output(self):
        document = make_document(tokens=10)

        via_main = "".join(
            summarize_document_stream(
                document, client=FakeClient(stream_deltas=["x", "y"]), threshold=1000
            )
        )
        via_orchestration = "".join(
            summarize_via_graph_stream(
                document, client=FakeClient(stream_deltas=["x", "y"]), threshold=1000
            )
        )

        assert via_main == via_orchestration


class TestNonStreamingPathUnchanged:
    """The streaming work must not disturb the existing path."""

    def test_summarize_via_graph_still_returns_a_string(self):
        from assistant.orchestration import summarize_via_graph

        client = FakeClient(responses=['{"summary": "unchanged"}'])

        result = summarize_via_graph(
            make_document(tokens=10), client=client, threshold=1000
        )

        assert result == '{"summary": "unchanged"}'

    def test_non_streaming_path_makes_no_stream_calls(self):
        from assistant.orchestration import summarize_via_graph

        client = FakeClient()

        summarize_via_graph(make_document(tokens=10), client=client, threshold=1000)

        assert client.stream_calls == []
