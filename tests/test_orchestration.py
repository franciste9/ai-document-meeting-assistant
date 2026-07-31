"""Tests for the LangGraph summarization pipeline.

Uses the same FakeClient pattern as test_main.py — no network calls.
"""

import json

import pytest

from assistant.models import Document, SpeakerTurn
from assistant.orchestration import (
    CHUNK,
    MERGE,
    SUMMARIZE_CHUNKS,
    SUMMARIZE_WHOLE,
    _route,
    _strip_code_fence,
    build_graph,
    draw_mermaid,
    summarize_via_graph,
)


class FakeClient:
    """Records prompts and replays canned responses."""

    def __init__(self, responses=None):
        self.responses = list(responses or ['{"summary": "ok"}'])
        self.calls = []

    def complete(self, messages, system=None, **kwargs):
        self.calls.append({"messages": messages, "system": system, **kwargs})
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


def make_document(text="short document", tokens=10, turns=None):
    return Document(
        source_path="./a.txt",
        doc_type="transcript" if turns else "text",
        title="A",
        raw_text=text,
        speaker_turns=turns,
        token_estimate=tokens,
    )


def make_state(tokens, threshold):
    return {
        "document": make_document(tokens=tokens),
        "threshold": threshold,
        "client": None,
        "chunks": None,
        "partials": [],
        "result": None,
    }


# -- graph structure ---------------------------------------------------------


class TestGraphStructure:
    def test_compiles(self):
        assert build_graph() is not None

    def test_has_the_four_named_nodes(self):
        nodes = build_graph().get_graph().nodes

        for name in (SUMMARIZE_WHOLE, CHUNK, SUMMARIZE_CHUNKS, MERGE):
            assert name in nodes

    def test_exports_a_mermaid_diagram(self):
        diagram = draw_mermaid()

        assert "graph" in diagram.lower()
        for name in (SUMMARIZE_WHOLE, CHUNK, SUMMARIZE_CHUNKS, MERGE):
            assert name in diagram

    def test_diagram_shows_the_chunk_path(self):
        diagram = draw_mermaid()

        assert f"{CHUNK} --> {SUMMARIZE_CHUNKS}" in diagram
        assert f"{SUMMARIZE_CHUNKS} --> {MERGE}" in diagram


# -- code fence stripping ----------------------------------------------------


class TestStripCodeFence:
    """The model wraps JSON in ```json despite the prompt forbidding it.

    Observed on 3 of 4 identical merge runs. Unstripped, it turns a correct
    answer into a 502 from the HTTP layer.
    """

    def test_strips_a_json_fence(self):
        raw = '```json\n{"summary": "ok"}\n```'
        assert _strip_code_fence(raw) == '{"summary": "ok"}'

    def test_strips_a_bare_fence(self):
        raw = '```\n{"summary": "ok"}\n```'
        assert _strip_code_fence(raw) == '{"summary": "ok"}'

    def test_strips_other_language_tags(self):
        raw = '```JSON\n{"summary": "ok"}\n```'
        assert _strip_code_fence(raw) == '{"summary": "ok"}'

    def test_result_is_parseable_after_stripping(self):
        raw = '```json\n{"summary": "ok", "decisions": [], "action_items": []}\n```'

        assert json.loads(_strip_code_fence(raw))["summary"] == "ok"

    def test_tolerates_surrounding_whitespace(self):
        raw = '\n\n```json\n{"summary": "ok"}\n```\n\n'
        assert _strip_code_fence(raw) == '{"summary": "ok"}'

    def test_handles_missing_trailing_newline(self):
        raw = '```json\n{"summary": "ok"}```'
        assert _strip_code_fence(raw) == '{"summary": "ok"}'

    def test_preserves_multiline_json(self):
        raw = '```json\n{\n  "summary": "ok",\n  "decisions": []\n}\n```'

        assert json.loads(_strip_code_fence(raw))["decisions"] == []

    def test_leaves_unfenced_json_untouched(self):
        raw = '{"summary": "ok"}'
        assert _strip_code_fence(raw) == raw

    def test_leaves_plain_text_untouched(self):
        raw = "not json at all"
        assert _strip_code_fence(raw) == raw

    def test_does_not_strip_a_fence_inside_a_larger_response(self):
        """Only a fence wrapping the whole response should be unwrapped."""
        raw = 'Here you go:\n```json\n{"summary": "ok"}\n```'
        assert _strip_code_fence(raw) == raw

    def test_preserves_backticks_inside_json_values(self):
        raw = '{"summary": "use the ``` fence syntax"}'
        assert _strip_code_fence(raw) == raw

    def test_preserves_json_containing_a_fenced_block(self):
        """A fence *inside* a value must survive intact."""
        raw = '{"summary": "run ```npm test``` first"}'

        assert json.loads(_strip_code_fence(raw))["summary"].count("```") == 2

    def test_empty_string_is_safe(self):
        assert _strip_code_fence("") == ""


class TestFenceStrippingInTheGraph:
    def test_whole_path_output_is_stripped(self):
        client = FakeClient(['```json\n{"summary": "whole"}\n```'])

        result = summarize_via_graph(
            make_document(tokens=10), client=client, threshold=1000
        )

        assert json.loads(result)["summary"] == "whole"

    def test_merge_output_is_stripped(self):
        text = "\n\n".join(f"para{i} " + "x" * 200 for i in range(6))
        document = make_document(text=text, tokens=len(text) // 4)
        client = FakeClient(
            ['{"summary": "part"}', '```json\n{"summary": "merged"}\n```']
        )

        result = summarize_via_graph(document, client=client, threshold=100)

        assert json.loads(result)["summary"] == "merged"

    def test_fenced_partials_are_stripped_before_merging(self):
        text = "\n\n".join(f"para{i} " + "x" * 200 for i in range(6))
        document = make_document(text=text, tokens=len(text) // 4)
        client = FakeClient(
            ['```json\n{"summary": "part"}\n```', '{"summary": "merged"}']
        )

        summarize_via_graph(document, client=client, threshold=100)

        merge_content = client.calls[-1]["messages"][0]["content"]
        assert "```" not in merge_content

    def test_unfenced_output_is_unchanged(self):
        client = FakeClient(['{"summary": "clean"}'])

        result = summarize_via_graph(
            make_document(tokens=10), client=client, threshold=1000
        )

        assert result == '{"summary": "clean"}'


# -- routing -----------------------------------------------------------------


class TestRouting:
    def test_well_below_threshold_goes_whole(self):
        assert _route(make_state(tokens=10, threshold=100)) == SUMMARIZE_WHOLE

    def test_just_below_threshold_goes_whole(self):
        assert _route(make_state(tokens=99, threshold=100)) == SUMMARIZE_WHOLE

    def test_exactly_at_threshold_goes_whole(self):
        """The original used `<=`, so the boundary itself is not chunked."""
        assert _route(make_state(tokens=100, threshold=100)) == SUMMARIZE_WHOLE

    def test_just_above_threshold_chunks(self):
        assert _route(make_state(tokens=101, threshold=100)) == CHUNK

    def test_well_above_threshold_chunks(self):
        assert _route(make_state(tokens=10_000, threshold=100)) == CHUNK

    def test_zero_token_document_goes_whole(self):
        assert _route(make_state(tokens=0, threshold=100)) == SUMMARIZE_WHOLE


# -- whole-document path -----------------------------------------------------


class TestWholeDocumentPath:
    def test_returns_the_model_response(self):
        client = FakeClient(['{"summary": "done"}'])

        result = summarize_via_graph(
            make_document(tokens=10), client=client, threshold=1000
        )

        assert result == '{"summary": "done"}'

    def test_makes_exactly_one_call(self):
        client = FakeClient()

        summarize_via_graph(make_document(tokens=10), client=client, threshold=1000)

        assert len(client.calls) == 1

    def test_uses_the_summary_system_prompt(self):
        client = FakeClient()

        summarize_via_graph(make_document(tokens=10), client=client, threshold=1000)

        assert "meeting and document analyst" in client.calls[0]["system"]

    def test_sends_the_document_body(self):
        client = FakeClient()

        summarize_via_graph(
            make_document(text="the body text", tokens=10),
            client=client,
            threshold=1000,
        )

        assert "the body text" in client.calls[0]["messages"][0]["content"]

    def test_does_not_use_the_merge_prompt(self):
        client = FakeClient()

        summarize_via_graph(make_document(tokens=10), client=client, threshold=1000)

        assert "merging partial analyses" not in client.calls[0]["system"]


# -- chunk-and-merge path ----------------------------------------------------


class TestChunkAndMergePath:
    def _big_document(self, paragraphs=6):
        text = "\n\n".join(f"para{i} " + "x" * 200 for i in range(paragraphs))
        return make_document(text=text, tokens=len(text) // 4)

    def test_returns_the_merged_result(self):
        client = FakeClient(['{"summary": "part"}', '{"summary": "merged"}'])

        result = summarize_via_graph(
            self._big_document(), client=client, threshold=100
        )

        assert result == '{"summary": "merged"}'

    def test_makes_one_call_per_chunk_plus_a_merge(self):
        client = FakeClient(['{"summary": "part"}', '{"summary": "merged"}'])

        summarize_via_graph(self._big_document(), client=client, threshold=100)

        # At least one call per chunk, plus the merge.
        assert len(client.calls) > 2

    def test_merge_is_the_final_call(self):
        client = FakeClient(['{"summary": "part"}', '{"summary": "merged"}'])

        summarize_via_graph(self._big_document(), client=client, threshold=100)

        assert "merging partial analyses" in client.calls[-1]["system"]

    def test_chunk_calls_use_the_summary_prompt(self):
        client = FakeClient(['{"summary": "part"}', '{"summary": "merged"}'])

        summarize_via_graph(self._big_document(), client=client, threshold=100)

        assert "meeting and document analyst" in client.calls[0]["system"]

    def test_chunk_calls_are_numbered(self):
        client = FakeClient(['{"summary": "part"}', '{"summary": "merged"}'])

        summarize_via_graph(self._big_document(), client=client, threshold=100)

        first = client.calls[0]["messages"][0]["content"]
        assert "part 1 of" in first

    def test_merge_receives_the_partials(self):
        client = FakeClient(['{"summary": "part"}', '{"summary": "merged"}'])

        summarize_via_graph(self._big_document(), client=client, threshold=100)

        merge_content = client.calls[-1]["messages"][0]["content"]
        assert '{"summary": "part"}' in merge_content

    def test_splits_on_speaker_turns_when_present(self):
        turns = [SpeakerTurn(speaker=f"S{i}", text="x" * 400) for i in range(8)]
        text = "\n".join(f"S{i}: " + "x" * 400 for i in range(8))
        document = make_document(text=text, tokens=len(text) // 4, turns=turns)

        client = FakeClient(['{"summary": "part"}', '{"summary": "merged"}'])
        summarize_via_graph(document, client=client, threshold=150)

        # Every chunk call should carry whole speaker turns.
        chunk_calls = client.calls[:-1]
        assert len(chunk_calls) > 1


# -- behavioral equivalence with the pre-refactor implementation -------------


class TestEquivalenceWithMain:
    """summarize_document should be a pass-through to the graph."""

    def test_main_delegates_to_the_graph(self):
        from assistant.main import summarize_document

        client = FakeClient(['{"summary": "via main"}'])

        result = summarize_document(
            make_document(tokens=10), client=client, threshold=1000
        )

        assert result == '{"summary": "via main"}'

    def test_main_and_graph_agree_on_the_whole_path(self):
        from assistant.main import summarize_document

        document = make_document(tokens=10)

        via_main = summarize_document(
            document, client=FakeClient(['{"a": 1}']), threshold=1000
        )
        via_graph = summarize_via_graph(
            document, client=FakeClient(['{"a": 1}']), threshold=1000
        )

        assert via_main == via_graph

    def test_main_and_graph_agree_on_the_chunk_path(self):
        from assistant.main import summarize_document

        text = "\n\n".join(f"para{i} " + "x" * 200 for i in range(6))
        document = make_document(text=text, tokens=len(text) // 4)
        responses = ['{"summary": "part"}', '{"summary": "merged"}']

        via_main = summarize_document(
            document, client=FakeClient(list(responses)), threshold=100
        )
        via_graph = summarize_via_graph(
            document, client=FakeClient(list(responses)), threshold=100
        )

        assert via_main == via_graph

    def test_main_and_graph_issue_the_same_calls(self):
        from assistant.main import summarize_document

        text = "\n\n".join(f"para{i} " + "x" * 200 for i in range(6))
        document = make_document(text=text, tokens=len(text) // 4)
        responses = ['{"summary": "part"}', '{"summary": "merged"}']

        main_client = FakeClient(list(responses))
        graph_client = FakeClient(list(responses))
        summarize_document(document, client=main_client, threshold=100)
        summarize_via_graph(document, client=graph_client, threshold=100)

        assert len(main_client.calls) == len(graph_client.calls)
        assert [c["system"] for c in main_client.calls] == [
            c["system"] for c in graph_client.calls
        ]


# -- defaults ----------------------------------------------------------------


class TestDefaults:
    def test_threshold_defaults_to_config(self, monkeypatch):
        monkeypatch.setenv("CHUNK_TOKEN_THRESHOLD", "50")
        client = FakeClient(['{"summary": "part"}', '{"summary": "merged"}'])
        text = "\n\n".join(f"para{i} " + "x" * 200 for i in range(6))

        summarize_via_graph(make_document(text=text, tokens=1000), client=client)

        # Threshold of 50 forces the chunk path.
        assert "merging partial analyses" in client.calls[-1]["system"]

    def test_missing_api_key_raises_when_no_client_given(self, monkeypatch):
        """The real client is constructed lazily, so ingest works without a key."""
        from assistant.errors import AssistantError

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(
            "assistant.client.ClaudeClient.complete",
            lambda *a, **k: (_ for _ in ()).throw(AssistantError("no key")),
        )

        with pytest.raises(AssistantError):
            summarize_via_graph(make_document(tokens=10), threshold=1000)
