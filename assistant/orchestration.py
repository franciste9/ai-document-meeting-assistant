"""Summarization pipeline as an explicit LangGraph StateGraph.

The control flow here is the same one that used to live inline in
`main.summarize_document`: short documents go out in a single call; long ones
are chunked, summarized per chunk, and merged. Making it a graph means the
pipeline is an artifact you can draw and inspect, not just branching you have
to read.

    START ─┬─(token_estimate <= threshold)─→ summarize_whole ──→ END
           └─(over threshold)──────────────→ chunk ──→ summarize_chunks ──→ merge ──→ END

Nothing about the prompts or the chunking rule changes — those are imported
from where they already live.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from assistant import config
from assistant.ingestion.normalize import chunk_document
from assistant.models import Document
from assistant.prompts.meeting_summary import (
    MERGE_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    build_chunk_messages,
    build_merge_messages,
    build_summary_messages,
)

# Node names. Referenced by the graph wiring and by the routing function, so
# they live in one place rather than as repeated string literals.
SUMMARIZE_WHOLE = "summarize_whole"
CHUNK = "chunk"
SUMMARIZE_CHUNKS = "summarize_chunks"
MERGE = "merge"


class SummarizeState(TypedDict):
    """State threaded through the graph.

    `client` is carried here rather than constructed inside a node so tests can
    inject a fake and the suite stays network-free.

    Note: a live client holds an HTTP session and is not serializable. That is
    fine because this graph runs one-shot with no checkpointer — but adding
    checkpointing later would mean moving the client out of state (e.g. into
    LangGraph's `configurable` channel).
    """

    document: Document
    threshold: int
    client: Any
    chunks: list[str] | None
    partials: list[str]
    result: str | None


# A response that is entirely one fenced block: ```json ... ``` or ``` ... ```.
# Anchored and DOTALL so it only matches when the fence wraps the whole payload —
# a response that merely *contains* backticks is left alone.
_FENCED_BLOCK = re.compile(
    r"\A```[^\n`]*\n(?P<body>.*?)\n?```\Z",
    re.DOTALL,
)


def _strip_code_fence(text: str) -> str:
    """Unwrap a Markdown code fence around the model's JSON.

    Both system prompts say "no code fences", but the model wraps the merge
    response in ```json often enough to matter — it was observed on 3 of 4
    identical runs. Callers parse this string with json.loads, so a fence turns
    a correct answer into a 502 from the HTTP layer and unformatted output from
    the CLI.

    Only strips when the fence encloses the entire response; anything else is
    returned untouched so we never corrupt a genuine answer.
    """
    match = _FENCED_BLOCK.match(text.strip())
    return match.group("body").strip() if match else text


def _route(state: SummarizeState) -> str:
    """Entry edge: does this document fit in a single call?

    Mirrors the original `<=` comparison exactly — a document sitting right on
    the threshold goes through the whole-document path.
    """
    if state["document"].token_estimate <= state["threshold"]:
        return SUMMARIZE_WHOLE
    return CHUNK


def _summarize_whole(state: SummarizeState) -> dict[str, Any]:
    """Summarize a document small enough to send in one piece."""
    result = state["client"].complete(
        messages=build_summary_messages(state["document"]),
        system=SUMMARY_SYSTEM_PROMPT,
    )
    return {"result": _strip_code_fence(result)}


def _chunk(state: SummarizeState) -> dict[str, Any]:
    """Split an oversized document on speaker-turn or paragraph boundaries."""
    chunks = chunk_document(state["document"], max_tokens=state["threshold"])
    return {"chunks": chunks}


def _summarize_chunks(state: SummarizeState) -> dict[str, Any]:
    """Summarize each chunk in turn.

    Sequential by design for this pass; parallel fan-out is a possible future
    optimization, not a requirement.
    """
    chunks = state["chunks"] or []
    partials = [
        # Normalized here too: these are embedded in the merge prompt, and
        # feeding the merge step fenced input invites a fenced result.
        _strip_code_fence(
            state["client"].complete(
                messages=build_chunk_messages(chunk, index + 1, len(chunks)),
                system=SUMMARY_SYSTEM_PROMPT,
            )
        )
        for index, chunk in enumerate(chunks)
    ]
    return {"partials": partials}


def _merge(state: SummarizeState) -> dict[str, Any]:
    """Combine the per-chunk summaries into one result."""
    result = state["client"].complete(
        messages=build_merge_messages(state["partials"]),
        system=MERGE_SYSTEM_PROMPT,
    )
    return {"result": _strip_code_fence(result)}


def build_graph() -> StateGraph:
    """Wire the nodes and edges, and compile.

    Exposed separately from `summarize_via_graph` so the compiled graph can be
    inspected or exported as a diagram without running it.
    """
    builder = StateGraph(SummarizeState)

    builder.add_node(SUMMARIZE_WHOLE, _summarize_whole)
    builder.add_node(CHUNK, _chunk)
    builder.add_node(SUMMARIZE_CHUNKS, _summarize_chunks)
    builder.add_node(MERGE, _merge)

    # Entry: one call, or chunk-and-merge.
    builder.add_conditional_edges(
        START,
        _route,
        {SUMMARIZE_WHOLE: SUMMARIZE_WHOLE, CHUNK: CHUNK},
    )

    builder.add_edge(SUMMARIZE_WHOLE, END)
    builder.add_edge(CHUNK, SUMMARIZE_CHUNKS)
    builder.add_edge(SUMMARIZE_CHUNKS, MERGE)
    builder.add_edge(MERGE, END)

    return builder.compile()


# Compiled once at import: the topology is static, so there's no reason to
# rebuild it per call.
_GRAPH = build_graph()


def summarize_via_graph(
    document: Document,
    client: Any | None = None,
    threshold: int | None = None,
) -> str:
    """Summarize a document by invoking the graph.

    Same contract as the original `summarize_document`: returns a JSON string
    with summary/decisions/action_items, taking the whole-document path or the
    chunk-and-merge path depending on the threshold.
    """
    # Imported lazily so `ingest` still works without an API key configured.
    if client is None:
        from assistant.client import ClaudeClient

        client = ClaudeClient()

    if threshold is None:
        threshold = config.get_chunk_token_threshold()

    final_state = _GRAPH.invoke(
        {
            "document": document,
            "threshold": threshold,
            "client": client,
            "chunks": None,
            "partials": [],
            "result": None,
        }
    )

    return final_state["result"]


def summarize_via_graph_stream(
    document: Document,
    client: Any | None = None,
    threshold: int | None = None,
) -> Iterator[str]:
    """As `summarize_via_graph`, but yields incrementally.

    The whole-document path streams directly. The chunk-and-merge path yields a
    short progress line per chunk — those calls stay non-streamed — and streams
    only the final merge step.

    Deliberately a parallel path rather than a streaming mode on the compiled
    graph: threading incremental yields through LangGraph's node/state model
    would add real complexity for no benefit, since the goal here is progress
    visibility, not different branching.

    Output is raw text. Unlike the non-streaming path there is no
    `_strip_code_fence` pass — you cannot strip a fence you have not finished
    receiving. A caller that needs parseable JSON should accumulate the stream
    and strip client-side, or call the non-streaming path instead.
    """
    # Imported lazily so `ingest` still works without an API key configured.
    if client is None:
        from assistant.client import ClaudeClient

        client = ClaudeClient()

    if threshold is None:
        threshold = config.get_chunk_token_threshold()

    # Same routing rule as `_route`, kept in sync deliberately.
    if document.token_estimate <= threshold:
        yield from client.complete_stream(
            messages=build_summary_messages(document),
            system=SUMMARY_SYSTEM_PROMPT,
        )
        return

    chunks = chunk_document(document, max_tokens=threshold)
    partials: list[str] = []

    for index, chunk in enumerate(chunks):
        yield f"[summarizing chunk {index + 1} of {len(chunks)}]\n"
        partials.append(
            _strip_code_fence(
                client.complete(
                    messages=build_chunk_messages(chunk, index + 1, len(chunks)),
                    system=SUMMARY_SYSTEM_PROMPT,
                )
            )
        )

    yield from client.complete_stream(
        messages=build_merge_messages(partials),
        system=MERGE_SYSTEM_PROMPT,
    )


def draw_mermaid() -> str:
    """The compiled graph as a Mermaid diagram, for docs and the ADR."""
    return _GRAPH.get_graph().draw_mermaid()
