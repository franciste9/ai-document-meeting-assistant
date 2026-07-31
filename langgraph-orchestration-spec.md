# Dev Spec: LangGraph Orchestration for Multi-Step Summarization

**Scope for this pass:** replace the hand-rolled branching inside `summarize_document()` with an explicit LangGraph `StateGraph`, with identical external behavior. This is a refactor of *how* summarization is orchestrated, not a change to *what* it produces — same prompts, same chunking rule, same output contract.

## Why this exists
Today, `main.summarize_document()` decides in plain Python whether a document needs chunking, then either makes one API call or loops over chunks and merges the results. That works, but it's implicit control flow — nothing about it is inspectable as a pipeline, and this week's course track (RAG orchestration) and the Saturday ADR (API/orchestration choices) both call for an explicit answer to "how is this actually orchestrated." This pass makes the pipeline a real graph: a diagram you can draw, not just code you can read.

## Tech Stack (additions only)
- `langgraph`

Nothing else changes. `anthropic`, `pydantic`, the loaders, the store, and the prompts are all reused as-is.

## Project Structure (additions only)
```
assistant/
  orchestration.py     # New. The StateGraph: state, nodes, edges, and the
                        # public summarize_via_graph() entrypoint.
tests/
  test_orchestration.py  # New. Routing + both execution paths, FakeClient only.
```

`main.py`, `client.py`, `store.py`, `config.py`, and everything under `ingestion/` and `prompts/` are unchanged — this pass restructures control flow, it doesn't touch prompts, chunking rules, or persistence.

## State (orchestration.py)
```python
class SummarizeState(TypedDict):
    document: Document
    threshold: int
    client: Any            # injected — never instantiate ClaudeClient() inside a node
    chunks: list[str] | None
    partials: list[str]
    result: str | None
```

`client` is carried in state (not a module-level global) specifically so tests can inject the existing `FakeClient` from `tests/test_main.py`'s pattern — no real network calls in the test suite, same as everywhere else in this repo.

## Graph Design

**Entry (conditional edge from `START`):** route on `document.token_estimate <= threshold`.
- `True` → `summarize_whole`
- `False` → `chunk`

**Node: `summarize_whole`**
- Calls `state["client"].complete(build_summary_messages(state["document"]), system=SUMMARY_SYSTEM_PROMPT)`.
- Sets `result`. Routes to `END`.

**Node: `chunk`**
- Calls the existing `chunk_document(document, max_tokens=threshold)` from `ingestion/normalize.py` — reuse as-is, don't reimplement chunking here.
- Sets `chunks`. Routes to `summarize_chunks`.

**Node: `summarize_chunks`**
- For each chunk, calls `state["client"].complete(build_chunk_messages(chunk, i+1, len(chunks)), system=SUMMARY_SYSTEM_PROMPT)`.
- Sequential calls are fine for this pass — no fan-out/parallel nodes required.
- Sets `partials`. Routes to `merge`.

**Node: `merge`**
- Calls `state["client"].complete(build_merge_messages(partials), system=MERGE_SYSTEM_PROMPT)`.
- Sets `result`. Routes to `END`.

All four node functions should be small and named for what they do (`_route`, `_summarize_whole`, `_chunk`, `_summarize_chunks`, `_merge`) — this graph is meant to be a legible artifact you can screenshot into the ADR or walk through in an interview, not just working code.

## Public Entrypoint
```python
def summarize_via_graph(
    document: Document,
    client: Any | None = None,
    threshold: int | None = None,
) -> str:
    """Builds and invokes the graph. Same contract as today's summarize_document:
    returns a JSON string with summary/decisions/action_items, using the
    whole-document path or the chunk-and-merge path depending on threshold."""
```

Defaults mirror the existing function: lazily construct a real `ClaudeClient()` if `client` is `None` (so `ingest` still works without an API key configured, same as today), and pull `threshold` from `config.get_chunk_token_threshold()` if not passed.

## Integration Point
`main.summarize_document()` becomes a thin wrapper:
```python
def summarize_document(document, client=None, threshold=None) -> str:
    return orchestration.summarize_via_graph(document, client=client, threshold=threshold)
```
Same name, same signature, same return type. This is deliberate: the CLI and the FastAPI routes both call `summarize_document()` today, and neither should need to change. If they'd have to change, the refactor has leaked past its intended boundary.

## Acceptance Criteria
- [ ] `orchestration.py` defines a `StateGraph` with the four nodes above and a conditional entry edge on the threshold check
- [ ] `summarize_via_graph()` produces output behaviorally identical to today's `summarize_document()` for both the whole-document path and the chunk-and-merge path
- [ ] `main.summarize_document()` delegates to `summarize_via_graph()` with an unchanged public signature
- [ ] All 215 existing tests still pass unmodified — they exercise `summarize_document()`'s behavior, which now runs through the graph underneath
- [ ] `tests/test_orchestration.py` covers: the routing decision at and around the threshold boundary, the whole-document path, and the chunk-and-merge path, all using `FakeClient` — no real network calls
- [ ] The compiled graph can be exported as a diagram (`graph.get_graph().draw_mermaid()` or equivalent) — save the output for the ADR

## Explicitly Out of Scope for This Pass
- Parallel/fan-out chunk summarization — sequential per-chunk calls are fine; concurrency is a possible future optimization, not required here
- Streaming graph execution or streaming tokens back to callers
- Graph state persistence / checkpointing between runs — each call is a one-shot invocation, nothing needs to survive across calls
- Cross-document memory or multi-turn conversation
- Any change to prompts, chunking rules, `client.py`'s retry/caching logic, or persistence — this pass only restructures the control flow that was already in `main.py`

## Notes for Claude Code
- Reuse `chunk_document`, `build_summary_messages`, `build_chunk_messages`, `build_merge_messages`, `SUMMARY_SYSTEM_PROMPT`, and `MERGE_SYSTEM_PROMPT` exactly as they exist in `ingestion/normalize.py` and `prompts/meeting_summary.py`. Nothing about chunking or prompt content should change — only how the steps are wired together.
- Keep `client` threaded through graph state rather than instantiated inside a node function; hardcoding `ClaudeClient()` inside a node breaks the `FakeClient` test pattern used throughout this repo.
- Write `tests/test_orchestration.py` alongside `orchestration.py`, not after — same convention as the rest of this repo.
- Do not delete or restructure `main.py`'s CLI commands, `client.py`, `store.py`, or anything under `ingestion/` — this is additive/refactor-only within the summarization path.
- Once this passes, it's worth a short note in the ADR on why an explicit graph over hand-rolled branching — that's the actual leader-lens payoff of today's work.
