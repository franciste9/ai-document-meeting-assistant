# Dev Spec: Streaming Responses + Auth Hardening

**Scope for this pass:** add a real streaming path through `ClaudeClient` → orchestration → CLI/API, and gate the FastAPI routes that cost money behind a shared bearer token. Additive only — nothing about the existing non-streaming path, the compiled graph, or its 215+ tests changes.

**Resolves two open questions from the PRD:**
1. Streaming applies to the whole-document path directly, and to the final merge step on the chunk-and-merge path. Per-chunk intermediate calls stay non-streamed, surfaced instead as short progress lines ("summarizing chunk 2 of 5"). Token-level streaming of every chunk call was considered and rejected — same perceived benefit, meaningfully more complexity.
2. The new auth gate applies to the network-facing API only. The CLI keeps using the Anthropic key directly, same as today — it's local and already access-controlled by the machine.

## Why this exists
See `streaming-auth-hardening-prd.md` for the full problem statement. Short version: `ClaudeClient.complete()` already threads a `stream` parameter into the SDK's streaming endpoint but discards every chunk until `get_final_message()` returns — so nothing today actually streams. And the FastAPI wrapper's cost-incurring routes are wide open, which was correctly deferred while nothing was deployed but needs closing before Saturday's public URL goes live.

## Current State (grounded in the actual code)
- `assistant/client.py`: `_send()` already branches on `stream: bool`, but both branches return a fully-materialized `anthropic.types.Message` — the streaming branch just calls `.get_final_message()` inside the `with` block.
- `assistant/orchestration.py`: `_summarize_whole`, `_summarize_chunks`, and `_merge` all call `state["client"].complete(...)` and store the full string result in `SummarizeState`. The compiled `_GRAPH` and `summarize_via_graph()` are synchronous, one-shot, and return a plain `str`. **None of this changes in this pass** — a new, separate streaming path is added alongside it.
- `assistant/api.py`: `POST /documents/{doc_id}/summarize` calls `summarizer(document)` synchronously inside an `async def` route and returns a validated `SummaryOut`. No auth dependency exists on any route today.
- `assistant/config.py`: has no notion of an inbound API token — only `get_api_key()` for the outbound Anthropic key.

## Component 1: `ClaudeClient.complete_stream()` (client.py)

New method alongside `complete()` / `complete_raw()`:

```python
def complete_stream(
    self,
    messages: list[dict],
    system: str | list[dict] | None = None,
    max_tokens: int = config.MAX_TOKENS,
    cache_system: bool = True,
) -> Iterator[str]:
    """Yield text deltas as they arrive from the API.

    Unlike `complete()`, this does not retry. A transient failure before or
    during the stream surfaces immediately as an AssistantError — once
    partial output may have already reached a caller (over HTTP or to a
    terminal), silently retrying from scratch would mean re-sending output
    that was already shown. That's a deliberate trade-off for this pass, not
    an oversight.
    """
```

Implementation: build the same `request` dict `complete()` builds (reuse `_build_system` as-is), then:
```python
with self._client.messages.stream(**request) as stream:
    for text in stream.text_stream:
        yield text
```
Wrap SDK exceptions in `AssistantError`, same as `_send`, but without the retry loop — a single attempt.

## Component 2: Streaming Orchestration (orchestration.py)

New function, additive alongside `summarize_via_graph`:

```python
def summarize_via_graph_stream(
    document: Document,
    client: Any | None = None,
    threshold: int | None = None,
) -> Iterator[str]:
    """As summarize_via_graph, but yields incrementally instead of returning
    a single string. Whole-document path streams directly; the chunk-and-merge
    path yields short progress lines during chunk summarization and streams
    the final merge step."""
```

Behavior:
- Same lazy-`ClaudeClient()` default and `config.get_chunk_token_threshold()` fallback as `summarize_via_graph`.
- Route the same way `_route` does (`document.token_estimate <= threshold`).
- **Whole-document path:** `yield from client.complete_stream(build_summary_messages(document), system=SUMMARY_SYSTEM_PROMPT)`.
- **Chunk-and-merge path:**
  1. `chunks = chunk_document(document, max_tokens=threshold)` — reuse directly, don't reimplement.
  2. For each chunk, yield a short progress line (e.g. `f"[summarizing chunk {i+1} of {len(chunks)}]\n"`), then call `client.complete(...)` (non-streamed — this is the resolved PRD trade-off) to get that chunk's partial result, collected into `partials`.
  3. Once all partials are collected, `yield from client.complete_stream(build_merge_messages(partials), system=MERGE_SYSTEM_PROMPT)`.
- Apply `_strip_code_fence` to fully-assembled text where it matters for downstream JSON parsing (the caller assembling the stream is responsible for this — see the note in Component 3 on why the streaming endpoint doesn't validate JSON mid-flight).
- This function does **not** invoke the compiled `_GRAPH` — it's a parallel, simpler code path that reuses `chunk_document` and the prompt builders directly. Threading incremental yields through LangGraph's node/state model would add real complexity for no benefit here, since progress visibility, not graph branching, is the actual goal this pass.

## Component 3: FastAPI Streaming Route + Auth Gate (api.py)

**Auth dependency** (new, small — can live in `api.py` or a new `assistant/api_auth.py`):
```python
def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
    token = config.get_api_token()
    if token is None:
        return  # No token configured — auth is a no-op. Local/dev/test default.
    expected = f"Bearer {token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid API token")
```
`config.get_api_token() -> str | None` reads `ASSISTANT_API_TOKEN` from the environment, returning `None` if unset — mirrors the pattern of `get_chunk_token_threshold()` but optional rather than defaulted. **Enforcement is conditional on the env var being set.** This means: existing tests and local dev continue exactly as today (no token configured → routes stay open, zero test changes required for regression safety), and the gate activates automatically the moment `ASSISTANT_API_TOKEN` is set — which is the deploy step for Saturday, not a separate flag to remember.

Apply `Depends(require_auth)` to:
- `POST /documents`
- `POST /documents/{doc_id}/summarize`
- `POST /documents/{doc_id}/summarize/stream` (new, below)

Leave unauthenticated: `GET /health`, `GET /documents`, `GET /documents/{doc_id}`, `/docs`. These don't call the Claude API and cost nothing to hit — gating them would break the "Swagger stays browsable" success criterion from the PRD for no security benefit.

**New route:**
```python
@app.post("/documents/{doc_id}/summarize/stream", tags=["documents"], dependencies=[Depends(require_auth)])
async def summarize_stream(doc_id: int) -> StreamingResponse:
    document = _load_document(doc_id)   # reuse as-is — same 404 behavior
    return StreamingResponse(
        summarize_via_graph_stream(document),
        media_type="text/plain",
    )
```
Notes:
- No `response_model` — `StreamingResponse` doesn't have one; document the shape in the route's docstring instead.
- Starlette's `StreamingResponse` accepts a plain (synchronous) generator and iterates it in a threadpool automatically (`iterate_in_threadpool`) — no manual `async def` generator or asyncio bridging needed. Don't over-engineer this.
- The existing `POST /documents/{doc_id}/summarize` route is unchanged — it remains the source of truth for a validated `SummaryOut`. The streaming route intentionally returns raw assembling text, not a validated/typed body: JSON can't be meaningfully validated mid-stream, and forcing the client to wait for the full body to validate would defeat the point of streaming. A caller wanting typed output should accumulate the stream and parse it client-side, or just call the non-streaming route.
- **Known limitation, documented not hidden:** Swagger's own "Try it out" panel does not render text as it arrives — it waits for the full response body like any other route. The visible streaming effect is for a terminal client (`curl -N`) or the CLI's `--stream` flag, not for someone clicking around `/docs`. Don't oversell this in the README as a Swagger-visible effect.

## Component 4: CLI Streaming Flag (main.py)

- New thin wrapper `summarize_document_stream(document, client=None, threshold=None) -> Iterator[str]` in `main.py`, delegating to `orchestration.summarize_via_graph_stream` — same pairing pattern as the existing `summarize_document` → `summarize_via_graph` delegation.
- Add `--stream` flag to the `summarize` subcommand in `build_parser()`.
- `_cmd_summarize`: when `--stream` is set, iterate `summarize_document_stream(document)` and `print(chunk, end="", flush=True)` per chunk, with a trailing newline at the end. When not set, behavior is exactly as today (`_pretty_print_result(summarize_document(document))`).
- Streamed CLI output is raw text, not pretty-printed JSON — there's no complete JSON to reformat until the stream ends. This mirrors the API streaming route's same trade-off; document it in the README's usage section rather than trying to buffer-then-pretty-print (which would silently defeat the purpose of `--stream`).

## Config Additions (config.py / .env.example)
```
ASSISTANT_API_TOKEN=      # unset = auth disabled (local/dev default).
                           # Set this before deploying publicly.
```
New function in `config.py`:
```python
def get_api_token() -> str | None:
    """The inbound API token, or None if auth is disabled (unset)."""
    token = os.getenv("ASSISTANT_API_TOKEN")
    return token.strip() if token and token.strip() else None
```

## Acceptance Criteria
- [ ] `ClaudeClient.complete_stream()` yields text deltas from a real streamed API call, and does not retry on failure (single attempt, wrapped in `AssistantError`)
- [ ] `orchestration.summarize_via_graph_stream()` streams the whole-document path directly, and streams only the final merge step on the chunk-and-merge path, yielding progress lines during per-chunk summarization
- [ ] `summarize_via_graph()` (non-streaming) and the compiled `_GRAPH` are byte-for-byte unchanged — this pass adds a parallel path, it doesn't touch the existing one
- [ ] `main.py` gains a `--stream` flag on `summarize` that prints incrementally; default (non-flagged) behavior is unchanged
- [ ] `POST /documents/{doc_id}/summarize/stream` returns a `StreamingResponse` that yields visibly over `curl -N`, and 404s on an unknown id via the same `_load_document` helper
- [ ] `require_auth` blocks `POST /documents`, `POST /documents/{doc_id}/summarize`, and the new stream route with `401` when `ASSISTANT_API_TOKEN` is set and the header is missing/wrong — and is a no-op (routes stay open) when the env var is unset
- [ ] `GET /health`, `GET /documents`, `GET /documents/{doc_id}`, and `/docs` remain unauthenticated in all cases
- [ ] All existing tests (215 base + orchestration + API) still pass unmodified with no token configured — the default test environment stays exactly as it is today
- [ ] New tests: `complete_stream` (yields chunks; single-attempt-no-retry on failure), `summarize_via_graph_stream` (both paths, using a `FakeClient` extended with a `complete_stream` method), the auth dependency (open when unset, 401 when set and wrong/missing, 200 when set and correct), and the new streaming route (200 + content assembly, 404 for missing doc) — all network-free, same `FakeClient` convention as the rest of the suite

## Explicitly Out of Scope for This Pass
- Streaming every per-chunk call — only the whole-document path and the final merge step stream; per-chunk calls stay synchronous with a progress-line placeholder (resolved PRD trade-off)
- Retrying a streaming call after it has started yielding chunks
- Server-Sent Events framing (`text/event-stream`) — plain incremental `text/plain` is sufficient since there's no frontend consumer yet; SSE is a reasonable future upgrade if one gets built
- Multi-user auth, per-caller quotas, or key rotation — a single shared bearer token is the whole mechanism for this pass
- Gating the CLI behind the same token — the CLI uses the Anthropic key directly and is unaffected by this pass
- Any change to `_GRAPH`, `summarize_via_graph`, the chunking rule, or the prompts — this pass is additive only

## Notes for Claude Code
- `complete_stream` and `summarize_via_graph_stream` are new, parallel code paths — do not refactor `complete()`, `summarize_via_graph()`, or the compiled graph to "share more code" with them. The non-streaming path's stability (215+ passing tests) is the higher priority; a little duplication between the streaming and non-streaming chunk-summarization loops is an acceptable, deliberate trade-off, not something to eliminate.
- Extend the existing `FakeClient` test helper (from `tests/test_main.py`'s pattern) with a `complete_stream` method that yields from a canned list of strings, rather than inventing a second fake — keep the test-double convention consistent across the suite.
- `require_auth`'s "no token configured → no enforcement" behavior is intentional and load-bearing for backward compatibility — don't change it to fail closed by default, and don't require every existing test to start setting `ASSISTANT_API_TOKEN`.
- Update `.env.example` and `README.md` once this passes: document `ASSISTANT_API_TOKEN`, the `--stream` CLI flag, the new streaming route, and the Swagger-doesn't-visibly-stream caveat so it doesn't read as a bug later.
