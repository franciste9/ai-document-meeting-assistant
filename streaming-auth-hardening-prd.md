# PRD: Streaming Responses + Auth Hardening

**Status:** Draft for review
**Owner:** Tiny
**Related:** `langgraph-orchestration-spec.md` (this depends on that being in place), dev spec to follow this PRD
**Item:** Week 2 — "Build: streaming responses + auth hardening"

## Problem

**Streaming.** `ClaudeClient.complete()` already accepts a `stream: bool` parameter, and `_send()` already calls `self._client.messages.stream(**request)` when it's set — but it immediately calls `.get_final_message()` on the stream and returns the aggregated result. Nothing about the caller's experience differs from `stream=False` today. For a single short document this doesn't matter, but the graph built on Wednesday makes multiple sequential model calls on the chunk-and-merge path (one per chunk, then a merge call) — a reviewer summarizing a longer transcript through the public demo would stare at a blank spinner for the full duration of several back-to-back API calls with zero feedback.

**Auth.** `config.get_api_key()` only checks that *this service* has a valid outbound Anthropic key. There's no check on who's allowed to call the FastAPI service itself. That gap was explicitly and correctly deferred in the FastAPI wrapper spec ("Authentication or API keys for callers of this service" is listed under its Out of Scope section) — reasonable, since nothing was deployed yet. Saturday's plan is to actually put this on a public URL. An open `/documents/{id}/summarize` endpoint sitting in front of a real, billed API key is a live cost/abuse surface the moment that URL is public, not a theoretical one.

## Goals

- Give callers visible incremental progress during multi-step summarization, particularly the chunk-and-merge path, instead of one long silent wait.
- Close the inbound-auth gap before the FastAPI service is actually deployed publicly this Saturday.
- Keep both changes proportionate to a portfolio demo — not a production auth system, not a full streaming protocol overhaul.

## Non-Goals

- No user accounts, sessions, or OAuth. A single shared credential gating the expensive routes is sufficient here.
- No production-grade rate limiting, quotas, or cost dashboards — still Week 3's AI Gateway scope, unchanged from the FastAPI spec's original boundary.
- No change to how the *outbound* Anthropic key is handled in `config.py` — that's already correct.
- No streaming on `ingest` — only `summarize` involves multiple sequential model calls worth showing progress on.
- No change to the graph's node structure or the chunking rule from Wednesday's work — this builds on top of it, not into it.

## Who This Is For

- A reviewer clicking through the public Swagger demo Saturday, who should see the summary build up rather than wait silently, and shouldn't be able to anonymously run charges against your API key.
- You, deploying Saturday — auth is a pre-deploy gate here, not an optional hardening pass for later.
- Future reuse of `ClaudeClient` in Week 3's Gateway service — a working, real streaming path is more reusable than a dead parameter that's technically wired but never actually streams anything.

## Success Criteria

- Summarizing a multi-chunk document shows visible incremental progress to the caller (streamed text and/or step-level status), not one long blocking wait.
- The FastAPI routes that cost money to call — `POST /documents` (ingestion is cheap but still a resource action) and `POST /documents/{id}/summarize` (the actual billed calls) — reject requests without a valid credential. `GET /health` and `/docs` stay open so the demo remains browsable without a key.
- No regression: all existing tests, including the LangGraph orchestration tests from Wednesday, still pass.
- The auth mechanism is simple enough to document in one line in the README (e.g., "include `Authorization: Bearer <token>`") — not a new system to explain.

## Requirements

1. `ClaudeClient` must expose a real streaming path — incremental output as it arrives from the API — not a blocking call to the streaming endpoint that discards everything until the end.
2. The LangGraph nodes that call the model must be able to run in streaming or non-streaming mode without changing their external contract (`summarize_via_graph()`'s signature and return type stay the same for non-streaming callers).
3. The CLI's `summarize` command should print incrementally when streaming is enabled, rather than waiting for the full result.
4. The FastAPI wrapper's cost-incurring routes must require a credential (e.g., a bearer token read from an environment variable) before deployment.
5. Missing or invalid credentials must return a clear `401`, distinguishable from the FastAPI spec's existing `502` (upstream Claude failure) and `404` (not found) cases.

## Risks & Open Questions

- **Risk — streaming through a multi-step graph is more complex than streaming one call.** Need to decide whether "streaming" means token-level streaming of every node's output, or just the final user-facing step. Recommendation: stream the whole-document path directly, and stream the final merge step on the chunk-and-merge path; treat per-chunk intermediate calls as non-streamed status updates ("summarizing chunk 2 of 5") rather than full token streams. Much lower complexity for effectively the same perceived benefit — worth confirming this trade-off before the dev spec locks it in.
- **Risk — a shared bearer token is a weak long-term auth model.** It's proportionate for a small-circle portfolio demo, but it's worth naming as a conscious trade-off in the ADR rather than something that reads as an oversight later.
- **Open question — does the CLI need the same credential?** Leaning no: the CLI runs locally against your own Anthropic key, already access-controlled by the machine it's on. The new inbound auth would apply to the network-facing API only. Flagging for your review before the dev spec assumes this.
- **Open question — where does the token live?** An environment variable is fine for now; a real secrets manager is Week 3 Gateway territory, not this pass.

## Out of Scope for This PRD

Implementation detail — exact header names, SSE vs. chunked-response format, where the token check is enforced (middleware vs. per-route dependency) — belongs in the dev spec, not here.
