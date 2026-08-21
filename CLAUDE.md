# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Ingests documents (PDF, DOCX, meeting transcripts, plain text), normalizes them, and sends them to the Claude API to produce structured output — summaries, decisions, and action items with owners. Two interfaces (CLI and FastAPI) sit on top of the same core functions; there's no hand-built frontend, just the Swagger UI at `/docs`.

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add ANTHROPIC_API_KEY

# Run the API
uvicorn assistant.api:app --reload    # http://127.0.0.1:8000/docs

# CLI
python -m assistant.main ingest ./meeting_notes.pdf
python -m assistant.main summarize --id 3 [--stream]
python -m assistant.main list
# both accept --db PATH (default: assistant.db / ASSISTANT_DB_PATH)

# Tests
python -m pytest tests/ -q                        # full suite, no network calls
python -m pytest tests/test_orchestration.py -q   # single file
python -m pytest tests/test_api.py -k stream -q   # single test by keyword

# Live API validation (billable — real Anthropic calls)
python scripts/validate_live.py

# Docker
docker build -t doc-assistant .
docker run -p 8080:8080 --env-file .env doc-assistant
```

No lint/format command is configured in this repo.

## Architecture

```
assistant/
  api.py                # FastAPI routes — thin layer over main.py and store.py
  api_auth.py           # Shared bearer-token gate for cost-incurring routes
  api_models.py         # HTTP request/response schemas (wire contract)
  orchestration.py      # LangGraph StateGraph for the summarization pipeline
  client.py             # Anthropic SDK wrapper: retry, caching, error wrapping
  config.py             # env loading, model constant, thresholds
  errors.py             # AssistantError
  models.py             # Document, SpeakerTurn, IngestResult (persistence models)
  store.py              # SQLite persistence
  main.py               # CLI entrypoint
  ingestion/
    loaders.py          # per-format parsers -> raw text
    normalize.py        # raw text -> Document; token estimate; chunking
    transcripts.py      # speaker/timestamp-aware parsing
  prompts/
    meeting_summary.py  # system prompt templates
```

**Single core, two interfaces.** `api.py` calls the same `ingest_file`, `summarize_document`, and `store` functions the CLI does — the HTTP layer only adds routing and validation, never business logic. `api_models.py` is deliberately separate from `models.py`: one is the wire contract, the other is persistence, and they're free to evolve independently.

**Summarization is a LangGraph `StateGraph`** (`orchestration.py`), not inline `if`/`else`:

```
START ─┬─(token_estimate <= threshold)─→ summarize_whole ──→ END
       └─(over threshold)──────────────→ chunk ──→ summarize_chunks ──→ merge ──→ END
```

Documents at or below `CHUNK_TOKEN_THRESHOLD` go out whole (no retrieval/vector store in this pass). Above it, the document is split on speaker-turn or paragraph boundaries with ~500 tokens of overlap, each chunk is summarized, and partials are merged in a final call. See [docs/summarization-graph.md](docs/summarization-graph.md) for the rendered diagram.

The graph strips Markdown code fences from model output itself (`_strip_code_fence` in `orchestration.py`) — both system prompts forbid fences, but the model wraps merge responses in ` ```json ` often enough (3 of 4 runs observed) to break `json.loads` downstream. Fixed once in the graph rather than in every caller.

**Streaming is a parallel path, not a graph mode.** `summarize_via_graph_stream` reimplements the same routing logic (`document.token_estimate <= threshold`) rather than threading incremental yields through LangGraph's state model. Only the whole-document call and the final merge call actually stream; per-chunk calls in the chunk-and-merge path stay synchronous behind `[summarizing chunk N of M]` progress lines. Streaming never retries — once partial output has reached a client, retrying from scratch would resend text already seen, so a mid-stream failure appends `[error: ...]` to the body instead of retrying or dropping the connection. Streamed output is raw text, not validated `SummaryOut` — there's no fence-stripping either, since you can't strip a fence mid-stream.

**Auth.** Routes that cost money (`POST /documents`, `POST /documents/{id}/summarize[/stream]`) sit behind a shared bearer token via `api_auth.py`. With `ASSISTANT_API_TOKEN` unset (local/test default), the gate is a no-op. Setting the env var is the only activation mechanism — a typo in the variable name fails open silently. Doesn't affect the CLI. Token is read per-request but the process only sees the env it started with, so rotation requires a restart.

**Errors and retries** (`client.py`, `errors.py`). Raw SDK/parser exceptions are wrapped in `AssistantError`. Rate limits, connection errors, and 5xx are retried up to 3x with exponential backoff + jitter; 4xx is not retried. This retry logic applies only to non-streaming calls — `complete_stream()` makes a single attempt, for the reason above.

**Prompt caching.** The system prompt is frozen (no timestamps/interpolation) and carries a `cache_control` breakpoint; static document content passed as system blocks is cached alongside it. Verify hits via `ClaudeClient.complete_raw(...).usage.cache_read_input_tokens`. The cacheable-prefix floor on `claude-sonnet-5` is 1024 tokens — `scripts/validate_live.py` pads the prompt past that floor deliberately. Don't read `normalize.estimate_tokens` (a `len/4` heuristic) as a billing figure; it's only meant to be approximately right for the chunking threshold. Use `client.messages.count_tokens()` for accurate counts.

**Ingestion formats** (`ingestion/loaders.py`, `ingestion/transcripts.py`): `.pdf` via `pypdf`, `.docx` via `python-docx`, `.vtt`/`.srt` cue blocks, `.txt`/`.md` sniffed as transcript (speaker-prefixed) or prose. Transcript parsing recognizes `[00:12:03] Alex: ...`, `00:12:03 Alex: ...`, bare `Alex: ...`, and WebVTT `<v Alex>` spans; files with no speaker markers become a single unattributed block.

## Configuration

Env vars, set in `.env` (see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | — | Required for `summarize` |
| `CLAUDE_MODEL` | `claude-sonnet-5` | Model for all completions |
| `CHUNK_TOKEN_THRESHOLD` | `150000` | Above this, documents are chunked |
| `ASSISTANT_DB_PATH` | `assistant.db` | SQLite location |
| `MAX_UPLOAD_BYTES` | `5000000` (~5MB) | Uploads above this get `413`. Read by `api.py`, not `config.py` |
| `ASSISTANT_API_TOKEN` | unset | Bearer token for cost-incurring routes; unset disables the gate |

## Deployment notes

Storage is ephemeral by design — SQLite on the container's local filesystem gets wiped on redeploy/restart by most free-tier hosts. This is an accepted limitation (a reviewer ingests and summarizes in one sitting), not a bug to fix with a managed database. The Dockerfile is deliberately platform-agnostic (Render/Fly.io/Railway/Cloud Run all accept it directly) rather than committing to one vendor's config format.
