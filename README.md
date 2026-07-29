# Document/Meeting Assistant

Ingests documents (PDF, DOCX, meeting transcripts, plain text), normalizes them,
and sends them to the Claude API to produce structured output — summaries,
decisions, and action items with owners.

**Scope of this pass:** SDK wiring, document ingestion, and an HTTP wrapper.
The interactive Swagger UI at `/docs` is the interface — there's no hand-built
frontend. No vector search, no auth.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

## Usage

### HTTP API

```bash
uvicorn assistant.api:app --reload
```

Then open **http://127.0.0.1:8000/docs** — the Swagger UI is fully interactive:
upload a file, list what's stored, and summarize it without writing any code.

| Route | Purpose |
| ----- | ------- |
| `GET /health` | Liveness probe |
| `POST /documents` | Upload and ingest a file (multipart, field name `file`) |
| `GET /documents` | List ingested documents |
| `GET /documents/{id}` | One document's metadata plus a ~500-char preview |
| `POST /documents/{id}/summarize` | Summary, decisions, and action items |

Summarizing requires `ANTHROPIC_API_KEY`; ingesting and listing don't.

### CLI

```bash
# Ingest a document — prints the new id and token estimate
python -m assistant.main ingest ./meeting_notes.pdf

# Summarize a stored document by id — prints structured JSON
python -m assistant.main summarize --id 3

# List what's been ingested
python -m assistant.main list
```

Both commands accept `--db PATH` to point at a different SQLite file
(default: `assistant.db`, overridable via `ASSISTANT_DB_PATH`).

### Example

```
$ python -m assistant.main ingest ./standup.txt
Ingested [3] standup
  type:           transcript
  token estimate: 89
  speaker turns:  5

$ python -m assistant.main summarize --id 3
{
  "summary": "The team reviewed the pending migration...",
  "decisions": ["Ship Thursday rather than Wednesday."],
  "action_items": [
    {"task": "Update the release notes", "owner": "Dev", "due": null}
  ]
}
```

## Supported formats

| Extension      | Handling |
| -------------- | -------- |
| `.pdf`         | Text extracted per page via `pypdf` |
| `.docx`        | Paragraphs extracted via `python-docx` |
| `.vtt`, `.srt` | Cue blocks parsed into speaker turns |
| `.txt`, `.md`  | Sniffed: speaker-prefixed content is parsed as a transcript, otherwise treated as prose |

Transcript parsing recognizes `[00:12:03] Alex: ...`, `00:12:03 Alex: ...`, and
bare `Alex: ...`, plus WebVTT `<v Alex>` voice spans. A file with no speaker
markers falls back to a single unattributed block.

## Deployment

```bash
docker build -t doc-assistant .
docker run -p 8080:8080 -e ANTHROPIC_API_KEY=sk-ant-... doc-assistant
```

Then open http://localhost:8080/docs.

The Dockerfile is deliberately platform-agnostic — Render, Fly.io, Railway, and
Cloud Run all accept a Dockerfile directly, so pick whichever has the fastest
free-tier setup rather than committing to one vendor's config format.

> **Storage is ephemeral.** SQLite lives on the container's local filesystem,
> which most free-tier hosts wipe on redeploy or restart. Documents ingested in
> one session may not survive into the next. That's an accepted limitation for
> a demo — a reviewer ingests and summarizes in one sitting — not a bug to fix
> with a managed database.

## Configuration

Set in `.env` (see `.env.example`):

| Variable                | Default            | Purpose |
| ----------------------- | ------------------ | ------- |
| `ANTHROPIC_API_KEY`     | —                  | Required for `summarize` |
| `CLAUDE_MODEL`          | `claude-sonnet-5`  | Model for all completions |
| `CHUNK_TOKEN_THRESHOLD` | `150000`           | Above this, documents are chunked |
| `ASSISTANT_DB_PATH`     | `assistant.db`     | SQLite location |
| `MAX_UPLOAD_BYTES`      | `5000000`          | Uploads above this get a `413` |

## Design notes

**Chunking.** Documents at or below the threshold are sent whole — there is no
retrieval or vector store in this pass. Above it, the document is split on
speaker-turn boundaries (or paragraph boundaries for non-transcripts) with
~500 tokens of overlap, each chunk is analyzed, and the partial results are
merged in a final call.

**Prompt caching.** The system prompt is frozen — no timestamps or per-request
interpolation — and carries a `cache_control` breakpoint, so repeated calls
aren't re-billed for those tokens. Static document content passed as system
blocks is cached alongside it. Verify hits via
`ClaudeClient.complete_raw(...).usage.cache_read_input_tokens`.

**Errors.** Raw SDK and parser exceptions are wrapped in `AssistantError`, so
the CLI prints a message rather than a traceback. Rate limits, connection
errors, and 5xx responses are retried up to 3 times with exponential backoff
and jitter; 4xx responses are not retried.

## Layout

```
assistant/
  api.py                # FastAPI routes — thin layer over main.py and store.py
  api_models.py         # HTTP request/response schemas
  client.py             # Anthropic SDK wrapper: retry, caching, error wrapping
  config.py             # env loading, model constant, thresholds
  errors.py             # AssistantError
  models.py             # Document, SpeakerTurn, IngestResult
  store.py              # SQLite persistence
  main.py               # CLI entrypoint
  ingestion/
    loaders.py          # per-format parsers -> raw text
    normalize.py        # raw text -> Document; token estimate; chunking
    transcripts.py      # speaker/timestamp-aware parsing
  prompts/
    meeting_summary.py  # system prompt templates
Dockerfile
tests/
```

`api.py` calls the same `ingest_file`, `summarize_document`, and `store`
functions the CLI does — the HTTP layer adds routing and validation, not logic.
`api_models.py` is kept separate from `models.py` on purpose: those are the
persistence models, these are the wire contract, and they should be free to
change independently.

## Tests

```bash
python -m pytest tests/ -q
```

268 tests, no network calls — the Anthropic client is faked throughout, and the
API tests override the summarize dependency with the same pattern.

### Live API validation

Two acceptance criteria — a real round trip, and prompt-cache hits on repeated
calls — can't be covered by the faked suite. With a key in `.env`:

```bash
python scripts/validate_live.py
```

Makes 3 billable calls and reports the observed `cache_creation_input_tokens`
and `cache_read_input_tokens`. Note the script pads the system prompt past the
model's minimum cacheable prefix (1024 tokens on `claude-sonnet-5`); the
summary prompt alone is ~250 tokens, which is below the floor and would never
produce a cache hit regardless of whether caching is wired correctly.

Last run against `claude-sonnet-5`:

```
Criterion 3 — complete() round-trips a real prompt
PASS  round trip returned 4 chars
      response: 'PONG'

Criterion 4 — prompt caching on repeated calls
      system prompt ~1250 tokens (minimum 1024)
      call 1: cache_creation_input_tokens=1765
      call 2: cache_read_input_tokens=1765
PASS  second call read 1765 tokens from cache
```

Written and read counts match exactly, so the full cached prefix was reused
rather than partially invalidated between calls.

The script's own `~1250 tokens` line is the `len/4` estimate from
`normalize.estimate_tokens`; the API billed 1765 for the same text. That gap
is expected — `len/4` is the spec's rough heuristic and undercounts against
the real tokenizer. It is fine for the chunking threshold, which only needs to
be approximately right, but don't read `token_estimate` as a billing figure.
For accurate counts use `client.messages.count_tokens()`.

## Acceptance criteria

### HTTP wrapper

| Criterion | Status | Verified by |
| --------- | ------ | ----------- |
| `GET /health` returns 200 | ✅ | `tests/test_api.py`, plus live `curl` |
| `POST /documents` ingests a real upload and returns metadata | ✅ | `.pdf`, `.docx`, `.txt`, `.vtt` all covered |
| `GET /documents` lists ingested documents | ✅ | `tests/test_api.py` |
| `GET /documents/{id}` returns metadata; `404` on unknown id | ✅ | `tests/test_api.py`, plus live `curl` |
| `POST /documents/{id}/summarize` returns parsed summary/decisions/action items | ✅ | `tests/test_api.py` with a faked client |
| Uploads over `MAX_UPLOAD_BYTES` return `413` | ✅ | Live: an 8MB upload returned `413`, not a silent accept |
| `/docs` is interactive end-to-end | ✅ | Swagger UI served; full upload → summarize round trip |
| App runs from the Dockerfile on the configured port | ⚠️ | `uvicorn assistant.api:app` verified directly; **image build not run — no Docker in the dev environment** |
| CLI behavior unchanged | ✅ | Output byte-identical before and after |
| All 215 existing tests still pass unmodified | ✅ | 268 total = 215 existing + 53 new |

### Core assistant

| Criterion | Status | Verified by |
| --------- | ------ | ----------- |
| Ingest a PDF, DOCX, and plain-text meeting transcript without errors | ✅ | CLI against real files, plus `.vtt` and prose `.txt` |
| Ingested documents persist to SQLite and are retrievable by id | ✅ | Round trip preserves speaker turns, timestamps, `created_at` |
| `ClaudeClient.complete()` round-trips a prompt with a real API key | ✅ | `scripts/validate_live.py` |
| Prompt caching applied to the system prompt on repeated calls | ✅ | `scripts/validate_live.py` — call 1 wrote 1765 tokens, call 2 read 1765 back |
| Over-threshold documents chunked on speaker/paragraph boundaries; under-threshold passed whole | ✅ | 1248-token transcript → 11 chunks, split only at speaker turns |
| Unit tests cover each loader and the normalize function | ✅ | 215 tests |
