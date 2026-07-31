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
| Route | Purpose | Auth |
| ----- | ------- | ---- |
| `GET /health` | Liveness probe | open |
| `POST /documents` | Upload and ingest a file (multipart, field name `file`) | 🔒 |
| `GET /documents` | List ingested documents | open |
| `GET /documents/{id}` | One document's metadata plus a ~500-char preview | open |
| `POST /documents/{id}/summarize` | Summary, decisions, and action items | 🔒 |
| `POST /documents/{id}/summarize/stream` | Same, streamed as `text/plain` | 🔒 |

Summarizing requires `ANTHROPIC_API_KEY`; ingesting and listing don't.

**Auth.** Routes marked 🔒 cost money to call, so they sit behind a shared
bearer token. With `ASSISTANT_API_TOKEN` unset — the local and test default —
the gate is a no-op and everything stays open. Set it and those routes start
returning `401` without `Authorization: Bearer <token>`. Read-only routes and
`/docs` stay open either way, so Swagger remains browsable.

```bash
curl -H "Authorization: Bearer $ASSISTANT_API_TOKEN" \
     -F "file=@notes.txt" http://localhost:8000/documents
```

**Streaming.** `/summarize/stream` returns text as it is generated rather than
one complete body. Small documents stream directly; larger ones emit a
`[summarizing chunk N of M]` line per chunk, then stream the merged result.

```bash
curl -N -X POST http://localhost:8000/documents/1/summarize/stream
```

The response is raw `text/plain`, not a validated `SummaryOut` — a body can't
be validated while it's still arriving. For typed output use the non-streaming
route, or accumulate the stream and parse it client-side.

> Swagger's "Try it out" panel buffers the whole response before rendering, so
> the incremental effect isn't visible there. Use `curl -N` or the CLI's
> `--stream` flag to actually see it. That's a limitation of the Swagger UI,
> not of the route.

### CLI

```bash
# Ingest a document — prints the new id and token estimate
python -m assistant.main ingest ./meeting_notes.pdf

# Summarize a stored document by id — prints structured JSON
python -m assistant.main summarize --id 3

# Same, but print incrementally as the model generates it
python -m assistant.main summarize --id 3 --stream

# List what's been ingested
python -m assistant.main list
```

`--stream` prints raw text rather than pretty-printed JSON: there's no complete
document to reformat until the stream ends, and buffering to reformat would
defeat the point. Omit the flag for formatted output.

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
docker run -p 8080:8080 --env-file .env doc-assistant
```

Then open http://localhost:8080/docs. The image is ~300MB and runs as a
non-root user.

`--env-file .env` keeps the key out of your shell history; `-e
ANTHROPIC_API_KEY=sk-ant-...` works too if you'd rather pass it inline.

The Dockerfile is deliberately platform-agnostic — Render, Fly.io, Railway, and
Cloud Run all accept a Dockerfile directly, so pick whichever has the fastest
free-tier setup rather than committing to one vendor's config format.

> **Storage is ephemeral.** SQLite lives on the container's local filesystem,
> which most free-tier hosts wipe on redeploy or restart. Documents ingested in
> one session may not survive into the next. That's an accepted limitation for
> a demo — a reviewer ingests and summarizes in one sitting — not a bug to fix
> with a managed database.

### Going public

**1. Generate a token.** Use a CSPRNG — not a password generator, not something
memorable:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

That's ~256 bits, URL-safe. `openssl rand -base64 32` works too.

**2. Set both secrets on the host.** Neither belongs in the image or the repo —
`.env` is for local use only:

| Host | How |
| ---- | --- |
| Render | Dashboard → Environment → Add Environment Variable |
| Fly.io | `fly secrets set ASSISTANT_API_TOKEN=... ANTHROPIC_API_KEY=...` |
| Railway | Variables tab |
| Cloud Run | `--set-env-vars`, or Secret Manager |

**3. Verify the gate is actually on.** This step matters more than it looks:
setting the env var *is* the activation mechanism, so a typo in the variable
**name** fails open silently — the routes stay wide open and nothing errors.

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST \
  https://your-app.example.com/documents/1/summarize
# expect: 401
```

Anything other than `401` means the token isn't being read. Check that first,
before assuming the deploy succeeded.

**Rotating the token requires a restart.** `get_api_token()` is read per
request, but the process only sees the environment it was started with. Change
the value on the host and redeploy; there's no live swap.

> **What this token does and doesn't protect.** One shared token means everyone
> you give it to has full access, and revoking it revokes everyone — the right
> scope for a demo, and explicitly all this pass sets out to do. But if the
> token leaks (a Slack paste, a terminal screenshot), someone can spend your
> Anthropic credits until you notice and redeploy. If the URL goes wider than a
> handful of reviewers, the next safeguard isn't more auth — it's a spending cap
> on the Anthropic side, since that's the failure mode that actually costs
> money.

## Configuration

Set in `.env` (see `.env.example`):

| Variable                | Default            | Purpose |
| ----------------------- | ------------------ | ------- |
| `ANTHROPIC_API_KEY`     | —                  | Required for `summarize` |
| `CLAUDE_MODEL`          | `claude-sonnet-5`  | Model for all completions |
| `CHUNK_TOKEN_THRESHOLD` | `150000`           | Above this, documents are chunked |
| `ASSISTANT_DB_PATH`     | `assistant.db`     | SQLite location |
| `MAX_UPLOAD_BYTES`      | `5000000` (~5MB)   | Uploads above this get a `413`. Read by `api.py`, not `config.py` — nothing outside the HTTP layer has a notion of upload size. |
| `ASSISTANT_API_TOKEN`   | unset              | Bearer token for the API's cost-incurring routes. Unset disables the gate; set it before deploying publicly. Does not affect the CLI. |

## Design notes

**Chunking.** Documents at or below the threshold are sent whole — there is no
retrieval or vector store in this pass. Above it, the document is split on
speaker-turn boundaries (or paragraph boundaries for non-transcripts) with
~500 tokens of overlap, each chunk is analyzed, and the partial results are
merged in a final call.

**Orchestration.** That branching is a LangGraph `StateGraph` in
`orchestration.py` rather than inline `if`/`else` — see
[docs/summarization-graph.md](docs/summarization-graph.md) for the diagram.
The graph normalizes its own output: both system prompts forbid code fences,
but the model wraps the merge response in ```` ```json ```` often enough to
matter (3 of 4 identical runs), which turned a correct answer into a `502`
from the HTTP layer. Stripping happens once in the graph so every caller gets
parseable JSON, rather than each consumer re-implementing the same guard.
The payoff is inspectability: the pipeline is an artifact you can render and
point at, and adding a step (a verification pass, a retry node) means adding a
node rather than threading another branch through a function. The cost is real
and worth naming — LangGraph pulls in ~19 transitive packages for a pipeline
whose control flow fits in 20 lines of Python. At this size it's a bet on where
the pipeline is going, not a simplification of where it is.

**Prompt caching.** The system prompt is frozen — no timestamps or per-request
interpolation — and carries a `cache_control` breakpoint, so repeated calls
aren't re-billed for those tokens. Static document content passed as system
blocks is cached alongside it. Verify hits via
`ClaudeClient.complete_raw(...).usage.cache_read_input_tokens`.

**Errors.** Raw SDK and parser exceptions are wrapped in `AssistantError`, so
the CLI prints a message rather than a traceback. Rate limits, connection
errors, and 5xx responses are retried up to 3 times with exponential backoff
and jitter; 4xx responses are not retried.

**Streaming deliberately doesn't retry.** `complete_stream()` makes a single
attempt. Once partial output has reached a terminal or an HTTP client, retrying
from scratch would re-send text the caller already saw — worse than failing
visibly. For the same reason a failure *after* the first byte can't become a
`502`: the status line is already sent, so the streaming route appends an
`[error: ...]` line to the body instead of dropping the connection silently.

**Only two calls stream.** The whole-document path streams, and so does the
final merge on the chunk-and-merge path. Per-chunk calls stay synchronous
behind progress lines — streaming every intermediate call adds real complexity
for the same perceived responsiveness. The streaming path is a parallel
implementation rather than a mode on the compiled graph, since threading
incremental yields through LangGraph's state model would buy nothing here.

## Layout

```
assistant/
  api.py                # FastAPI routes — thin layer over main.py and store.py
  api_auth.py           # Shared bearer-token gate for cost-incurring routes
  api_models.py         # HTTP request/response schemas
  orchestration.py      # LangGraph StateGraph for the summarization pipeline
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
| App runs from the Dockerfile on the configured port | ✅ | Image built and run; all routes exercised inside the container |
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
