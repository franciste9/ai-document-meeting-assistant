# Document/Meeting Assistant

Ingests documents (PDF, DOCX, meeting transcripts, plain text), normalizes them,
and sends them to the Claude API to produce structured output — summaries,
decisions, and action items with owners.

**Scope of this pass:** SDK wiring + document ingestion. No UI, no vector
search, no deployment.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

## Usage

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

## Configuration

Set in `.env` (see `.env.example`):

| Variable                | Default            | Purpose |
| ----------------------- | ------------------ | ------- |
| `ANTHROPIC_API_KEY`     | —                  | Required for `summarize` |
| `CLAUDE_MODEL`          | `claude-sonnet-5`  | Model for all completions |
| `CHUNK_TOKEN_THRESHOLD` | `150000`           | Above this, documents are chunked |
| `ASSISTANT_DB_PATH`     | `assistant.db`     | SQLite location |

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
tests/
```

## Tests

```bash
python -m pytest tests/ -q
```

215 tests, no network calls — the Anthropic client is faked throughout.

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
