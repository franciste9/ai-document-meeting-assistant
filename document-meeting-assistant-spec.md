# Dev Spec: Document/Meeting Assistant
**Scope for this pass:** SDK wiring + document ingestion only. No UI, no vector search, no deployment.

## Goal
A Python CLI/service that ingests documents (PDFs, DOCX, meeting transcripts, plain text), normalizes them, and sends them to the Claude API to produce structured output (summaries, decisions, action items).

## Tech Stack
- Python 3.11+
- `anthropic` SDK (official Python client)
- `python-dotenv` for config
- `pypdf` or `pdfplumber` for PDF parsing
- `python-docx` for Word docs
- `pydantic` for data models / validation
- `sqlite3` (stdlib) for local persistence — no external DB for this pass

## Project Structure
```
/assistant
  __init__.py
  client.py            # Anthropic SDK wrapper
  config.py            # env var loading, model constants
  ingestion/
    __init__.py
    loaders.py         # per-format parsers -> raw text
    normalize.py        # raw text -> Document model
    transcripts.py      # speaker/timestamp-aware parsing for meetings
  models.py             # pydantic models: Document, SpeakerTurn, IngestResult
  store.py              # SQLite persistence for ingested docs
  prompts/
    __init__.py
    meeting_summary.py  # system prompt templates
  main.py               # CLI entrypoint
tests/
  test_loaders.py
  test_normalize.py
  test_client.py
.env.example
requirements.txt
README.md
```

## Data Models (models.py)
```python
class SpeakerTurn(BaseModel):
    speaker: str | None
    timestamp: str | None
    text: str

class Document(BaseModel):
    source_path: str
    doc_type: Literal["pdf", "docx", "transcript", "text"]
    title: str | None
    created_at: datetime | None
    raw_text: str
    speaker_turns: list[SpeakerTurn] | None = None
    token_estimate: int

class IngestResult(BaseModel):
    document: Document
    chunks: list[str] | None = None  # only populated if doc exceeds threshold
```

## Component 1: SDK Wiring (client.py)
- Instantiate `anthropic.Anthropic()` once at module load using `ANTHROPIC_API_KEY` from env
- `ClaudeClient` class with:
  - `complete(messages: list[dict], system: str = None, stream: bool = False) -> str`
  - Retry logic (exponential backoff) on `RateLimitError`, `APIStatusError`, `APIConnectionError` — max 3 retries
  - Model name pulled from `config.py` constant, not hardcoded per call (default: `claude-sonnet-5`)
  - Support `cache_control` blocks for system prompt + any static document content passed in context, to avoid re-billing full doc tokens on repeated calls
  - Raise a custom `AssistantError` on unrecoverable failures, don't let raw SDK exceptions leak to CLI

## Component 2: Document Ingestion
**loaders.py** — one function per format, each returns raw text + basic metadata:
- `load_pdf(path) -> str`
- `load_docx(path) -> str`
- `load_transcript(path) -> list[SpeakerTurn]` (handles .vtt, .srt, plain speaker-prefixed text)
- `load_text(path) -> str`
- `detect_format(path) -> str` dispatches by extension

**normalize.py**
- `normalize(raw_text_or_turns, source_path, doc_type) -> Document`
- Estimate token count (rough: `len(text) / 4`) and store on the model
- Strip excessive whitespace, normalize line endings

**transcripts.py**
- Parse speaker + timestamp when present (`[00:12:03] Alex: ...`)
- Fall back to treating the whole file as one unattributed block if no speaker markers found

**Chunking rule (only if needed):**
- If `token_estimate > 150_000`, split on speaker-turn or paragraph boundaries with ~500 token overlap
- Otherwise pass the whole document in context — no retrieval/vector store in this pass

## Component 3: Local Persistence (store.py)
- SQLite table `documents(id, source_path, doc_type, title, created_at, raw_text, token_estimate)`
- `save_document(doc: Document) -> int`
- `get_document(id: int) -> Document`
- `list_documents() -> list[Document]`

## Component 4: CLI Entrypoint (main.py)
```
python -m assistant.main ingest ./meeting_notes.pdf
python -m assistant.main summarize --id 3
```
- `ingest`: detect format → load → normalize → save to store → print doc id + token estimate
- `summarize`: fetch doc by id → build prompt from `prompts/meeting_summary.py` → call `ClaudeClient.complete()` → print structured JSON result (summary, decisions, action_items with owners)

## Config (config.py / .env.example)
```
ANTHROPIC_API_KEY=
CLAUDE_MODEL=claude-sonnet-5
CHUNK_TOKEN_THRESHOLD=150000
```

## Acceptance Criteria for This Pass
- [ ] Can ingest a PDF, DOCX, and a plain-text meeting transcript without errors
- [ ] Ingested documents persist to SQLite and are retrievable by id
- [ ] `ClaudeClient.complete()` successfully round-trips a prompt with a real API key
- [ ] Prompt caching is applied to the system prompt on repeated calls (verify via API usage/cache hit in response)
- [ ] Documents over the token threshold are chunked on speaker/paragraph boundaries; documents under threshold are passed whole
- [ ] Basic unit tests cover each loader and the normalize function

## Explicitly Out of Scope for This Pass
- Vector store / embeddings / semantic search across multiple documents
- Web UI or API server
- Multi-user auth
- Deployment/hosting config

## Notes for Claude Code
- Follow the folder structure above exactly; create stub files with docstrings first, then implement bottom-up (loaders → normalize → models → client → store → main).
- Write tests alongside each module, not at the end.
- Use type hints throughout; validate with pydantic at ingestion boundaries.
