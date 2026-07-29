# Dev Spec: FastAPI Wrapper for Document/Meeting Assistant

**Scope for this pass:** expose the existing CLI logic over HTTP so the project has a deployable, clickable public demo. No hand-built frontend — the interactive Swagger UI (`/docs`) that FastAPI generates automatically *is* the UI for this pass. No auth, no rate limiting beyond a basic upload-size cap, no changes to the ingestion/client/store logic itself.

## Why this exists
The core assistant (SDK wiring, ingestion, prompt caching, persistence) is built and tested — 215 passing tests, verified live API round-trip and cache hit. What it doesn't have is anything a reviewer can click without cloning the repo and running Python locally. This pass wraps the existing functions in routes and deploys them, so there's a public URL where someone can upload a file and get a summary back through a browser.

## Tech Stack (additions only)
- `fastapi`
- `uvicorn[standard]` (ASGI server)
- `python-multipart` (required by FastAPI for file uploads)
- Everything else is unchanged from the existing `requirements.txt`.

## Project Structure (additions only)
```
assistant/
  api.py             # FastAPI app + routes. Thin — calls into main.ingest_file,
                      # main.summarize_document, and store.* directly.
  api_models.py       # Pydantic request/response schemas for the API layer.
                      # Deliberately separate from models.py: those are the
                      # persistence/domain models, these are the HTTP contract.
Dockerfile
.dockerignore
```

Nothing in `main.py`, `client.py`, `store.py`, `config.py`, or `ingestion/` changes. This pass only adds a layer on top.

## API Models (api_models.py)
```python
class DocumentOut(BaseModel):
    id: int
    title: str | None
    doc_type: DocType
    token_estimate: int
    created_at: datetime | None
    speaker_turn_count: int | None      # None if not a transcript
    chunked: bool
    chunk_count: int | None             # only set if chunked

class DocumentDetailOut(DocumentOut):
    raw_text_preview: str               # first ~500 chars, not the full text —
                                         # avoids dumping huge PDFs into a JSON response

class ActionItemOut(BaseModel):
    task: str
    owner: str | None
    due: str | None

class SummaryOut(BaseModel):
    summary: str
    decisions: list[str]
    action_items: list[ActionItemOut]
```

## Routes (api.py)

**`GET /health`**
Returns `{"status": "ok"}`. Used by the deployment platform's health check, not by reviewers.

**`POST /documents`** — multipart file upload (field name: `file`)
- Reject anything over `MAX_UPLOAD_BYTES` (see Config) with `413`.
- Write the upload to a `tempfile.NamedTemporaryFile`, preserving the original suffix (`.pdf`, `.docx`, `.txt`, `.vtt`, `.srt`) so `loaders.detect_format` keeps working off the extension. Clean up the temp file in a `finally` block regardless of outcome.
- Call the existing `ingest_file(temp_path)` — do not reimplement ingestion logic here.
- Return `DocumentOut` (id, type, token estimate, whether it was chunked, etc).

**`GET /documents`**
Calls `store.list_documents()`, returns `list[DocumentOut]`.

**`GET /documents/{id}`**
Calls `store.get_document(id)`. Returns `DocumentDetailOut` (adds the truncated text preview). Raise `404` if `store.get_document` raises `AssistantError` for a missing id — check the specific case explicitly rather than relying on the generic handler below, so a bad id reads clearly as "not found" and not as a generic upstream failure.

**`POST /documents/{id}/summarize`**
- Fetch the document via `store.get_document(id)` (404 if missing, same as above).
- Call the existing `summarize_document(document)` — reuse it as-is, including its chunk-and-merge behavior for large documents.
- `summarize_document` returns a raw string that's expected to be JSON. Try `json.loads()` on it:
  - On success, validate/coerce into `SummaryOut` and return it.
  - On failure, return `502` with the raw text in the error detail — this means the model didn't follow the output format, which is a real signal worth surfacing, not something to paper over.

**Shared exception handling**
Register one `@app.exception_handler(AssistantError)` as a catch-all for anything not handled explicitly in a route (e.g. missing `ANTHROPIC_API_KEY`, retries exhausted against the Claude API, a corrupted DB read). Return `502` with `{"detail": str(exc)}` — these are upstream/config failures, not client errors, so `502` fits better than a blanket `500`.

## Config additions (config.py / .env.example)
```
MAX_UPLOAD_BYTES=5000000   # ~5MB. Rejects larger uploads with 413.
PORT=8080                   # read by the Dockerfile's CMD, not by config.py directly
```

## Deployment
**Dockerfile** — `python:3.11-slim` base, install `requirements.txt`, copy the app, then:
```
CMD ["uvicorn", "assistant.api:app", "--host", "0.0.0.0", "--port", "8080"]
```
Deliberately platform-agnostic (Render, Fly.io, Railway, Cloud Run all take a Dockerfile) rather than committing to one vendor's config format — pick whichever has the fastest free-tier setup when you get there.

**`.dockerignore`**: `.venv/`, `__pycache__/`, `.git/`, `tests/`, `assistant.db`, `.pytest_cache/` — don't bake a stale local DB or the dev venv into the image.

**Known limitation, not a bug:** SQLite lives on the container's local filesystem. On most free-tier PaaS hosts that filesystem is ephemeral — data ingested in one session may not survive a redeploy or restart. That's acceptable for a demo (a reviewer ingests and summarizes in one sitting) and shouldn't be "fixed" by adding a managed DB in this pass; note it in the README instead.

## Acceptance Criteria for This Pass
- [ ] `GET /health` returns 200
- [ ] `POST /documents` accepts a real upload (test with a `.pdf`, `.docx`, and `.txt`), ingests it via the existing `ingest_file`, and returns metadata as JSON
- [ ] `GET /documents` lists previously ingested documents
- [ ] `GET /documents/{id}` returns one document's metadata, `404` on an unknown id
- [ ] `POST /documents/{id}/summarize` returns parsed `summary` / `decisions` / `action_items`, using the existing chunk-and-merge path for oversized documents
- [ ] Uploads over `MAX_UPLOAD_BYTES` return `413`, not a silent accept or a `500`
- [ ] `/docs` (Swagger UI) is fully interactive end-to-end: a reviewer with no code can upload a file and get a summary back
- [ ] The app builds and runs from the provided `Dockerfile` and responds on the configured port
- [ ] The existing CLI (`python -m assistant.main ...`) still behaves identically — this pass only adds routes, it doesn't touch CLI behavior
- [ ] All 215 existing tests still pass unmodified; new tests cover each route's happy path plus the 404 and 413 cases, using `fastapi.testclient.TestClient` and the same `FakeClient` pattern already used in `tests/test_main.py` (no real network calls)

## Explicitly Out of Scope for This Pass
- Authentication or API keys for callers of this service
- Rate limiting or per-caller quotas beyond the flat file-size cap — real rate limiting, cost tracking, and a guardrails dashboard are Week 3's AI Gateway service, not this wrapper
- A hand-built frontend — Swagger's `/docs` is the UI here
- A managed/persistent database — SQLite stays as-is; ephemeral storage on the deploy host is accepted, not solved
- Streaming responses over HTTP (the underlying `complete()` call is synchronous; can be added later if the demo needs it)
- Any change to `main.py`, `client.py`, `store.py`, `config.py`, or the `ingestion/` modules — this pass is additive only

## Notes for Claude Code
- Reuse `ingest_file`, `summarize_document`, and the `store` module exactly as they exist today — this spec is a thin HTTP layer on top, not a rewrite.
- `summarize_document` accepts an optional `client` argument specifically to support the `FakeClient` pattern already established in `tests/test_main.py`; use FastAPI's dependency-injection or a simple default-arg override in tests so API tests stay network-free like the rest of the suite.
- Write the API tests alongside `api.py`, not after — same convention as the rest of this repo.
- Once this passes, update `README.md`: change the "Scope of this pass" line, add `uvicorn assistant.api:app --reload` for local dev, and add the Docker run command plus the ephemeral-storage caveat.
