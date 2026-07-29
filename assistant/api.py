"""FastAPI wrapper over the existing assistant.

Thin by design: routes validate input, call the same `ingest_file`,
`summarize_document`, and `store` functions the CLI uses, and shape the result
for the wire. No ingestion, persistence, or prompting logic lives here.

Run locally:
    uvicorn assistant.api:app --reload

The interactive Swagger UI at /docs is the interface for this pass.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from assistant import config, store
from assistant.api_models import (
    DocumentDetailOut,
    DocumentOut,
    ErrorOut,
    HealthOut,
    SummaryOut,
)
from assistant.errors import AssistantError
from assistant.main import ingest_file, summarize_document
from assistant.models import Document

# Extensions `loaders.detect_format` knows how to dispatch on. The temp file
# must keep the original suffix or detection fails.
SUPPORTED_SUFFIXES = frozenset({".pdf", ".docx", ".txt", ".md", ".vtt", ".srt"})

# Read uploads in chunks so the size cap can trip before the whole body is in
# memory. A flat `await file.read()` would buffer the entire upload first,
# which defeats the point of having a cap.
_UPLOAD_CHUNK_BYTES = 64 * 1024

app = FastAPI(
    title="Document/Meeting Assistant",
    version="0.2.0",
    description=(
        "Upload a document or meeting transcript, then summarize it into "
        "decisions and action items.\n\n"
        "Supported formats: PDF, DOCX, plain text, Markdown, WebVTT, SubRip. "
        "Plain-text files are sniffed for speaker markers and parsed as "
        "transcripts when they look like one.\n\n"
        "**Summarizing requires a configured `ANTHROPIC_API_KEY`;** ingestion "
        "and listing do not."
    ),
)


# -- dependencies -------------------------------------------------------------


def get_summarizer():
    """The summarize callable, overridable in tests via dependency_overrides.

    Keeps API tests network-free using the same fake-client pattern as the
    rest of the suite.
    """
    return summarize_document


# -- error handling -----------------------------------------------------------


@app.exception_handler(AssistantError)
async def handle_assistant_error(_: Request, exc: AssistantError) -> JSONResponse:
    """Catch-all for failures a route didn't handle explicitly.

    These are upstream or configuration problems — a missing API key, retries
    exhausted against the Claude API, a corrupted DB read — not malformed
    client requests, so 502 fits better than a blanket 500.
    """
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc)},
    )


def _is_missing_document(exc: AssistantError) -> bool:
    """True when the error is a missing id rather than a real store failure.

    `store.get_document` raises AssistantError for both; only the first should
    read as 404.
    """
    return str(exc).startswith("No document with id")


def _load_document(doc_id: int) -> Document:
    """Fetch a document, turning a missing id into a 404."""
    try:
        return store.get_document(doc_id)
    except AssistantError as exc:
        if _is_missing_document(exc):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No document with id {doc_id}",
            ) from exc
        # A genuine store failure — let the AssistantError handler return 502.
        raise


# -- routes -------------------------------------------------------------------


@app.get("/health", response_model=HealthOut, tags=["meta"])
async def health() -> HealthOut:
    """Liveness probe for the deployment platform."""
    return HealthOut()


@app.post(
    "/documents",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
    responses={
        413: {"model": ErrorOut, "description": "Upload exceeds the size limit"},
        415: {"model": ErrorOut, "description": "Unsupported file type"},
        502: {"model": ErrorOut, "description": "Ingestion failed"},
    },
)
async def create_document(
    file: Annotated[UploadFile, File(description="PDF, DOCX, TXT, MD, VTT or SRT")],
) -> DocumentOut:
    """Ingest an uploaded document and persist it.

    The file is written to a temp path — preserving its suffix, since format
    detection keys off the extension — and handed to the same `ingest_file`
    the CLI uses.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{suffix or file.filename}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            ),
        )

    max_bytes = config.get_max_upload_bytes()
    temp_path = await _spool_upload(file, file.filename or "", suffix, max_bytes)

    try:
        doc_id, result = ingest_file(temp_path)
    finally:
        _cleanup(temp_path)

    return DocumentOut.from_ingest_result(doc_id, result)


def _cleanup(temp_path: Path) -> None:
    """Remove the spooled file and the directory holding it."""
    temp_path.unlink(missing_ok=True)
    try:
        temp_path.parent.rmdir()
    except OSError:
        # Directory not empty or already gone — nothing worth failing over.
        pass


def _safe_name(filename: str, suffix: str) -> str:
    """A filesystem-safe name derived from an untrusted upload filename.

    The upload is written under this name inside a private temp directory, so
    `ingest_file` sees the caller's real filename: the suffix drives format
    detection, and the stem becomes the document title for transcripts. The
    value is client-controlled, so directory components are stripped and only
    known-safe characters are kept.
    """
    stem = Path(filename).stem
    cleaned = "".join(c for c in stem if c.isalnum() or c in "-_ ").strip()
    return f"{cleaned[:64] or 'upload'}{suffix}"


async def _spool_upload(
    file: UploadFile, filename: str, suffix: str, max_bytes: int
) -> Path:
    """Stream an upload into a private temp directory, enforcing the size cap.

    Raises 413 as soon as the cap is exceeded, without buffering the rest.
    A dedicated directory lets the file keep the caller's name without
    colliding with concurrent uploads of the same name.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="assistant-upload-"))
    temp_path = temp_dir / _safe_name(filename, suffix)
    written = 0

    try:
        with temp_path.open("wb") as handle:
            while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > max_bytes:
                    # Literal 413 rather than the status constant: Starlette
                    # renamed HTTP_413_REQUEST_ENTITY_TOO_LARGE to
                    # HTTP_413_CONTENT_TOO_LARGE, so the number is the stable
                    # spelling across versions.
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Upload exceeds the {max_bytes:,}-byte limit."
                        ),
                    )
                handle.write(chunk)
    except BaseException:
        # Includes the 413 above — never leave a temp file behind.
        _cleanup(temp_path)
        raise

    return temp_path


@app.get(
    "/documents",
    response_model=list[DocumentOut],
    tags=["documents"],
    responses={502: {"model": ErrorOut, "description": "Store read failed"}},
)
async def list_documents() -> list[DocumentOut]:
    """All ingested documents, oldest first."""
    documents = store.list_documents()
    ids = [doc_id for doc_id, _ in store.list_document_ids()]
    return [
        DocumentOut.from_document(doc_id, document)
        for doc_id, document in zip(ids, documents)
    ]


@app.get(
    "/documents/{doc_id}",
    response_model=DocumentDetailOut,
    tags=["documents"],
    responses={
        404: {"model": ErrorOut, "description": "No document with that id"},
        502: {"model": ErrorOut, "description": "Store read failed"},
    },
)
async def get_document(doc_id: int) -> DocumentDetailOut:
    """One document's metadata, plus a truncated preview of its text."""
    document = _load_document(doc_id)
    return DocumentDetailOut.from_document(doc_id, document)


@app.post(
    "/documents/{doc_id}/summarize",
    response_model=SummaryOut,
    tags=["documents"],
    responses={
        404: {"model": ErrorOut, "description": "No document with that id"},
        502: {
            "model": ErrorOut,
            "description": "The model returned unparseable output, or the API call failed",
        },
    },
)
async def summarize(
    doc_id: int,
    summarizer: Annotated[Any, Depends(get_summarizer)],
) -> SummaryOut:
    """Summarize a stored document into decisions and action items.

    Oversized documents go through the same chunk-and-merge path as the CLI.
    Requires `ANTHROPIC_API_KEY` to be configured.
    """
    document = _load_document(doc_id)
    raw = summarizer(document)

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        # The model didn't honor the output contract. Surface it rather than
        # silently returning an empty summary — it's a real signal.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Model did not return valid JSON: {raw!r}",
        ) from exc

    try:
        return SummaryOut.model_validate(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Model returned JSON in an unexpected shape: {raw!r}",
        ) from exc
