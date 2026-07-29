"""Request/response schemas for the HTTP layer.

Deliberately separate from `models.py`: those are the persistence/domain
models, these are the wire contract. Keeping them apart means the API shape
can change without touching what gets stored, and vice versa.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from assistant.models import Document, DocType, IngestResult

# How much of the document body `GET /documents/{id}` echoes back. Full text
# would mean dumping an entire PDF into a JSON response.
PREVIEW_CHARS = 500


class DocumentOut(BaseModel):
    """Metadata for one ingested document."""

    id: int
    title: str | None = None
    doc_type: DocType
    token_estimate: int
    created_at: datetime | None = None
    speaker_turn_count: int | None = Field(
        default=None, description="Number of speaker turns; null if not a transcript."
    )
    chunked: bool = Field(
        default=False,
        description="Whether the document exceeded the token threshold and was chunked.",
    )
    chunk_count: int | None = Field(
        default=None, description="Number of chunks; only set when chunked."
    )

    @classmethod
    def from_document(
        cls,
        doc_id: int,
        document: Document,
        chunks: list[str] | None = None,
    ) -> DocumentOut:
        return cls(
            id=doc_id,
            title=document.title,
            doc_type=document.doc_type,
            token_estimate=document.token_estimate,
            created_at=document.created_at,
            speaker_turn_count=(
                len(document.speaker_turns)
                if document.speaker_turns is not None
                else None
            ),
            chunked=bool(chunks),
            chunk_count=len(chunks) if chunks else None,
        )

    @classmethod
    def from_ingest_result(cls, doc_id: int, result: IngestResult) -> DocumentOut:
        return cls.from_document(doc_id, result.document, result.chunks)


class DocumentDetailOut(DocumentOut):
    """One document, with a truncated preview of its text."""

    raw_text_preview: str = Field(
        description=f"First ~{PREVIEW_CHARS} characters of the document body."
    )

    @classmethod
    def from_document(
        cls,
        doc_id: int,
        document: Document,
        chunks: list[str] | None = None,
    ) -> DocumentDetailOut:
        base = DocumentOut.from_document(doc_id, document, chunks)
        return cls(
            **base.model_dump(),
            raw_text_preview=_preview(document.raw_text),
        )


def _preview(text: str) -> str:
    """Truncate to `PREVIEW_CHARS`, marking that content was cut."""
    if len(text) <= PREVIEW_CHARS:
        return text
    return text[:PREVIEW_CHARS] + "…"


class ActionItemOut(BaseModel):
    """One piece of follow-up work identified in a document."""

    task: str
    owner: str | None = None
    due: str | None = None


class SummaryOut(BaseModel):
    """The structured result of summarizing a document."""

    summary: str
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItemOut] = Field(default_factory=list)


class HealthOut(BaseModel):
    """Liveness response for the deployment platform's health check."""

    status: str = "ok"


class ErrorOut(BaseModel):
    """Error envelope. Matches FastAPI's default `{"detail": ...}` shape."""

    detail: str
