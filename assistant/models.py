"""Pydantic models for the Document/Meeting Assistant.

These are the validation boundary for ingestion: loaders produce raw text or
speaker turns, `normalize` turns them into a `Document`, and everything
downstream (store, client, CLI) works against these types.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

DocType = Literal["pdf", "docx", "transcript", "text"]


class SpeakerTurn(BaseModel):
    """A single attributed utterance in a meeting transcript.

    `speaker` and `timestamp` are both optional: transcripts without speaker
    markers fall back to one unattributed turn holding the whole document.
    """

    speaker: str | None = None
    timestamp: str | None = None
    text: str


class Document(BaseModel):
    """A normalized document ready for persistence or for sending to Claude."""

    source_path: str
    doc_type: DocType
    title: str | None = None
    created_at: datetime | None = None
    raw_text: str
    speaker_turns: list[SpeakerTurn] | None = None
    token_estimate: int = Field(ge=0)


class IngestResult(BaseModel):
    """The outcome of ingesting one file.

    `chunks` is populated only when the document exceeds the configured token
    threshold; otherwise the whole document is passed in context.
    """

    document: Document
    chunks: list[str] | None = None
