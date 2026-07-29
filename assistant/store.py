"""SQLite persistence for ingested documents.

Uses stdlib `sqlite3` — no external DB in this pass. Speaker turns are stored
as JSON alongside the document so transcripts survive a round trip.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from assistant import config
from assistant.errors import AssistantError
from assistant.models import Document, SpeakerTurn

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path    TEXT    NOT NULL,
    doc_type       TEXT    NOT NULL,
    title          TEXT,
    created_at     TEXT,
    raw_text       TEXT    NOT NULL,
    token_estimate INTEGER NOT NULL,
    speaker_turns  TEXT
);
"""


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection with the schema applied and row access by name."""
    path = str(db_path or config.DB_PATH)
    try:
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        connection.execute(_SCHEMA)
        connection.commit()
    except sqlite3.Error as exc:
        raise AssistantError(f"Could not open database {path}: {exc}") from exc
    return connection


def _serialize_turns(turns: list[SpeakerTurn] | None) -> str | None:
    if turns is None:
        return None
    return json.dumps([turn.model_dump() for turn in turns])


def _deserialize_turns(blob: str | None) -> list[SpeakerTurn] | None:
    if blob is None:
        return None
    return [SpeakerTurn(**item) for item in json.loads(blob)]


def _row_to_document(row: sqlite3.Row) -> Document:
    created_at = row["created_at"]
    return Document(
        source_path=row["source_path"],
        doc_type=row["doc_type"],
        title=row["title"],
        created_at=datetime.fromisoformat(created_at) if created_at else None,
        raw_text=row["raw_text"],
        speaker_turns=_deserialize_turns(row["speaker_turns"]),
        token_estimate=row["token_estimate"],
    )


def save_document(doc: Document, db_path: str | Path | None = None) -> int:
    """Persist a document and return its new id."""
    connection = _connect(db_path)
    try:
        cursor = connection.execute(
            """
            INSERT INTO documents (
                source_path, doc_type, title, created_at,
                raw_text, token_estimate, speaker_turns
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.source_path,
                doc.doc_type,
                doc.title,
                doc.created_at.isoformat() if doc.created_at else None,
                doc.raw_text,
                doc.token_estimate,
                _serialize_turns(doc.speaker_turns),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)
    except sqlite3.Error as exc:
        raise AssistantError(f"Could not save document: {exc}") from exc
    finally:
        connection.close()


def get_document(doc_id: int, db_path: str | Path | None = None) -> Document:
    """Fetch a document by id, or raise if it doesn't exist."""
    connection = _connect(db_path)
    try:
        row = connection.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
    except sqlite3.Error as exc:
        raise AssistantError(f"Could not read document {doc_id}: {exc}") from exc
    finally:
        connection.close()

    if row is None:
        raise AssistantError(f"No document with id {doc_id}")

    return _row_to_document(row)


def list_documents(db_path: str | Path | None = None) -> list[Document]:
    """All stored documents, oldest first."""
    connection = _connect(db_path)
    try:
        rows = connection.execute("SELECT * FROM documents ORDER BY id").fetchall()
    except sqlite3.Error as exc:
        raise AssistantError(f"Could not list documents: {exc}") from exc
    finally:
        connection.close()

    return [_row_to_document(row) for row in rows]


def list_document_ids(db_path: str | Path | None = None) -> list[tuple[int, str]]:
    """(id, title-or-source) pairs, for CLI listings."""
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            "SELECT id, title, source_path FROM documents ORDER BY id"
        ).fetchall()
    except sqlite3.Error as exc:
        raise AssistantError(f"Could not list documents: {exc}") from exc
    finally:
        connection.close()

    return [(row["id"], row["title"] or row["source_path"]) for row in rows]
