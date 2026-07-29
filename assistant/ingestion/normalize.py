"""Raw text or speaker turns -> a validated `Document`.

Also owns the chunking rule: documents over the configured token threshold are
split on speaker-turn or paragraph boundaries with overlap; documents under it
are passed whole (no retrieval / vector store in this pass).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from assistant.models import Document, DocType, SpeakerTurn

# Rough heuristic from the spec: ~4 characters per token.
_CHARS_PER_TOKEN = 4

# Chunk overlap, in tokens, carried between consecutive chunks.
_OVERLAP_TOKENS = 500

# Collapse 3+ blank lines down to a paragraph break.
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

# Runs of spaces/tabs (not newlines).
_HORIZONTAL_RUNS = re.compile(r"[ \t]+")

# Trailing whitespace at end of each line.
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)


def estimate_tokens(text: str) -> int:
    """Rough token count: `len(text) / 4`, per the spec."""
    return len(text) // _CHARS_PER_TOKEN


def clean_text(raw: str) -> str:
    """Normalize line endings and strip excessive whitespace."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _HORIZONTAL_RUNS.sub(" ", text)
    text = _TRAILING_SPACE.sub("", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _render_turn(turn: SpeakerTurn) -> str:
    """Render a speaker turn back to a canonical line of text."""
    prefix = ""
    if turn.timestamp:
        prefix += f"[{turn.timestamp}] "
    if turn.speaker:
        prefix += f"{turn.speaker}: "
    return f"{prefix}{turn.text}".strip()


def turns_to_text(turns: list[SpeakerTurn]) -> str:
    """Flatten speaker turns into the document's `raw_text`."""
    return "\n".join(_render_turn(turn) for turn in turns).strip()


def _derive_title(text: str, source_path: str, doc_type: DocType) -> str | None:
    """Use the first non-empty line if it reads like a title, else the filename.

    Transcripts skip the first-line heuristic entirely: the opening line is
    someone's first utterance, not a title.
    """
    if doc_type != "transcript":
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                # A short leading line is a plausible title; anything longer is
                # body text, so fall back to the filename.
                if len(stripped) <= 120:
                    return stripped
                break

    stem = Path(source_path).stem
    return stem or None


def normalize(
    raw_text_or_turns: str | list[SpeakerTurn],
    source_path: str,
    doc_type: DocType,
    *,
    title: str | None = None,
    created_at: datetime | None = None,
) -> Document:
    """Build a `Document` from raw text or parsed speaker turns.

    Estimates the token count, strips excessive whitespace, and normalizes line
    endings. `created_at` defaults to the source file's mtime when available.
    """
    if isinstance(raw_text_or_turns, str):
        turns: list[SpeakerTurn] | None = None
        text = clean_text(raw_text_or_turns)
    else:
        turns = [
            SpeakerTurn(
                speaker=turn.speaker,
                timestamp=turn.timestamp,
                text=clean_text(turn.text),
            )
            for turn in raw_text_or_turns
        ]
        text = turns_to_text(turns)

    if created_at is None:
        created_at = _source_mtime(source_path)

    return Document(
        source_path=str(source_path),
        doc_type=doc_type,
        title=(
            title
            if title is not None
            else _derive_title(text, str(source_path), doc_type)
        ),
        created_at=created_at,
        raw_text=text,
        speaker_turns=turns,
        token_estimate=estimate_tokens(text),
    )


def _source_mtime(source_path: str) -> datetime | None:
    """The source file's modification time, or None if it isn't on disk."""
    try:
        stat = Path(source_path).stat()
    except OSError:
        return None
    return datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)


def _split_units(document: Document) -> list[str]:
    """The atomic units a document may be chunked on.

    Speaker turns when present, paragraphs otherwise.
    """
    if document.speaker_turns:
        return [_render_turn(turn) for turn in document.speaker_turns]

    paragraphs = [p.strip() for p in document.raw_text.split("\n\n")]
    return [p for p in paragraphs if p]


def chunk_document(
    document: Document,
    max_tokens: int,
    overlap_tokens: int = _OVERLAP_TOKENS,
) -> list[str]:
    """Split a document on speaker-turn or paragraph boundaries.

    Consecutive chunks overlap by roughly `overlap_tokens` tokens so context
    isn't lost at the seams.
    """
    units = _split_units(document)
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)

        # A single oversized unit becomes its own chunk rather than being split
        # mid-sentence.
        if current and current_tokens + unit_tokens > max_tokens:
            chunks.append("\n".join(current))
            current, current_tokens = _carry_overlap(current, overlap_tokens)

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append("\n".join(current))

    return chunks


def _carry_overlap(units: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Take trailing units from the finished chunk to seed the next one."""
    if overlap_tokens <= 0:
        return [], 0

    carried: list[str] = []
    carried_tokens = 0

    for unit in reversed(units):
        unit_tokens = estimate_tokens(unit)
        if carried and carried_tokens + unit_tokens > overlap_tokens:
            break
        carried.insert(0, unit)
        carried_tokens += unit_tokens

    return carried, carried_tokens
