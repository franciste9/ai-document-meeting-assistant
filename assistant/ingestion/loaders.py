"""Per-format loaders: file on disk -> raw text (or speaker turns).

Each loader is I/O only. Transcript *parsing* lives in `transcripts.py`;
`load_transcript` reads the file and delegates.
"""

from __future__ import annotations

from pathlib import Path

from assistant.errors import AssistantError
from assistant.models import SpeakerTurn

# How many lines of a .txt file to inspect when deciding whether it is a
# speaker-prefixed transcript rather than prose.
_SNIFF_LINES = 40

_EXTENSION_MAP = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".vtt": "transcript",
    ".srt": "transcript",
    ".txt": "text",
    ".md": "text",
}


def _read_text_file(path: str | Path) -> str:
    """Read a UTF-8 text file, tolerating undecodable bytes."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise AssistantError(f"Could not read {path}: {exc}") from exc


def detect_format(path: str | Path) -> str:
    """Dispatch by extension, sniffing `.txt` content to separate prose from
    speaker-prefixed transcripts.

    Returns one of: "pdf", "docx", "transcript", "text".
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext not in _EXTENSION_MAP:
        raise AssistantError(
            f"Unsupported file type '{ext or p.name}'. "
            f"Supported: {', '.join(sorted(_EXTENSION_MAP))}"
        )

    fmt = _EXTENSION_MAP[ext]

    # A .txt/.md file holding a meeting transcript should be parsed as one.
    if fmt == "text" and _looks_like_transcript(p):
        return "transcript"

    return fmt


def _looks_like_transcript(path: Path) -> bool:
    """True if the first `_SNIFF_LINES` lines carry speaker/timestamp markers."""
    # Imported here to keep the module import graph acyclic at load time.
    from assistant.ingestion.transcripts import has_speaker_markers

    try:
        with Path(path).open(encoding="utf-8", errors="replace") as handle:
            head = "".join(next(handle, "") for _ in range(_SNIFF_LINES))
    except OSError as exc:
        raise AssistantError(f"Could not read {path}: {exc}") from exc

    return has_speaker_markers(head)


def load_pdf(path: str | Path) -> str:
    """Extract text from a PDF, one page per block."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise AssistantError("pypdf is required to read PDF files") from exc

    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except AssistantError:
        raise
    except Exception as exc:
        raise AssistantError(f"Could not parse PDF {path}: {exc}") from exc

    return "\n\n".join(pages)


def load_docx(path: str | Path) -> str:
    """Extract text from a Word document, one paragraph per line."""
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise AssistantError("python-docx is required to read .docx files") from exc

    try:
        document = docx.Document(str(path))
        paragraphs = [p.text for p in document.paragraphs]
    except AssistantError:
        raise
    except Exception as exc:
        raise AssistantError(f"Could not parse DOCX {path}: {exc}") from exc

    return "\n".join(paragraphs)


def load_text(path: str | Path) -> str:
    """Read a plain-text file verbatim."""
    return _read_text_file(path)


def load_transcript(path: str | Path) -> list[SpeakerTurn]:
    """Read a transcript and delegate parsing to `transcripts.parse_transcript`.

    Handles .vtt, .srt, and plain speaker-prefixed text. Falls back to a single
    unattributed turn when no speaker markers are present.
    """
    from assistant.ingestion.transcripts import parse_transcript

    raw = _read_text_file(path)
    return parse_transcript(raw, suffix=Path(path).suffix.lower())
