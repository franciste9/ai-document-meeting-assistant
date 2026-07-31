"""CLI entrypoint.

    python -m assistant.main ingest ./meeting_notes.pdf
    python -m assistant.main summarize --id 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from assistant import config, orchestration, store
from assistant.errors import AssistantError
from assistant.ingestion import loaders
from assistant.ingestion.normalize import chunk_document, normalize
from assistant.models import Document, IngestResult

_LOADERS = {
    "pdf": loaders.load_pdf,
    "docx": loaders.load_docx,
    "transcript": loaders.load_transcript,
    "text": loaders.load_text,
}


def ingest_file(path: str | Path, db_path: str | Path | None = None) -> tuple[int, IngestResult]:
    """detect -> load -> normalize -> save. Returns the new id and the result."""
    source = Path(path)
    if not source.exists():
        raise AssistantError(f"No such file: {source}")

    doc_type = loaders.detect_format(source)
    loaded = _LOADERS[doc_type](source)
    document = normalize(loaded, str(source), doc_type)

    threshold = config.get_chunk_token_threshold()
    chunks = (
        chunk_document(document, max_tokens=threshold)
        if document.token_estimate > threshold
        else None
    )

    doc_id = store.save_document(document, db_path=db_path)
    return doc_id, IngestResult(document=document, chunks=chunks)


def summarize_document(
    document: Document,
    client=None,
    threshold: int | None = None,
) -> str:
    """Build the prompt, call Claude, and return the structured JSON result.

    Documents over the token threshold are analyzed chunk-by-chunk and merged;
    documents under it are sent whole. The branching itself lives in
    `orchestration.py` as an explicit graph; this is the stable entrypoint the
    CLI and the HTTP routes both call.
    """
    return orchestration.summarize_via_graph(
        document, client=client, threshold=threshold
    )


def summarize_document_stream(
    document: Document,
    client=None,
    threshold: int | None = None,
) -> Iterator[str]:
    """Stream a summary incrementally instead of returning it all at once.

    Same pairing as `summarize_document` → `summarize_via_graph`: this is the
    stable entrypoint, the pipeline lives in `orchestration.py`.
    """
    return orchestration.summarize_via_graph_stream(
        document, client=client, threshold=threshold
    )


def _pretty_print_result(raw: str) -> None:
    """Print the model's JSON, re-indented when it parses cleanly."""
    try:
        print(json.dumps(json.loads(raw), indent=2))
    except json.JSONDecodeError:
        # Print what came back rather than discarding it.
        print(raw)


def _cmd_ingest(args: argparse.Namespace) -> int:
    doc_id, result = ingest_file(args.path, db_path=args.db)
    document = result.document

    print(f"Ingested [{doc_id}] {document.title or document.source_path}")
    print(f"  type:           {document.doc_type}")
    print(f"  token estimate: {document.token_estimate:,}")

    if document.speaker_turns is not None:
        print(f"  speaker turns:  {len(document.speaker_turns)}")

    if result.chunks:
        print(f"  chunks:         {len(result.chunks)} (over threshold)")

    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    document = store.get_document(args.id, db_path=args.db)

    if args.stream:
        # Raw text, not pretty-printed: there is no complete JSON to reformat
        # until the stream ends, and buffering to reformat would defeat the
        # point of --stream.
        for chunk in summarize_document_stream(document):
            print(chunk, end="", flush=True)
        print()
        return 0

    _pretty_print_result(summarize_document(document))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    rows = store.list_document_ids(db_path=args.db)
    if not rows:
        print("No documents ingested yet.")
        return 0

    for doc_id, label in rows:
        print(f"[{doc_id}] {label}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m assistant.main",
        description="Ingest documents and meeting transcripts, and summarize them.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"SQLite database path (default: {config.DB_PATH})",
    )

    subcommands = parser.add_subparsers(dest="command", required=True)

    ingest = subcommands.add_parser("ingest", help="Ingest a document into the store.")
    ingest.add_argument("path", help="Path to a .pdf, .docx, .txt, .md, .vtt or .srt file")
    ingest.set_defaults(func=_cmd_ingest)

    summarize = subcommands.add_parser(
        "summarize", help="Summarize a stored document by id."
    )
    summarize.add_argument("--id", type=int, required=True, help="Document id")
    summarize.add_argument(
        "--stream",
        action="store_true",
        help="Print the summary incrementally as it arrives (raw text, not JSON)",
    )
    summarize.set_defaults(func=_cmd_summarize)

    listing = subcommands.add_parser("list", help="List stored documents.")
    listing.set_defaults(func=_cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except AssistantError as exc:
        # The CLI boundary: errors surface as a message, not a traceback.
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
