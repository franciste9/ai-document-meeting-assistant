"""System prompt templates for meeting/document summarization.

The system prompt is kept byte-stable so it caches cleanly across calls; the
document itself goes in a separate block that also carries a cache breakpoint.
"""

from __future__ import annotations

from assistant.models import Document

# Frozen — no timestamps, no per-request interpolation, or the cache breaks.
SUMMARY_SYSTEM_PROMPT = """\
You are a meeting and document analyst. You read transcripts and documents and \
extract what actually happened: the substance, the decisions reached, and the \
work assigned.

Return a single JSON object with exactly these keys:

{
  "summary": "A prose summary of the document, 2-5 sentences.",
  "decisions": ["Each decision that was actually reached, one per entry."],
  "action_items": [
    {
      "task": "What needs to be done.",
      "owner": "Who owns it, or null if unassigned.",
      "due": "When it is due, or null if unstated."
    }
  ]
}

Rules:
- Output only the JSON object. No preamble, no code fences, no commentary.
- Record only what the document supports. Do not infer owners or deadlines \
that were not stated; use null.
- A decision is a conclusion the participants settled on, not a topic they \
discussed. If nothing was decided, return an empty list.
- An action item is work someone is expected to do after this meeting.
- Prefer the speaker names as written in the transcript.\
"""


def build_summary_messages(document: Document) -> list[dict]:
    """The user turn: the document content to analyze."""
    return [
        {
            "role": "user",
            "content": (
                f"Analyze the following document.\n\n"
                f"<document title=\"{document.title or 'Untitled'}\" "
                f"type=\"{document.doc_type}\">\n"
                f"{document.raw_text}\n"
                f"</document>"
            ),
        }
    ]


def build_chunk_messages(chunk: str, index: int, total: int) -> list[dict]:
    """The user turn for one chunk of an over-threshold document."""
    return [
        {
            "role": "user",
            "content": (
                f"Analyze part {index} of {total} of a longer document. "
                f"Report only what this part supports.\n\n"
                f"<document_part index=\"{index}\" of=\"{total}\">\n"
                f"{chunk}\n"
                f"</document_part>"
            ),
        }
    ]


MERGE_SYSTEM_PROMPT = """\
You are merging partial analyses of one document that was processed in parts.

You will receive several JSON objects, each covering one part. Combine them \
into a single JSON object with the same shape:

{
  "summary": "A unified prose summary of the whole document, 2-5 sentences.",
  "decisions": ["..."],
  "action_items": [{"task": "...", "owner": "...", "due": "..."}]
}

Rules:
- Output only the JSON object. No preamble, no code fences, no commentary.
- Merge duplicate decisions and action items that refer to the same thing.
- Preserve every distinct decision and action item.
- Do not introduce anything absent from the parts.\
"""


def build_merge_messages(partials: list[str]) -> list[dict]:
    """The user turn for merging per-chunk analyses into one result."""
    joined = "\n\n".join(
        f"<part index=\"{i + 1}\">\n{partial}\n</part>"
        for i, partial in enumerate(partials)
    )
    return [
        {
            "role": "user",
            "content": f"Merge these partial analyses into one result.\n\n{joined}",
        }
    ]
