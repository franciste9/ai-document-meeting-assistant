"""Regenerate docs/summarization-graph.md from the compiled graph.

    .venv/bin/python scripts/render_graph.py

Keeps the committed diagram in sync with the code rather than hand-maintained.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.orchestration import draw_mermaid  # noqa: E402

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "summarization-graph.md"

TEMPLATE = """\
# Summarization pipeline

Generated from the compiled `StateGraph` in [`assistant/orchestration.py`](../assistant/orchestration.py).
Regenerate with `python scripts/render_graph.py` — don't edit this file by hand.

```mermaid
{diagram}
```

## How it reads

The entry edge is conditional on `document.token_estimate <= threshold`:

| Path | When | Calls |
| ---- | ---- | ----- |
| `summarize_whole` | Document fits under the threshold | 1 |
| `chunk` → `summarize_chunks` → `merge` | Document exceeds it | 1 per chunk, plus 1 merge |

A document sitting exactly on the threshold takes the whole-document path —
the comparison is `<=`, matching the behavior this graph replaced.

`chunk` delegates to `chunk_document()` in `ingestion/normalize.py`, so chunks
break on speaker-turn boundaries for transcripts and paragraph boundaries
otherwise, with ~500 tokens of overlap.
"""


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(TEMPLATE.format(diagram=draw_mermaid().strip()), encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(Path.cwd())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
