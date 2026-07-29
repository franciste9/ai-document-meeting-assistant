"""Speaker/timestamp-aware transcript parsing.

Owns all transcript parsing: WebVTT (.vtt), SubRip (.srt), and plain
speaker-prefixed text. `loaders.load_transcript` reads the file and delegates
here.

Falls back to a single unattributed `SpeakerTurn` holding the whole document
when no speaker markers are found.
"""

from __future__ import annotations

import re

from assistant.models import SpeakerTurn

# "[00:12:03] Alex: ..." or "(00:12) Alex: ..." — bracketed timestamp, speaker, text.
_BRACKETED = re.compile(
    r"^[\[(](?P<timestamp>\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?)[\])]\s*"
    r"(?P<speaker>[^:\n]{1,64}?)\s*:\s*"
    r"(?P<text>.*)$"
)

# "00:12:03 Alex: ..." — bare leading timestamp, no brackets.
_BARE_TIMESTAMP = re.compile(
    r"^(?P<timestamp>\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?)\s+"
    r"(?P<speaker>[^:\n]{1,64}?)\s*:\s*"
    r"(?P<text>.*)$"
)

# "Alex: ..." — speaker only. Deliberately strict so prose like
# "Note: revenue grew" is not mistaken for an utterance: the speaker must look
# like a name (letters, digits, spaces, and a few name punctuation marks) and
# be at most four words.
_SPEAKER_ONLY = re.compile(
    r"^(?P<speaker>[A-Za-z0-9][\w.''\- ]{0,48})\s*:\s+(?P<text>\S.*)$"
)

# Cue-timing line in .vtt/.srt: "00:00:01.000 --> 00:00:04.000".
_CUE_TIMING = re.compile(
    r"^(?P<start>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})"
)

# "<v Alex>Let's ship it." — WebVTT voice span.
_VOICE_SPAN = re.compile(r"^<v\s+(?P<speaker>[^>]+)>(?P<text>.*?)(?:</v>)?$")

# Inline tags to strip from cue text, e.g. "<b>", "<00:00:02.000>".
_INLINE_TAG = re.compile(r"</?[^>]+>")

# Words that commonly begin a prose line ending in a colon; these should not be
# read as speaker names.
_PROSE_LEAD_INS = frozenset(
    {
        "note",
        "warning",
        "caution",
        "example",
        "summary",
        "todo",
        "fixme",
        "http",
        "https",
        "see",
        "source",
        "reference",
        "agenda",
        "attendees",
        "date",
        "time",
        "location",
        "subject",
        "from",
        "to",
        "cc",
        "re",
    }
)

# A speaker name should be short — "Alex", "Alex Chen", "Dr. Priya Raman".
_MAX_SPEAKER_WORDS = 4


def _is_plausible_speaker(name: str) -> bool:
    """Reject prose lead-ins and over-long fragments that merely contain a colon."""
    cleaned = name.strip()
    if not cleaned:
        return False
    if cleaned.lower() in _PROSE_LEAD_INS:
        return False
    if len(cleaned.split()) > _MAX_SPEAKER_WORDS:
        return False
    # A sentence fragment ending in punctuation is not a name.
    return not cleaned.endswith((".", "!", "?", ",", ";"))


def _match_speaker_line(line: str) -> re.Match | None:
    """Return the first pattern that matches `line` as an attributed utterance."""
    for pattern in (_BRACKETED, _BARE_TIMESTAMP, _SPEAKER_ONLY):
        match = pattern.match(line.strip())
        if match and _is_plausible_speaker(match.group("speaker")):
            return match
    return None


def has_speaker_markers(text: str) -> bool:
    """True if `text` contains speaker/timestamp markers or subtitle cue timings.

    Used by `loaders.detect_format` to tell a plain-text transcript from prose.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _CUE_TIMING.match(stripped) or _VOICE_SPAN.match(stripped):
            return True
        if _match_speaker_line(stripped):
            return True
    return False


def _normalize_cue_time(value: str) -> str:
    """Trim milliseconds from a cue timestamp: "00:00:01.000" -> "00:00:01"."""
    return re.split(r"[.,]", value, maxsplit=1)[0]


def _clean_cue_text(line: str) -> str:
    return _INLINE_TAG.sub("", line).strip()


def _parse_cue_blocks(raw: str) -> list[SpeakerTurn]:
    """Parse .vtt/.srt cue blocks into speaker turns.

    Speaker attribution comes from a WebVTT `<v Name>` span or a "Name:" prefix
    inside the cue text; otherwise the turn is unattributed.
    """
    turns: list[SpeakerTurn] = []
    timestamp: str | None = None
    pending: list[str] = []
    speaker: str | None = None

    def flush() -> None:
        nonlocal pending, speaker, timestamp
        text = " ".join(part for part in pending if part).strip()
        if text:
            turns.append(SpeakerTurn(speaker=speaker, timestamp=timestamp, text=text))
        pending = []
        speaker = None

    for line in raw.splitlines():
        stripped = line.strip()

        if not stripped:
            flush()
            continue

        # WEBVTT header, NOTE blocks, and bare numeric SRT sequence numbers.
        if stripped.upper().startswith("WEBVTT") or stripped.startswith("NOTE"):
            continue
        if stripped.isdigit() and not pending:
            continue

        cue = _CUE_TIMING.match(stripped)
        if cue:
            flush()
            timestamp = _normalize_cue_time(cue.group("start"))
            continue

        voice = _VOICE_SPAN.match(stripped)
        if voice:
            if pending:
                flush()
            speaker = voice.group("speaker").strip()
            pending.append(_clean_cue_text(voice.group("text")))
            continue

        text = _clean_cue_text(stripped)
        if not text:
            continue

        # "Alex: ..." inside a cue attributes the whole cue.
        if speaker is None and not pending:
            match = _SPEAKER_ONLY.match(text)
            if match and _is_plausible_speaker(match.group("speaker")):
                speaker = match.group("speaker").strip()
                pending.append(match.group("text").strip())
                continue

        pending.append(text)

    flush()
    return turns


def _parse_speaker_prefixed(raw: str) -> list[SpeakerTurn]:
    """Parse plain speaker-prefixed text, e.g. "[00:12:03] Alex: ...".

    Continuation lines are appended to the preceding turn.
    """
    turns: list[SpeakerTurn] = []

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        match = _match_speaker_line(stripped)
        if match:
            groups = match.groupdict()
            turns.append(
                SpeakerTurn(
                    speaker=groups["speaker"].strip(),
                    timestamp=groups.get("timestamp"),
                    text=groups["text"].strip(),
                )
            )
        elif turns:
            turns[-1].text = f"{turns[-1].text} {stripped}".strip()
        else:
            # Preamble before the first attributed line.
            turns.append(SpeakerTurn(speaker=None, timestamp=None, text=stripped))

    return turns


def _unattributed(raw: str) -> list[SpeakerTurn]:
    """Fallback: the whole document as one unattributed turn."""
    text = raw.strip()
    return [SpeakerTurn(speaker=None, timestamp=None, text=text)] if text else []


def parse_transcript(raw: str, suffix: str = "") -> list[SpeakerTurn]:
    """Parse transcript text into speaker turns.

    `suffix` is the source file extension (".vtt", ".srt", ...) and selects the
    cue-block parser. Anything else is treated as speaker-prefixed text, with a
    single-unattributed-turn fallback when no markers are found.
    """
    if not raw.strip():
        return []

    if suffix.lower() in (".vtt", ".srt"):
        turns = _parse_cue_blocks(raw)
        return turns or _unattributed(raw)

    if has_speaker_markers(raw):
        turns = _parse_speaker_prefixed(raw)
        if turns:
            return turns

    return _unattributed(raw)
