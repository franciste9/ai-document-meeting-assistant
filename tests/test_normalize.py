"""Tests for normalization, token estimation, and chunking."""

from datetime import datetime, timezone

import pytest

from assistant.ingestion.normalize import (
    chunk_document,
    clean_text,
    estimate_tokens,
    normalize,
    turns_to_text,
)
from assistant.models import Document, SpeakerTurn


class TestEstimateTokens:
    def test_four_chars_per_token(self):
        assert estimate_tokens("a" * 400) == 100

    def test_empty_text_is_zero(self):
        assert estimate_tokens("") == 0

    def test_scales_linearly(self):
        assert estimate_tokens("a" * 800) == 2 * estimate_tokens("a" * 400)


class TestCleanText:
    def test_normalizes_crlf(self):
        assert "\r" not in clean_text("a\r\nb\r\nc")

    def test_normalizes_lone_cr(self):
        assert clean_text("a\rb") == "a\nb"

    def test_collapses_horizontal_whitespace(self):
        assert clean_text("a      b\tc") == "a b c"

    def test_collapses_excess_blank_lines(self):
        assert clean_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_preserves_paragraph_breaks(self):
        assert clean_text("a\n\nb") == "a\n\nb"

    def test_strips_trailing_line_whitespace(self):
        assert clean_text("a   \nb   ") == "a\nb"

    def test_strips_leading_and_trailing(self):
        assert clean_text("\n\n  hello  \n\n") == "hello"

    def test_empty_stays_empty(self):
        assert clean_text("") == ""

    def test_whitespace_only_becomes_empty(self):
        assert clean_text("  \n\n \t ") == ""


class TestTurnsToText:
    def test_renders_timestamp_and_speaker(self):
        turns = [SpeakerTurn(speaker="Alex", timestamp="00:00:01", text="Hello.")]
        assert turns_to_text(turns) == "[00:00:01] Alex: Hello."

    def test_renders_speaker_without_timestamp(self):
        assert turns_to_text([SpeakerTurn(speaker="Alex", text="Hi.")]) == "Alex: Hi."

    def test_renders_unattributed_turn_as_bare_text(self):
        assert turns_to_text([SpeakerTurn(text="Just narration.")]) == "Just narration."

    def test_joins_turns_with_newlines(self):
        turns = [SpeakerTurn(speaker="A", text="one"), SpeakerTurn(speaker="B", text="two")]
        assert turns_to_text(turns) == "A: one\nB: two"


class TestNormalizeFromText:
    def test_builds_document(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("hello world", encoding="utf-8")

        doc = normalize("hello world", str(path), "text")

        assert isinstance(doc, Document)
        assert doc.source_path == str(path)
        assert doc.doc_type == "text"
        assert doc.raw_text == "hello world"

    def test_speaker_turns_is_none_for_plain_text(self):
        assert normalize("hello", "./a.txt", "text").speaker_turns is None

    def test_token_estimate_matches_cleaned_text(self):
        doc = normalize("a" * 400, "./a.txt", "text")
        assert doc.token_estimate == 100

    def test_cleans_whitespace(self):
        doc = normalize("a\r\n\n\n\n\nb   ", "./a.txt", "text")
        assert doc.raw_text == "a\n\nb"

    def test_explicit_title_wins(self):
        doc = normalize("body text", "./a.txt", "text", title="My Title")
        assert doc.title == "My Title"

    def test_derives_title_from_short_first_line(self):
        doc = normalize("Weekly Sync\n\nbody text here", "./a.txt", "text")
        assert doc.title == "Weekly Sync"

    def test_falls_back_to_filename_for_long_first_line(self):
        doc = normalize("x" * 200, "./quarterly-review.txt", "text")
        assert doc.title == "quarterly-review"

    def test_transcript_title_comes_from_filename_not_first_utterance(self):
        """A speaker's opening line is not a title."""
        turns = [
            SpeakerTurn(speaker="Alex", timestamp="00:00:04", text="Morning all."),
            SpeakerTurn(speaker="Priya", timestamp="00:00:11", text="Deployed."),
        ]
        doc = normalize(turns, "./standup.txt", "transcript")

        assert doc.title == "standup"

    def test_explicit_title_still_wins_for_transcripts(self):
        turns = [SpeakerTurn(speaker="Alex", text="Morning all.")]
        doc = normalize(turns, "./standup.txt", "transcript", title="Daily Standup")

        assert doc.title == "Daily Standup"

    def test_explicit_created_at_wins(self):
        stamp = datetime(2026, 7, 29, 9, 30, tzinfo=timezone.utc)
        doc = normalize("body", "./a.txt", "text", created_at=stamp)
        assert doc.created_at == stamp

    def test_created_at_from_file_mtime(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_text("body", encoding="utf-8")

        doc = normalize("body", str(path), "text")

        assert doc.created_at is not None

    def test_created_at_is_none_for_missing_file(self):
        assert normalize("body", "./does-not-exist.txt", "text").created_at is None

    def test_empty_text_yields_zero_tokens(self):
        doc = normalize("", "./a.txt", "text")
        assert doc.raw_text == ""
        assert doc.token_estimate == 0

    @pytest.mark.parametrize("doc_type", ["pdf", "docx", "transcript", "text"])
    def test_carries_doc_type(self, doc_type):
        assert normalize("body", "./a.txt", doc_type).doc_type == doc_type


class TestNormalizeFromTurns:
    def _turns(self):
        return [
            SpeakerTurn(speaker="Alex", timestamp="00:00:01", text="Morning."),
            SpeakerTurn(speaker="Priya", timestamp="00:00:09", text="Migration is up."),
        ]

    def test_preserves_turns(self):
        doc = normalize(self._turns(), "./m.txt", "transcript")

        assert doc.speaker_turns is not None
        assert len(doc.speaker_turns) == 2
        assert doc.speaker_turns[0].speaker == "Alex"

    def test_raw_text_is_flattened_turns(self):
        doc = normalize(self._turns(), "./m.txt", "transcript")

        assert "[00:00:01] Alex: Morning." in doc.raw_text
        assert "[00:00:09] Priya: Migration is up." in doc.raw_text

    def test_cleans_whitespace_inside_turn_text(self):
        turns = [SpeakerTurn(speaker="Alex", text="too    many     spaces")]
        doc = normalize(turns, "./m.txt", "transcript")

        assert doc.speaker_turns[0].text == "too many spaces"

    def test_token_estimate_reflects_flattened_text(self):
        doc = normalize(self._turns(), "./m.txt", "transcript")
        assert doc.token_estimate == len(doc.raw_text) // 4

    def test_empty_turn_list(self):
        doc = normalize([], "./m.txt", "transcript")

        assert doc.raw_text == ""
        assert doc.speaker_turns == []


class TestChunkDocument:
    def _doc(self, text, turns=None):
        return normalize(turns if turns is not None else text, "./a.txt", "text")

    def test_short_document_is_one_chunk(self):
        chunks = chunk_document(self._doc("para one\n\npara two"), max_tokens=1000)
        assert len(chunks) == 1

    def test_splits_on_paragraph_boundaries(self):
        # Each paragraph ~25 tokens; a 30-token cap forces one para per chunk.
        text = "\n\n".join("x" * 100 for _ in range(4))
        chunks = chunk_document(self._doc(text), max_tokens=30, overlap_tokens=0)

        assert len(chunks) == 4

    def test_never_splits_mid_paragraph(self):
        text = "\n\n".join(f"para{i} " + "x" * 100 for i in range(4))
        chunks = chunk_document(self._doc(text), max_tokens=30, overlap_tokens=0)

        for i in range(4):
            assert sum(f"para{i}" in c for c in chunks) == 1

    def test_splits_on_speaker_turn_boundaries(self):
        turns = [SpeakerTurn(speaker=f"S{i}", text="x" * 100) for i in range(4)]
        doc = normalize(turns, "./m.txt", "transcript")

        chunks = chunk_document(doc, max_tokens=30, overlap_tokens=0)

        assert len(chunks) == 4
        for i in range(4):
            assert sum(f"S{i}:" in c for c in chunks) == 1

    def test_overlap_repeats_trailing_content(self):
        text = "\n\n".join(f"para{i} " + "x" * 100 for i in range(4))
        doc = self._doc(text)

        with_overlap = chunk_document(doc, max_tokens=60, overlap_tokens=30)
        without_overlap = chunk_document(doc, max_tokens=60, overlap_tokens=0)

        joined_with = "".join(with_overlap)
        assert len(joined_with) > len("".join(without_overlap))

    def test_overlap_content_is_shared_between_chunks(self):
        text = "\n\n".join(f"para{i} " + "x" * 100 for i in range(4))
        chunks = chunk_document(self._doc(text), max_tokens=60, overlap_tokens=30)

        # Some paragraph appears in two consecutive chunks.
        assert any(
            sum(f"para{i}" in c for c in chunks) > 1 for i in range(4)
        )

    def test_zero_overlap_shares_nothing(self):
        text = "\n\n".join(f"para{i} " + "x" * 100 for i in range(4))
        chunks = chunk_document(self._doc(text), max_tokens=60, overlap_tokens=0)

        for i in range(4):
            assert sum(f"para{i}" in c for c in chunks) == 1

    def test_oversized_unit_becomes_its_own_chunk(self):
        doc = self._doc("x" * 4000)
        chunks = chunk_document(doc, max_tokens=100, overlap_tokens=0)

        assert len(chunks) == 1

    def test_empty_document_yields_no_chunks(self):
        assert chunk_document(self._doc(""), max_tokens=100) == []

    def test_chunks_cover_all_content(self):
        text = "\n\n".join(f"para{i}" for i in range(10))
        chunks = chunk_document(self._doc(text), max_tokens=5, overlap_tokens=0)

        joined = "\n".join(chunks)
        for i in range(10):
            assert f"para{i}" in joined
