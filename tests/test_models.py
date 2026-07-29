"""Tests for the pydantic models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from assistant.models import Document, IngestResult, SpeakerTurn


class TestSpeakerTurn:
    def test_text_is_required(self):
        with pytest.raises(ValidationError):
            SpeakerTurn()

    def test_speaker_and_timestamp_default_to_none(self):
        turn = SpeakerTurn(text="Just some words.")
        assert turn.speaker is None
        assert turn.timestamp is None

    def test_fully_attributed_turn(self):
        turn = SpeakerTurn(speaker="Alex", timestamp="00:12:03", text="Let's ship it.")
        assert turn.speaker == "Alex"
        assert turn.timestamp == "00:12:03"
        assert turn.text == "Let's ship it."


class TestDocument:
    def _minimal(self, **overrides):
        payload = {
            "source_path": "./notes.txt",
            "doc_type": "text",
            "raw_text": "hello",
            "token_estimate": 1,
        }
        payload.update(overrides)
        return payload

    def test_minimal_document(self):
        doc = Document(**self._minimal())
        assert doc.title is None
        assert doc.created_at is None
        assert doc.speaker_turns is None

    @pytest.mark.parametrize("doc_type", ["pdf", "docx", "transcript", "text"])
    def test_accepts_each_valid_doc_type(self, doc_type):
        assert Document(**self._minimal(doc_type=doc_type)).doc_type == doc_type

    def test_rejects_unknown_doc_type(self):
        with pytest.raises(ValidationError):
            Document(**self._minimal(doc_type="spreadsheet"))

    def test_rejects_negative_token_estimate(self):
        with pytest.raises(ValidationError):
            Document(**self._minimal(token_estimate=-1))

    def test_zero_token_estimate_is_allowed(self):
        assert Document(**self._minimal(raw_text="", token_estimate=0)).token_estimate == 0

    def test_carries_speaker_turns(self):
        doc = Document(
            **self._minimal(
                doc_type="transcript",
                speaker_turns=[SpeakerTurn(speaker="Alex", text="Hi")],
            )
        )
        assert doc.speaker_turns[0].speaker == "Alex"

    def test_created_at_accepts_datetime(self):
        stamp = datetime(2026, 7, 29, 9, 30)
        assert Document(**self._minimal(created_at=stamp)).created_at == stamp

    def test_created_at_parses_iso_string(self):
        doc = Document(**self._minimal(created_at="2026-07-29T09:30:00"))
        assert doc.created_at == datetime(2026, 7, 29, 9, 30)

    def test_missing_required_field_raises(self):
        payload = self._minimal()
        del payload["raw_text"]
        with pytest.raises(ValidationError):
            Document(**payload)


class TestIngestResult:
    def _doc(self):
        return Document(
            source_path="./notes.txt",
            doc_type="text",
            raw_text="hello",
            token_estimate=1,
        )

    def test_chunks_default_to_none(self):
        assert IngestResult(document=self._doc()).chunks is None

    def test_carries_chunks(self):
        result = IngestResult(document=self._doc(), chunks=["a", "b"])
        assert result.chunks == ["a", "b"]

    def test_document_is_required(self):
        with pytest.raises(ValidationError):
            IngestResult()
