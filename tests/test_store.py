"""Tests for SQLite persistence."""

from datetime import datetime, timezone

import pytest

from assistant.errors import AssistantError
from assistant.models import Document, SpeakerTurn
from assistant.store import (
    get_document,
    list_document_ids,
    list_documents,
    save_document,
)


@pytest.fixture
def db(tmp_path):
    """A fresh database file per test."""
    return tmp_path / "test.db"


def make_doc(**overrides):
    payload = {
        "source_path": "./notes.txt",
        "doc_type": "text",
        "title": "Weekly Sync",
        "created_at": datetime(2026, 7, 29, 9, 30, tzinfo=timezone.utc),
        "raw_text": "hello world",
        "speaker_turns": None,
        "token_estimate": 3,
    }
    payload.update(overrides)
    return Document(**payload)


class TestSaveDocument:
    def test_returns_an_id(self, db):
        assert isinstance(save_document(make_doc(), db_path=db), int)

    def test_ids_increment(self, db):
        first = save_document(make_doc(), db_path=db)
        second = save_document(make_doc(), db_path=db)
        assert second > first

    def test_creates_the_database_file(self, db):
        save_document(make_doc(), db_path=db)
        assert db.exists()

    def test_schema_is_created_on_demand(self, db):
        # No explicit setup call — save_document applies the schema itself.
        assert save_document(make_doc(), db_path=db) == 1


class TestGetDocument:
    def test_round_trips_all_fields(self, db):
        original = make_doc()
        doc_id = save_document(original, db_path=db)

        fetched = get_document(doc_id, db_path=db)

        assert fetched.source_path == original.source_path
        assert fetched.doc_type == original.doc_type
        assert fetched.title == original.title
        assert fetched.raw_text == original.raw_text
        assert fetched.token_estimate == original.token_estimate

    def test_round_trips_created_at(self, db):
        original = make_doc()
        doc_id = save_document(original, db_path=db)

        assert get_document(doc_id, db_path=db).created_at == original.created_at

    def test_round_trips_null_created_at(self, db):
        doc_id = save_document(make_doc(created_at=None), db_path=db)
        assert get_document(doc_id, db_path=db).created_at is None

    def test_round_trips_null_title(self, db):
        doc_id = save_document(make_doc(title=None), db_path=db)
        assert get_document(doc_id, db_path=db).title is None

    def test_round_trips_speaker_turns(self, db):
        turns = [
            SpeakerTurn(speaker="Alex", timestamp="00:00:01", text="Morning."),
            SpeakerTurn(speaker="Priya", timestamp="00:00:09", text="Migration is up."),
        ]
        doc_id = save_document(
            make_doc(doc_type="transcript", speaker_turns=turns), db_path=db
        )

        fetched = get_document(doc_id, db_path=db)

        assert fetched.speaker_turns is not None
        assert len(fetched.speaker_turns) == 2
        assert fetched.speaker_turns[0].speaker == "Alex"
        assert fetched.speaker_turns[0].timestamp == "00:00:01"
        assert fetched.speaker_turns[1].text == "Migration is up."

    def test_round_trips_unattributed_turns(self, db):
        turns = [SpeakerTurn(text="Just narration.")]
        doc_id = save_document(
            make_doc(doc_type="transcript", speaker_turns=turns), db_path=db
        )

        fetched = get_document(doc_id, db_path=db)
        assert fetched.speaker_turns[0].speaker is None

    def test_null_speaker_turns_stay_null(self, db):
        doc_id = save_document(make_doc(speaker_turns=None), db_path=db)
        assert get_document(doc_id, db_path=db).speaker_turns is None

    def test_empty_turn_list_round_trips(self, db):
        doc_id = save_document(make_doc(speaker_turns=[]), db_path=db)
        assert get_document(doc_id, db_path=db).speaker_turns == []

    def test_returns_the_requested_document(self, db):
        first = save_document(make_doc(raw_text="first"), db_path=db)
        second = save_document(make_doc(raw_text="second"), db_path=db)

        assert get_document(first, db_path=db).raw_text == "first"
        assert get_document(second, db_path=db).raw_text == "second"

    def test_missing_id_raises(self, db):
        with pytest.raises(AssistantError, match="No document with id 999"):
            get_document(999, db_path=db)

    def test_preserves_multiline_text(self, db):
        text = "line one\n\nline two\nline three"
        doc_id = save_document(make_doc(raw_text=text), db_path=db)
        assert get_document(doc_id, db_path=db).raw_text == text

    def test_preserves_unicode(self, db):
        text = "café — naïve — 日本語 — 🎯"
        doc_id = save_document(make_doc(raw_text=text), db_path=db)
        assert get_document(doc_id, db_path=db).raw_text == text

    def test_preserves_sql_metacharacters(self, db):
        text = "Robert'); DROP TABLE documents;--"
        doc_id = save_document(make_doc(raw_text=text), db_path=db)
        assert get_document(doc_id, db_path=db).raw_text == text


class TestListDocuments:
    def test_empty_database_returns_empty_list(self, db):
        assert list_documents(db_path=db) == []

    def test_returns_all_documents(self, db):
        save_document(make_doc(raw_text="one"), db_path=db)
        save_document(make_doc(raw_text="two"), db_path=db)
        save_document(make_doc(raw_text="three"), db_path=db)

        assert len(list_documents(db_path=db)) == 3

    def test_ordered_by_id(self, db):
        save_document(make_doc(raw_text="first"), db_path=db)
        save_document(make_doc(raw_text="second"), db_path=db)

        texts = [d.raw_text for d in list_documents(db_path=db)]
        assert texts == ["first", "second"]

    def test_returns_document_models(self, db):
        save_document(make_doc(), db_path=db)
        assert all(isinstance(d, Document) for d in list_documents(db_path=db))

    def test_includes_speaker_turns(self, db):
        turns = [SpeakerTurn(speaker="Alex", text="Hi")]
        save_document(make_doc(doc_type="transcript", speaker_turns=turns), db_path=db)

        assert list_documents(db_path=db)[0].speaker_turns[0].speaker == "Alex"


class TestListDocumentIds:
    def test_empty_database(self, db):
        assert list_document_ids(db_path=db) == []

    def test_returns_id_and_title_pairs(self, db):
        doc_id = save_document(make_doc(title="Weekly Sync"), db_path=db)
        assert list_document_ids(db_path=db) == [(doc_id, "Weekly Sync")]

    def test_falls_back_to_source_path_when_untitled(self, db):
        doc_id = save_document(
            make_doc(title=None, source_path="./untitled.txt"), db_path=db
        )
        assert list_document_ids(db_path=db) == [(doc_id, "./untitled.txt")]


class TestPersistenceAcrossConnections:
    def test_data_survives_reconnect(self, db):
        doc_id = save_document(make_doc(raw_text="persisted"), db_path=db)

        # Each call opens and closes its own connection.
        assert get_document(doc_id, db_path=db).raw_text == "persisted"
        assert len(list_documents(db_path=db)) == 1
