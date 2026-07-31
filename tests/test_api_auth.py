"""Tests for the bearer-token gate and the streaming HTTP route.

Network-free — the summarize dependency is overridden with a fake.
"""

import io

import pytest
from fastapi.testclient import TestClient

from assistant import config
from assistant.api import app, get_summarizer
from assistant.errors import AssistantError

TOKEN = "test-token-value"
PROSE = "Quarterly Review\n\nRevenue grew faster than forecast this quarter.\n"


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "auth-test.db"))
    yield


@pytest.fixture(autouse=True)
def no_token_by_default(monkeypatch):
    """Auth is off unless a test opts in — mirrors the default environment."""
    monkeypatch.delenv("ASSISTANT_API_TOKEN", raising=False)
    yield


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def with_token(monkeypatch):
    monkeypatch.setenv("ASSISTANT_API_TOKEN", TOKEN)
    return TOKEN


def auth_header(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


def txt_upload(name="notes.txt", body=PROSE):
    return {"file": (name, io.BytesIO(body.encode("utf-8")), "text/plain")}


def ingest(client, **kwargs):
    return client.post("/documents", files=txt_upload(), **kwargs).json()["id"]


# -- config.get_api_token ----------------------------------------------------


class TestGetApiToken:
    def test_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("ASSISTANT_API_TOKEN", raising=False)
        assert config.get_api_token() is None

    def test_returns_the_token_when_set(self, monkeypatch):
        monkeypatch.setenv("ASSISTANT_API_TOKEN", "abc123")
        assert config.get_api_token() == "abc123"

    def test_none_when_blank(self, monkeypatch):
        monkeypatch.setenv("ASSISTANT_API_TOKEN", "   ")
        assert config.get_api_token() is None

    def test_none_when_empty(self, monkeypatch):
        monkeypatch.setenv("ASSISTANT_API_TOKEN", "")
        assert config.get_api_token() is None

    def test_strips_surrounding_whitespace(self, monkeypatch):
        monkeypatch.setenv("ASSISTANT_API_TOKEN", "  abc123  ")
        assert config.get_api_token() == "abc123"


# -- auth disabled (the default) ---------------------------------------------


class TestAuthDisabled:
    """No token configured — every route stays open, as today."""

    def test_upload_works_without_a_header(self, client):
        assert client.post("/documents", files=txt_upload()).status_code == 201

    def test_summarize_works_without_a_header(self, client):
        app.dependency_overrides[get_summarizer] = lambda: (
            lambda doc: '{"summary": "ok", "decisions": [], "action_items": []}'
        )
        doc_id = ingest(client)

        assert client.post(f"/documents/{doc_id}/summarize").status_code == 200

    def test_stream_works_without_a_header(self, client):
        doc_id = ingest(client)

        app.dependency_overrides[get_summarizer] = lambda: (lambda doc: "")
        response = client.post(f"/documents/{doc_id}/summarize/stream")

        assert response.status_code in (200, 502)

    def test_a_stray_header_is_ignored(self, client):
        response = client.post(
            "/documents", files=txt_upload(), headers=auth_header("anything")
        )

        assert response.status_code == 201


# -- auth enabled ------------------------------------------------------------


class TestAuthEnabled:
    def test_upload_without_a_header_is_401(self, client, with_token):
        assert client.post("/documents", files=txt_upload()).status_code == 401

    def test_upload_with_a_wrong_token_is_401(self, client, with_token):
        response = client.post(
            "/documents", files=txt_upload(), headers=auth_header("wrong")
        )

        assert response.status_code == 401

    def test_upload_with_the_right_token_succeeds(self, client, with_token):
        response = client.post(
            "/documents", files=txt_upload(), headers=auth_header()
        )

        assert response.status_code == 201

    def test_401_explains_the_problem(self, client, with_token):
        detail = client.post("/documents", files=txt_upload()).json()["detail"]

        assert "token" in detail.lower()

    def test_401_carries_the_www_authenticate_header(self, client, with_token):
        response = client.post("/documents", files=txt_upload())

        assert response.headers.get("www-authenticate") == "Bearer"

    def test_bare_token_without_the_bearer_prefix_is_401(self, client, with_token):
        response = client.post(
            "/documents", files=txt_upload(), headers={"Authorization": TOKEN}
        )

        assert response.status_code == 401

    def test_wrong_scheme_is_401(self, client, with_token):
        response = client.post(
            "/documents",
            files=txt_upload(),
            headers={"Authorization": f"Basic {TOKEN}"},
        )

        assert response.status_code == 401

    def test_token_prefix_is_rejected(self, client, with_token):
        """A partial match must not pass."""
        response = client.post(
            "/documents", files=txt_upload(), headers=auth_header(TOKEN[:-1])
        )

        assert response.status_code == 401

    def test_summarize_without_a_header_is_401(self, client, with_token):
        assert client.post("/documents/1/summarize").status_code == 401

    def test_stream_without_a_header_is_401(self, client, with_token):
        assert client.post("/documents/1/summarize/stream").status_code == 401

    def test_auth_is_checked_before_the_document_is_looked_up(
        self, client, with_token
    ):
        """An unauthenticated caller shouldn't learn whether an id exists."""
        assert client.post("/documents/999/summarize").status_code == 401

    def test_summarize_with_the_right_token_succeeds(self, client, with_token):
        app.dependency_overrides[get_summarizer] = lambda: (
            lambda doc: '{"summary": "ok", "decisions": [], "action_items": []}'
        )
        doc_id = client.post(
            "/documents", files=txt_upload(), headers=auth_header()
        ).json()["id"]

        response = client.post(
            f"/documents/{doc_id}/summarize", headers=auth_header()
        )

        assert response.status_code == 200


class TestUnauthenticatedRoutesStayOpen:
    """Read-only routes cost nothing to hit and must stay browsable."""

    def test_health_is_open(self, client, with_token):
        assert client.get("/health").status_code == 200

    def test_list_documents_is_open(self, client, with_token):
        assert client.get("/documents").status_code == 200

    def test_get_document_is_open(self, client, with_token):
        doc_id = client.post(
            "/documents", files=txt_upload(), headers=auth_header()
        ).json()["id"]

        assert client.get(f"/documents/{doc_id}").status_code == 200

    def test_get_unknown_document_still_404s(self, client, with_token):
        assert client.get("/documents/999").status_code == 404

    def test_docs_is_open(self, client, with_token):
        assert client.get("/docs").status_code == 200

    def test_openapi_is_open(self, client, with_token):
        assert client.get("/openapi.json").status_code == 200


# -- streaming route ---------------------------------------------------------


class TestStreamRoute:
    @pytest.fixture(autouse=True)
    def fake_stream(self):
        """Patch the orchestration stream the route calls through."""
        import assistant.orchestration as orchestration

        original = orchestration.summarize_via_graph_stream
        yield
        orchestration.summarize_via_graph_stream = original

    def _patch_stream(self, monkeypatch, deltas):
        def fake(document, client=None, threshold=None):
            yield from deltas

        monkeypatch.setattr("assistant.main.orchestration.summarize_via_graph_stream", fake)

    def test_returns_200(self, client, monkeypatch):
        self._patch_stream(monkeypatch, ["hello"])
        doc_id = ingest(client)

        assert client.post(f"/documents/{doc_id}/summarize/stream").status_code == 200

    def test_assembles_the_streamed_content(self, client, monkeypatch):
        self._patch_stream(monkeypatch, ["one ", "two ", "three"])
        doc_id = ingest(client)

        response = client.post(f"/documents/{doc_id}/summarize/stream")

        assert response.text == "one two three"

    def test_content_type_is_plain_text(self, client, monkeypatch):
        self._patch_stream(monkeypatch, ["x"])
        doc_id = ingest(client)

        response = client.post(f"/documents/{doc_id}/summarize/stream")

        assert response.headers["content-type"].startswith("text/plain")

    def test_progress_lines_are_included(self, client, monkeypatch):
        self._patch_stream(
            monkeypatch, ["[summarizing chunk 1 of 2]\n", "result"]
        )
        doc_id = ingest(client)

        response = client.post(f"/documents/{doc_id}/summarize/stream")

        assert "[summarizing chunk 1 of 2]" in response.text

    def test_unknown_id_is_404(self, client):
        assert client.post("/documents/999/summarize/stream").status_code == 404

    def test_404_uses_the_shared_message(self, client):
        detail = client.post("/documents/999/summarize/stream").json()["detail"]

        assert "No document with id 999" in detail

    def test_mid_stream_failure_is_reported_in_band(self, client, monkeypatch):
        """Status is already 200 once bytes flow — surface the error in the body."""

        def failing(document, client=None, threshold=None):
            yield "partial output"
            raise AssistantError("upstream exploded")

        monkeypatch.setattr(
            "assistant.main.orchestration.summarize_via_graph_stream", failing
        )
        doc_id = ingest(client)

        response = client.post(f"/documents/{doc_id}/summarize/stream")

        assert response.status_code == 200
        assert "partial output" in response.text
        assert "[error: upstream exploded]" in response.text

    def test_route_is_documented_in_openapi(self, client):
        paths = client.get("/openapi.json").json()["paths"]

        assert "/documents/{doc_id}/summarize/stream" in paths
