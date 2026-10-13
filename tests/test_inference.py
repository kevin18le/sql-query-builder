"""Tests for JSON response parsing, context formatting, and completion assembly.

All external API calls are mocked; no Fireworks credentials are needed.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.fireworks import inference
from src.fireworks.inference import (
    SqlCompletion,
    complete_partial_sql,
    complete_sql,
    extract_json_from_response,
    format_retrieved_context,
)


class TestExtractJsonFromResponse:
    def test_pure_json(self):
        result = extract_json_from_response('{"sql": "SELECT 1"}', SqlCompletion)
        assert result.sql == "SELECT 1"

    def test_json_with_reasoning_tags(self):
        response = '<think>reasoning here</think> {"sql": "SELECT 2"}'
        result = extract_json_from_response(response, SqlCompletion)
        assert result.sql == "SELECT 2"

    def test_json_embedded_in_text(self):
        response = 'Here is the answer: {"sql": "SELECT 3"} hope this helps'
        result = extract_json_from_response(response, SqlCompletion)
        assert result.sql == "SELECT 3"

    def test_empty_response_raises(self):
        with pytest.raises(ValueError, match="Empty response"):
            extract_json_from_response("", SqlCompletion)
        with pytest.raises(ValueError, match="Empty response"):
            extract_json_from_response(None, SqlCompletion)

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="Could not extract JSON"):
            extract_json_from_response("no json here", SqlCompletion)

    def test_invalid_schema_raises(self):
        with pytest.raises(ValueError):
            extract_json_from_response('{"wrong_key": 1}', SqlCompletion)


class TestFormatRetrievedContext:
    def test_empty_results(self):
        assert format_retrieved_context([]) == ""

    def test_document_separators_and_attributes(self):
        results = [
            {
                "table_name": "customers",
                "content": "Table: customers",
                "score": 0.9,
                "attributes": {"row_count": 50, "primary_keys": ["customer_id"]},
            },
            {
                "table_name": "accounts",
                "content": "Table: accounts",
                "score": 0.8,
                "attributes": {},
            },
        ]
        out = format_retrieved_context(results)
        assert "# Document 1" in out
        assert "# Document 2" in out
        assert "Table: customers" in out
        assert "row_count: 50" in out
        assert "Score: 0.9" in out


def _fake_raw_response(content, headers=None, prompt_tokens=10, completion_tokens=5):
    parsed = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )
    raw = MagicMock()
    raw.parse.return_value = parsed
    raw.headers = headers or {}
    return raw


class TestCompletePartialSql:
    def test_success_records_metrics(self, monkeypatch):
        raw = _fake_raw_response(
            '{"sql": "tomers;"}',
            headers={"fireworks-server-time-to-first-token": "0.3"},
        )
        monkeypatch.setattr(inference, "OpenAI", lambda **kw: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=lambda **kw: raw)))))

        recorded = {}
        monkeypatch.setattr(inference, "record_llm_metrics", lambda **kw: recorded.update(kw))

        result = complete_partial_sql("key", "model", "SELECT * FROM cus")
        assert result == "tomers;"
        assert recorded["time_to_first_token"] == pytest.approx(0.3)
        assert recorded["input_tokens"] == 10
        assert recorded["output_tokens"] == 5

    def test_missing_ttft_header_stays_none(self, monkeypatch):
        raw = _fake_raw_response('{"sql": "x"}', headers={})
        monkeypatch.setattr(inference, "OpenAI", lambda **kw: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=lambda **kw: raw)))))

        recorded = {}
        monkeypatch.setattr(inference, "record_llm_metrics", lambda **kw: recorded.update(kw))

        complete_partial_sql("key", "model", "SELECT 1")
        # Total latency must not be substituted for missing TTFT
        assert recorded["time_to_first_token"] is None

    def test_model_not_found_raises_value_error(self, monkeypatch):
        class FakeNotFound(Exception):
            status_code = 404

        def raise_404(**kw):
            raise FakeNotFound("not found")

        monkeypatch.setattr(inference, "OpenAI", lambda **kw: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=raise_404)))))

        with pytest.raises(ValueError, match="model not found"):
            complete_partial_sql("key", "bad-model", "SELECT 1")

    def test_empty_completion_raises(self, monkeypatch):
        raw = _fake_raw_response("")
        monkeypatch.setattr(inference, "OpenAI", lambda **kw: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=lambda **kw: raw)))))
        monkeypatch.setattr(inference, "record_llm_metrics", lambda **kw: None)

        with pytest.raises(ValueError, match="Error parsing SQL completion"):
            complete_partial_sql("key", "model", "SELECT 1")

    def test_system_prompt_interpolates_schema(self, monkeypatch):
        captured = {}

        def capture_create(**kw):
            captured.update(kw)
            return _fake_raw_response('{"sql": "x"}')

        monkeypatch.setattr(inference, "OpenAI", lambda **kw: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(
                with_raw_response=SimpleNamespace(create=capture_create)))))
        monkeypatch.setattr(inference, "record_llm_metrics", lambda **kw: None)

        complete_partial_sql("key", "model", "SELECT 1")
        system_prompt = captured["messages"][0]["content"]
        # The JSON schema must be interpolated, not a literal placeholder
        assert "{SqlCompletion.model_json_schema()}" not in system_prompt
        assert '"sql"' in system_prompt


class TestCompleteSql:
    def test_without_retrieval_returns_none_context(self, monkeypatch):
        monkeypatch.setattr(inference, "complete_partial_sql", lambda *a, **kw: "completion")
        completion, context = complete_sql("key", "model", "SELECT 1", use_retrieval=False)
        assert completion == "completion"
        assert context is None

    def test_with_retrieval_formats_context(self, monkeypatch):
        monkeypatch.setattr(inference, "search_vector", lambda *a, **kw: [
            {"table_name": "t", "content": "c", "score": 1.0, "attributes": {}}
        ])
        seen = {}

        def fake_complete(api_key, model, partial_sql, schema_context):
            seen["schema_context"] = schema_context
            return "completion"

        monkeypatch.setattr(inference, "complete_partial_sql", fake_complete)
        completion, context = complete_sql("key", "model", "SELECT 1", use_retrieval=True, top_k=1)
        assert "# Document 1" in context
        assert seen["schema_context"] == context
