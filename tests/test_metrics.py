"""Tests for metrics recording, interaction logging, and header parsing."""

import json

import pytest

from src import metrics


@pytest.fixture
def tmp_storage(tmp_path, monkeypatch):
    """Point metrics/interactions storage at temporary files."""
    metrics_file = tmp_path / "llm_metrics.json"
    interactions_file = tmp_path / "interactions.json"
    monkeypatch.setattr(metrics, "METRICS_FILE", metrics_file)
    monkeypatch.setattr(metrics, "INTERACTIONS_FILE", interactions_file)
    return metrics_file, interactions_file


class TestInteractionLogging:
    def test_log_interaction_persists(self, tmp_storage):
        _, interactions_file = tmp_storage
        metrics.log_interaction("SELECT * FROM cus", "tomers;", applied_autocomplete=True)

        records = json.loads(interactions_file.read_text())
        assert len(records) == 1
        assert records[0]["applied_autocomplete"] == 1
        assert records[0]["input_sql"] == "SELECT * FROM cus"

    def test_apply_rate_summary(self, tmp_storage):
        metrics.log_interaction("SELECT 1", "a", True)
        metrics.log_interaction("SELECT 2", "b", False)

        summary = metrics.get_interactions_summary()
        assert summary["total_interactions"] == 2
        assert summary["applied_count"] == 1
        assert summary["apply_rate_percent"] == 50.0

    def test_empty_summary(self, tmp_storage):
        summary = metrics.get_interactions_summary()
        assert summary["total_interactions"] == 0

    def test_export_interactions_jsonl(self, tmp_storage):
        metrics.log_interaction("SELECT 1", "a", True)
        path = metrics.export_interactions()
        lines = open(path).read().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["input_sql"] == "SELECT 1"

    def test_export_interactions_rejects_other_formats(self, tmp_storage):
        with pytest.raises(ValueError):
            metrics.export_interactions(format="csv")


class TestMetricsRecording:
    def test_record_and_summarize(self, tmp_storage):
        metrics.record_llm_metrics(1.0, 0.2, 10, 20, "model-a")
        metrics.record_llm_metrics(3.0, None, 30, 40, "model-a")

        summary = metrics.get_metrics_summary()
        assert summary["total_requests"] == 2
        assert summary["average_time_to_complete_seconds"] == 2.0
        # TTFT averaged only over requests that reported it
        assert summary["average_time_to_first_token_seconds"] == 0.2
        assert summary["requests_missing_server_ttft"] == 1
        assert summary["total_tokens"] == 100

    def test_missing_ttft_stored_as_null(self, tmp_storage):
        metrics_file, _ = tmp_storage
        metrics.record_llm_metrics(1.5, None, 5, 5, "model-a")
        record = json.loads(metrics_file.read_text())[0]
        assert record["time_to_first_token_seconds"] is None

    def test_export_metrics_json(self, tmp_storage):
        metrics.record_llm_metrics(1.0, 0.5, 10, 10, "model-a")
        path = metrics.export_metrics(format="json")
        data = json.loads(open(path).read())
        assert data["summary"]["total_requests"] == 1
        assert len(data["metrics"]) == 1


class TestHeaderParsing:
    def test_plain_values_are_seconds(self):
        headers = {
            "fireworks-server-time-to-first-token": "0.292",
            "fireworks-server-processing-time": "1.5",
        }
        result = metrics.extract_metrics_from_headers(headers)
        assert result["time_to_first_token"] == pytest.approx(0.292)
        assert result["server_latency"] == pytest.approx(1.5)

    def test_values_above_one_stay_seconds(self):
        # Fireworks reports plain header values in seconds, even > 1
        headers = {"fireworks-server-time-to-first-token": "2.9"}
        result = metrics.extract_metrics_from_headers(headers)
        assert result["time_to_first_token"] == pytest.approx(2.9)

    def test_ms_suffix_converted(self):
        headers = {"fireworks-server-time-to-first-token": "250ms"}
        result = metrics.extract_metrics_from_headers(headers)
        assert result["time_to_first_token"] == pytest.approx(0.25)

    def test_case_insensitive(self):
        headers = {"Fireworks-Server-Time-To-First-Token": "0.1"}
        result = metrics.extract_metrics_from_headers(headers)
        assert result["time_to_first_token"] == pytest.approx(0.1)

    def test_missing_headers_return_none(self):
        result = metrics.extract_metrics_from_headers({})
        assert result["time_to_first_token"] is None
        assert result["server_latency"] is None

    def test_unparseable_header_returns_none(self):
        headers = {"fireworks-server-time-to-first-token": "not-a-number"}
        result = metrics.extract_metrics_from_headers(headers)
        assert result["time_to_first_token"] is None
