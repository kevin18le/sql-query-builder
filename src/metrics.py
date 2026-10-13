"""Performance metrics and interaction logging.

Metrics and interactions are appended to JSON files under ``data/`` so they
persist across app sessions and can be exported for analysis.

Latency fields:
- ``time_to_complete_seconds``: client-observed end-to-end request latency.
- ``time_to_first_token_seconds``: server-reported TTFT from Fireworks
  response headers, in seconds. ``None`` when the header is absent; the
  client-observed total latency is never substituted for it.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_FILE_PATH = Path(__file__).parents[1]

METRICS_FILE = _FILE_PATH / "data" / "llm_metrics.json"
INTERACTIONS_FILE = _FILE_PATH / "data" / "interactions.json"


def _append_json_record(path: Path, record: Dict) -> None:
    """Append a record to a JSON array file, creating it if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        records = []
        if path.exists():
            with open(path, "r") as f:
                records = json.load(f)
        records.append(record)
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save record to {path}: {e}")


def _load_json_records(path: Path) -> List[Dict]:
    """Load a JSON array file, returning an empty list on any failure."""
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load records from {path}: {e}")
    return []


### Interaction Logging ###

def log_interaction(
    input_sql: str,
    suggested_completion: str,
    applied_autocomplete: bool,
    timestamp: Optional[str] = None,
) -> Dict:
    """Log an accept/reject autocomplete interaction."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    interaction = {
        "timestamp": timestamp,
        "input_sql": input_sql,
        "suggested_completion": suggested_completion,
        "applied_autocomplete": 1 if applied_autocomplete else 0,
    }
    _append_json_record(INTERACTIONS_FILE, interaction)
    return interaction


def get_all_interactions() -> List[Dict]:
    """Return all logged interactions (including previous sessions)."""
    return _load_json_records(INTERACTIONS_FILE)


def get_interactions_summary() -> Dict:
    """Return summary statistics over all logged interactions."""
    all_interactions = get_all_interactions()

    if not all_interactions:
        return {"total_interactions": 0, "message": "No interactions logged yet"}

    total = len(all_interactions)
    applied = sum(1 for i in all_interactions if i.get("applied_autocomplete") == 1)
    return {
        "total_interactions": total,
        "applied_count": applied,
        "not_applied_count": total - applied,
        "apply_rate_percent": round(applied / total * 100, 2),
    }


def export_interactions(format: str = "jsonl") -> str:
    """Export interaction logs to a timestamped JSONL file (e.g. for fine-tuning)."""
    if format.lower() != "jsonl":
        raise ValueError(f"Unsupported export format: {format}. Only 'jsonl' is supported.")

    export_file = INTERACTIONS_FILE.parent / f"interactions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(export_file, "w") as f:
        for interaction in get_all_interactions():
            f.write(json.dumps(interaction) + "\n")
    return str(export_file)


### LLM Performance Metrics ###

def record_llm_metrics(
    time_to_complete: float,
    time_to_first_token: Optional[float],
    input_tokens: int,
    output_tokens: int,
    model: str,
    timestamp: Optional[str] = None,
) -> Dict:
    """Record metrics for a single LLM request.

    ``time_to_first_token`` is server-reported and may be ``None`` when the
    API response does not include the TTFT header.
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    metrics = {
        "timestamp": timestamp,
        "model": model,
        "time_to_complete_seconds": round(time_to_complete, 4),
        "time_to_first_token_seconds": (
            round(time_to_first_token, 4) if time_to_first_token is not None else None
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    _append_json_record(METRICS_FILE, metrics)
    return metrics


def get_all_metrics() -> List[Dict]:
    """Return all recorded metrics (including previous sessions)."""
    return _load_json_records(METRICS_FILE)


def get_metrics_summary() -> Dict:
    """Return summary statistics over all recorded metrics.

    TTFT averages are computed only over requests where the server reported
    TTFT; the count of missing TTFT values is reported explicitly.
    """
    all_metrics = get_all_metrics()

    if not all_metrics:
        return {"total_requests": 0, "message": "No metrics recorded yet"}

    total = len(all_metrics)
    avg_complete = sum(m["time_to_complete_seconds"] for m in all_metrics) / total

    ttft_values = [
        m["time_to_first_token_seconds"]
        for m in all_metrics
        if m.get("time_to_first_token_seconds") is not None
    ]
    total_input = sum(m["input_tokens"] for m in all_metrics)
    total_output = sum(m["output_tokens"] for m in all_metrics)

    return {
        "total_requests": total,
        "average_time_to_complete_seconds": round(avg_complete, 4),
        "average_time_to_first_token_seconds": (
            round(sum(ttft_values) / len(ttft_values), 4) if ttft_values else None
        ),
        "requests_with_server_ttft": len(ttft_values),
        "requests_missing_server_ttft": total - len(ttft_values),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_input + total_output,
    }


def export_metrics(format: str = "json") -> str:
    """Export performance metrics to a timestamped JSON or CSV file."""
    all_metrics = get_all_metrics()
    summary = get_metrics_summary()

    if format.lower() == "csv":
        export_file = METRICS_FILE.parent / f"llm_metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(export_file, "w", newline="") as f:
            if all_metrics:
                writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
                writer.writeheader()
                writer.writerows(all_metrics)
    else:
        export_file = METRICS_FILE.parent / f"llm_metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "summary": summary,
            "metrics": all_metrics,
        }
        with open(export_file, "w") as f:
            json.dump(export_data, f, indent=2)

    return str(export_file)


### Response Header Parsing ###

def extract_metrics_from_headers(headers: Dict) -> Dict[str, Optional[float]]:
    """Extract server-reported performance metrics from Fireworks response headers.

    Fireworks reports performance metrics in response headers for
    non-streaming requests, in seconds (verified empirically, e.g.
    ``fireworks-server-time-to-first-token: 0.292``):
    - ``fireworks-server-time-to-first-token``
    - ``fireworks-server-processing-time``

    Returns values in seconds, or ``None`` when a header is absent or
    unparseable. Values with an explicit "ms"/"s" suffix are converted
    accordingly; plain numeric values are treated as seconds.
    """
    metrics: Dict[str, Optional[float]] = {
        "time_to_first_token": None,
        "server_latency": None,
    }

    header_mappings = {
        "time_to_first_token": [
            "fireworks-server-time-to-first-token",
            "x-fireworks-time-to-first-token",
        ],
        "server_latency": [
            "fireworks-server-processing-time",
            "fireworks-server-latency",
            "x-fireworks-latency",
        ],
    }

    headers_lower = {k.lower(): v for k, v in headers.items()}

    for metric_key, header_names in header_mappings.items():
        for header_name in header_names:
            if header_name not in headers_lower:
                continue
            raw = str(headers_lower[header_name]).strip().lower()
            try:
                if raw.endswith("ms"):
                    metrics[metric_key] = float(raw[:-2]) / 1000.0
                elif raw.endswith("s"):
                    metrics[metric_key] = float(raw[:-1])
                else:
                    # Plain numeric values are seconds
                    metrics[metric_key] = float(raw)
                break
            except ValueError:
                continue

    return metrics
