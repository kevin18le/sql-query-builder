#!/usr/bin/env python3
"""Lightweight evaluation runner for SQL autocomplete.

Compares retrieval-enabled vs retrieval-disabled completion over a small
dataset of partial SQL queries (``eval/dataset.json``) and writes a
Markdown + JSON report to ``eval/``.

Recorded per run: client-observed latency, token usage, prefix preservation
(the completion must not repeat the input), and — when PostgreSQL is
available — syntax/schema validity via ``EXPLAIN`` in a read-only
transaction (no query is ever executed).

Limitations: syntactically valid SQL is not necessarily semantically
correct; validity here means "PostgreSQL can plan it", not "it answers the
intended question".

Usage:
    python -m eval.run_eval [--model MODEL] [--top-k N]
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import psycopg2
from dotenv import load_dotenv

from src.config import LLM_MODEL
from src.fireworks.inference import complete_sql

load_dotenv()

_FILE_PATH = Path(__file__).parents[1]
_DATASET_PATH = _FILE_PATH / "eval" / "dataset.json"


def check_sql_validity(full_sql: str, conn) -> Optional[bool]:
    """Plan (but never execute) the completed SQL in a read-only transaction.

    Returns True/False, or None when no database connection is available.
    """
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("BEGIN READ ONLY;")
            try:
                cur.execute(f"EXPLAIN {full_sql}")
                return True
            finally:
                cur.execute("ROLLBACK;")
    except psycopg2.Error:
        return False
    except Exception:
        return False


def get_db_connection():
    """Best-effort PostgreSQL connection; None when unavailable."""
    try:
        return psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            port=os.getenv("PGPORT", "5432"),
            database=os.getenv("PGDATABASE", "demo_bank"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", ""),
        )
    except psycopg2.Error as e:
        print(f"Warning: no database connection ({e}); skipping validity checks")
        return None


def run_one(api_key, model, partial_sql, use_retrieval, top_k, conn):
    """Run a single completion and collect evaluation fields."""
    start = time.time()
    completion, context = complete_sql(
        api_key, model, partial_sql, use_retrieval=use_retrieval, top_k=top_k
    )
    latency = time.time() - start

    full_sql = partial_sql + completion
    return {
        "completion": completion,
        "latency_seconds": round(latency, 4),
        "prefix_preserved": partial_sql not in completion,
        "sql_valid": check_sql_validity(full_sql, conn),
        "retrieved_chunks": context.count("# Document") if context else 0,
    }


def summarize(results, key):
    valid = [r for r in results if r[key].get("sql_valid") is True]
    latencies = [r[key]["latency_seconds"] for r in results]
    prefix = sum(1 for r in results if r[key]["prefix_preserved"])
    return {
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 4) if latencies else None,
        "prefix_preserved": f"{prefix}/{len(results)}",
        "sql_valid": f"{len(valid)}/{len(results)}" if valid else "n/a (no database)",
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SQL autocomplete with/without retrieval")
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit("FIREWORKS_API_KEY not set (see .env.example)")

    dataset = json.loads(_DATASET_PATH.read_text())
    conn = get_db_connection()

    results = []
    for item in dataset:
        print(f"[{item['name']}] retrieval on/off...")
        entry = {"name": item["name"], "category": item["category"], "partial_sql": item["partial_sql"]}
        entry["with_retrieval"] = run_one(api_key, args.model, item["partial_sql"], True, args.top_k, conn)
        entry["without_retrieval"] = run_one(api_key, args.model, item["partial_sql"], False, args.top_k, conn)
        results.append(entry)

    if conn:
        conn.close()

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "top_k": args.top_k,
        "summary": {
            "with_retrieval": summarize(results, "with_retrieval"),
            "without_retrieval": summarize(results, "without_retrieval"),
        },
        "results": results,
    }

    json_path = _FILE_PATH / "eval" / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_path.write_text(json.dumps(report, indent=2))

    # Markdown report
    lines = [
        "# SQL Autocomplete Evaluation",
        "",
        f"- Date: {report['timestamp']}",
        f"- Model: `{args.model}`",
        f"- Examples: {len(results)} | top_k: {args.top_k}",
        "",
        "## Summary",
        "",
        "| Configuration | Avg latency (s) | Prefix preserved | SQL valid (EXPLAIN) |",
        "|---|---|---|---|",
    ]
    for key, label in [("with_retrieval", "With retrieval"), ("without_retrieval", "Without retrieval")]:
        s = report["summary"][key]
        lines.append(f"| {label} | {s['avg_latency_seconds']} | {s['prefix_preserved']} | {s['sql_valid']} |")
    lines += [
        "",
        "Note: SQL validity means PostgreSQL could *plan* the completed query",
        "(EXPLAIN in a read-only transaction). It says nothing about whether the",
        "query is semantically correct for the user's intent.",
        "",
        "## Per-example results",
        "",
    ]
    for r in results:
        lines.append(f"### {r['name']} ({r['category']})")
        lines.append(f"- Partial: `{r['partial_sql']}`")
        for key, label in [("with_retrieval", "with retrieval"), ("without_retrieval", "without retrieval")]:
            e = r[key]
            lines.append(
                f"- {label}: `{e['completion']}` "
                f"(latency {e['latency_seconds']}s, valid: {e['sql_valid']}, "
                f"prefix preserved: {e['prefix_preserved']}, chunks: {e['retrieved_chunks']})"
            )
        lines.append("")

    md_path = json_path.with_suffix(".md")
    md_path.write_text("\n".join(lines))
    print(f"\nReports written:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
