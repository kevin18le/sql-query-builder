"""SQL completion inference via the Fireworks AI OpenAI-compatible API.

Uses json_schema structured output so completions are returned as
validated JSON. Performance metrics are recorded per request; see
``src.metrics`` for storage and export.
"""

import json
import re
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from src.config import INFERENCE_BASE_URL
from src.metrics import extract_metrics_from_headers, record_llm_metrics
from src.search.vector_search import search_vector

load_dotenv()


class SqlCompletion(BaseModel):
    sql: str


def extract_json_from_response(response: str, response_model: type[BaseModel]) -> BaseModel:
    """Parse a model response into ``response_model``.

    Tries direct JSON parsing first (the expected path with json_schema
    structured output), then falls back to extracting a JSON object from
    responses that contain reasoning tags or surrounding text.
    """
    if not response or not response.strip():
        raise ValueError("Empty response content from model")

    try:
        return response_model.model_validate(json.loads(response.strip()))
    except (json.JSONDecodeError, ValueError):
        pass

    # Strip reasoning tags, then look for a JSON object
    reasoning_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if reasoning_match:
        json_match = re.search(r"</think>\s*(\{.*\})", response, re.DOTALL)
    else:
        json_match = re.search(r"(\{.*\})", response, re.DOTALL)

    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return response_model.model_validate_json(json_str)
        except Exception as e:
            raise ValueError(f"Failed to parse extracted JSON: {e}. Extracted JSON: {json_str}")

    raise ValueError(f"Could not extract JSON from response. Response content: {response[:500]}")


def complete_partial_sql(
    api_key: str,
    model: str,
    partial_sql: str,
    schema_context: Optional[str] = None,
) -> str:
    """Generate a completion for a partial SQL query.

    Args:
        api_key: Fireworks API key.
        model: Fireworks model identifier.
        partial_sql: The partial SQL query to complete.
        schema_context: Retrieved schema context, or None to complete
            without retrieval.

    Returns:
        The suggested SQL continuation (without the original input).

    Raises:
        ValueError: On unknown models, empty responses, or unparseable output.
    """
    schema_block = schema_context if schema_context else "(no schema context provided)"

    system_prompt = f"""You are a SQL autocomplete assistant.
Given a partial SQL query and database schema, you suggest the next tokens to complete the query.
Return ONLY the completion, not the original input, in JSON format with the following schema:
{json.dumps(SqlCompletion.model_json_schema())}"""

    user_prompt = f"""SCHEMA:
# RETRIEVED SCHEMA CONTEXT
{schema_block}

# PARTIAL_SQL:
{partial_sql}

Return only the continuation of PARTIAL_SQL."""

    client = OpenAI(api_key=api_key, base_url=INFERENCE_BASE_URL)

    # Client-observed start time for end-to-end latency
    start_time = time.time()

    try:
        raw_response = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "SqlCompletion",
                    "schema": SqlCompletion.model_json_schema(),
                },
            },
        )
    except Exception as api_error:
        status_code = getattr(api_error, "status_code", None)
        if status_code == 404 or type(api_error).__name__ == "NotFoundError":
            raise ValueError(
                f"LLM model not found: {model}. "
                f"Please check that the model name is correct and available."
            ) from api_error
        raise

    time_to_complete = time.time() - start_time

    response = raw_response.parse()

    try:
        headers = dict(raw_response.headers) if hasattr(raw_response, "headers") else {}
    except Exception as e:
        print(f"Warning: Could not extract headers: {e}")
        headers = {}

    # Server-reported metrics (may be None when headers are absent).
    # The client-observed total latency is never substituted for TTFT.
    header_metrics = extract_metrics_from_headers(headers)
    time_to_first_token = header_metrics.get("time_to_first_token")

    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0

    record_llm_metrics(
        time_to_complete=time_to_complete,
        time_to_first_token=time_to_first_token,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
    )

    try:
        response_content = response.choices[0].message.content
        parsed_response = extract_json_from_response(response_content, SqlCompletion)
        return parsed_response.sql
    except Exception as e:
        error_msg = f"Error parsing SQL completion response: {e}"
        if response.choices:
            content = response.choices[0].message.content
            error_msg += f"\nResponse content (first 500 chars): {(content or '')[:500]}"
        raise ValueError(error_msg) from e


def format_retrieved_context(results: List[Dict]) -> str:
    """Format retrieved schema chunks with '# Document {n}' separators."""
    if not results:
        return ""

    formatted_parts = []
    for idx, r in enumerate(results, start=1):
        parts = [
            f"# Document {idx}",
            f"Table: {r['table_name']}",
            f"Content: {r.get('content', '')}",
        ]
        for attr_key, attr_value in r.get("attributes", {}).items():
            parts.append(f"{attr_key}: {attr_value}")
        parts.append("")
        parts.append(f"Score: {r.get('score', '')}")
        formatted_parts.append("\n".join(parts))

    return "\n\n".join(formatted_parts)


def complete_sql(
    api_key: str,
    model: str,
    partial_sql: str,
    use_retrieval: bool = False,
    top_k: int = 5,
) -> Tuple[str, Optional[str]]:
    """Complete a partial SQL query, optionally with schema retrieval.

    Returns:
        Tuple of (completion, formatted_context). The context is None when
        retrieval is disabled.
    """
    retrieved_context = None
    if use_retrieval:
        results = search_vector(partial_sql, top_k, api_key=api_key)
        retrieved_context = format_retrieved_context(results)

    completion = complete_partial_sql(api_key, model, partial_sql, retrieved_context)
    return completion, retrieved_context
