# TODO: CREATE A DEPLOYMENT ON FIREWORKS AI FOR BETTER PERFORMANCE
# TODO: TRACK AND EXPORT PERFORMANCE METRICS
# TODO: ADD FEW-SHOT EXAMPLES TO THE SYSTEM PROMPT
# TODO: INFERENCE PERFORMANCE OPTIMIZATION (try to get sub-second latency)
# NICE TO HAVE: TELEMETRY LOGGING TO CREATE RFT DATASET

import os
import re
import json
import time
from datetime import datetime
from pydantic import BaseModel
import yaml
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional
from pathlib import Path
from src.config import INFERENCE_BASE_URL
from src.search.vector_search import search_vector

load_dotenv()

_FILE_PATH = Path(__file__).parents[2]

# TODO: create a prompt library yaml file (if needed)
def load_prompt_library():
    """Load prompts from YAML configuration."""
    prompt_file = _FILE_PATH / "configs" / "prompt_library.yaml"
    if prompt_file.exists():
        with open(prompt_file, "r") as f:
            return yaml.safe_load(f)
    return {}


PROMPT_LIBRARY = load_prompt_library()

### Performance Metrics Tracking ###
_METRICS_STORAGE = []
_METRICS_FILE = _FILE_PATH / "data" / "llm_metrics.json"

### Interaction Logging ###
_INTERACTIONS_FILE = _FILE_PATH / "data" / "interactions.json"

def log_interaction(
    input_sql: str,
    suggested_completion: str,
    applied_autocomplete: bool,
    timestamp: Optional[str] = None
) -> Dict:
    """
    Log an autocomplete interaction to the interactions table.
    
    Args:
        input_sql: The original partial SQL query
        suggested_completion: The suggested completion from the model
        applied_autocomplete: Whether the user applied the autocomplete (True/False)
        timestamp: Optional timestamp (defaults to current time)
    
    Returns:
        Dictionary containing the logged interaction
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    interaction = {
        "timestamp": timestamp,
        "input_sql": input_sql,
        "suggested_completion": suggested_completion,
        "applied_autocomplete": 1 if applied_autocomplete else 0,
    }
    
    # Save to file (append mode)
    _INTERACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Load existing interactions if file exists
        if _INTERACTIONS_FILE.exists():
            with open(_INTERACTIONS_FILE, 'r') as f:
                existing_interactions = json.load(f)
        else:
            existing_interactions = []
        
        # Append new interaction
        existing_interactions.append(interaction)
        
        # Save back to file
        with open(_INTERACTIONS_FILE, 'w') as f:
            json.dump(existing_interactions, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save interaction to file: {e}")
    
    return interaction

def record_llm_metrics(
    time_to_complete: float,
    time_to_first_token: float,
    input_tokens: int,
    output_tokens: int,
    model: str,
    timestamp: Optional[str] = None
) -> Dict:
    """
    Record LLM request performance metrics.
    
    Args:
        time_to_complete: Total time to complete generation in seconds
        time_to_first_token: Time to first token in seconds
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        model: Model name used
        timestamp: Optional timestamp (defaults to current time)
    
    Returns:
        Dictionary containing the recorded metrics
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    metrics = {
        "timestamp": timestamp,
        "model": model,
        "time_to_complete_seconds": round(time_to_complete, 4),
        "time_to_first_token_seconds": round(time_to_first_token, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    
    _METRICS_STORAGE.append(metrics)
    
    # Save to file (append mode)
    _METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Load existing metrics if file exists
        if _METRICS_FILE.exists():
            with open(_METRICS_FILE, 'r') as f:
                existing_metrics = json.load(f)
        else:
            existing_metrics = []
        
        # Append new metrics
        existing_metrics.append(metrics)
        
        # Save back to file
        with open(_METRICS_FILE, 'w') as f:
            json.dump(existing_metrics, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save metrics to file: {e}")
    
    return metrics

def get_all_metrics() -> List[Dict]:
    """
    Get all recorded metrics.
    
    Returns:
        List of all recorded metrics dictionaries
    """
    # Load from file to get all metrics (including from previous sessions)
    try:
        if _METRICS_FILE.exists():
            with open(_METRICS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load metrics from file: {e}")
    
    return _METRICS_STORAGE.copy()

def get_metrics_summary() -> Dict:
    """
    Get summary statistics of all recorded metrics.
    
    Returns:
        Dictionary with summary statistics
    """
    all_metrics = get_all_metrics()
    
    if not all_metrics:
        return {
            "total_requests": 0,
            "message": "No metrics recorded yet"
        }
    
    total_requests = len(all_metrics)
    avg_time_to_complete = sum(m["time_to_complete_seconds"] for m in all_metrics) / total_requests
    avg_time_to_first_token = sum(m["time_to_first_token_seconds"] for m in all_metrics) / total_requests
    total_input_tokens = sum(m["input_tokens"] for m in all_metrics)
    total_output_tokens = sum(m["output_tokens"] for m in all_metrics)
    
    return {
        "total_requests": total_requests,
        "average_time_to_complete_seconds": round(avg_time_to_complete, 4),
        "average_time_to_first_token_seconds": round(avg_time_to_first_token, 4),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
    }

def export_metrics(format: str = "json") -> str:
    """
    Export performance metrics to a file.
    
    Args:
        format: Export format - "json" or "csv" (default: "json")
    
    Returns:
        Path to the exported file
    """
    all_metrics = get_all_metrics()
    summary = get_metrics_summary()
    
    if format.lower() == "csv":
        import csv
        export_file = _METRICS_FILE.parent / f"llm_metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(export_file, 'w', newline='') as f:
            if all_metrics:
                writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
                writer.writeheader()
                writer.writerows(all_metrics)
    else:  # JSON format
        export_file = _METRICS_FILE.parent / f"llm_metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "summary": summary,
            "metrics": all_metrics
        }
        
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    return str(export_file)

def get_all_interactions() -> List[Dict]:
    """
    Get all logged interactions.
    
    Returns:
        List of all interaction dictionaries
    """
    try:
        if _INTERACTIONS_FILE.exists():
            with open(_INTERACTIONS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load interactions from file: {e}")
    
    return []

def get_interactions_summary() -> Dict:
    """
    Get summary statistics of all logged interactions.
    
    Returns:
        Dictionary with summary statistics
    """
    all_interactions = get_all_interactions()
    
    if not all_interactions:
        return {
            "total_interactions": 0,
            "message": "No interactions logged yet"
        }
    
    total_interactions = len(all_interactions)
    applied_count = sum(1 for i in all_interactions if i.get("applied_autocomplete") == 1)
    not_applied_count = total_interactions - applied_count
    apply_rate = (applied_count / total_interactions * 100) if total_interactions > 0 else 0
    
    return {
        "total_interactions": total_interactions,
        "applied_count": applied_count,
        "not_applied_count": not_applied_count,
        "apply_rate_percent": round(apply_rate, 2),
    }

def export_interactions(format: str = "json") -> str:
    """
    Export interaction logs to a file for fine-tuning.
    
    Args:
        format: Export format - "json" (default: "json")
    
    Returns:
        Path to the exported file
    """
    all_interactions = get_all_interactions()
    summary = get_interactions_summary()
    
    if format.lower() == "json":
        export_file = _INTERACTIONS_FILE.parent / f"interactions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "summary": summary,
            "interactions": all_interactions
        }
        
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
    else:
        raise ValueError(f"Unsupported export format: {format}. Only 'json' is supported.")
    
    return str(export_file)

def extract_metrics_from_headers(headers: Dict) -> Dict[str, Optional[float]]:
    """
    Extract performance metrics from Fireworks AI response headers.
    
    According to Fireworks AI docs (https://docs.fireworks.ai/guides/querying-text-models#usage-and-performance-tracking),
    performance metrics are in response headers for non-streaming requests.
    Common headers include:
    - fireworks-server-time-to-first-token (in milliseconds)
    - fireworks-server-latency (in milliseconds)
    - fireworks-prompt-tokens (also available in response body)
    
    Args:
        headers: Dictionary of response headers (case-insensitive matching)
    
    Returns:
        Dictionary with extracted metrics (values in seconds, None if not found)
    """
    metrics = {
        "time_to_first_token": None,
        "server_latency": None,
    }
    
    # Fireworks AI header names for performance metrics
    # Headers are typically in milliseconds, we convert to seconds
    header_mappings = {
        "time_to_first_token": [
            "fireworks-server-time-to-first-token",
            "x-fireworks-time-to-first-token",
        ],
        "server_latency": [
            "fireworks-server-latency",
            "x-fireworks-latency",
        ],
    }
    
    # Convert headers to lowercase for case-insensitive matching
    headers_lower = {k.lower(): v for k, v in headers.items()}
    
    for metric_key, header_names in header_mappings.items():
        for header_name in header_names:
            header_lower = header_name.lower()
            if header_lower in headers_lower:
                try:
                    # Headers are typically strings, convert to float
                    value = float(headers_lower[header_lower])
                    # Fireworks AI headers are typically in milliseconds, convert to seconds
                    # If value is already < 1, assume it's already in seconds
                    if value > 1:
                        metrics[metric_key] = value / 1000.0
                    else:
                        metrics[metric_key] = value
                    break
                except (ValueError, TypeError) as e:
                    # If conversion fails, try to parse as string with units
                    header_value = str(headers_lower[header_lower]).strip().lower()
                    if header_value.endswith('ms'):
                        try:
                            metrics[metric_key] = float(header_value[:-2]) / 1000.0
                            break
                        except ValueError:
                            pass
                    elif header_value.endswith('s'):
                        try:
                            metrics[metric_key] = float(header_value[:-1])
                            break
                        except ValueError:
                            pass
    
    return metrics

### SQL Completion Utilities ###
class SqlCompletion(BaseModel):
    sql: str

def extract_json_from_response(response: str, response_model: BaseModel) -> BaseModel:
    """Extract JSON from response that may contain reasoning tags."""
    # First, try to parse the response directly as JSON (works for json_schema format)
    try:
        # Try direct JSON parsing first (when using json_schema, response should be pure JSON)
        json_data = json.loads(response.strip())
        parsed_response = response_model.model_validate(json_data)
        return parsed_response
    except (json.JSONDecodeError, ValueError):
        # If direct parsing fails, try regex extraction for responses with extra text/reasoning
        pass
    
    # Extract reasoning (if present)
    reasoning_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None

    # Extract JSON using regex
    if reasoning:
        json_match = re.search(r"</think>\s*(\{.*\})", response, re.DOTALL)
    else:
        # Try to find JSON object (supports multiline)
        json_match = re.search(r"(\{.*\})", response, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1).strip()
        try:
            parsed_response = response_model.model_validate_json(json_str)
            return parsed_response
        except Exception as e:
            raise ValueError(f"Failed to parse extracted JSON: {e}. Extracted JSON: {json_str}")
    else:
        raise ValueError(f"Could not extract JSON from response. Response content: {response[:500]}")

def complete_partial_sql(api_key: str, model: str, partial_sql: str, schema_context: str=None) -> str:
    if schema_context is None:
        schema_context = """
        """
    
    system_prompt = """
    You are a SQL autocomplete assistant. 
    Given a partial SQL query and database schema, you suggest the next tokens to complete the query. 
    Return ONLY the completion, not the original input in JSON format with the following schema:
    {SqlCompletion.model_json_schema()}
    """

    user_prompt = f"""SCHEMA:
# RETRIEVED SCHEMA CONTEXT
{schema_context}

# PARTIAL_SQL:
{partial_sql}

Return only the continuation of PARTIAL_SQL."""

    client = OpenAI(
        api_key=api_key,
        base_url=INFERENCE_BASE_URL,
    )
    
    # Record start time for measuring total completion time
    start_time = time.time()
    
    # Use with_raw_response to access headers for performance metrics
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
                "schema": SqlCompletion.model_json_schema()
            }
        }
    )
    
    # Calculate total time to complete
    time_to_complete = time.time() - start_time
    
    # Extract response object and headers
    response = raw_response.parse()
    
    # Extract headers - handle different header object types
    headers = {}
    try:
        if hasattr(raw_response, 'headers'):
            # Headers might be a dict, case-insensitive dict, or other type
            if isinstance(raw_response.headers, dict):
                headers = raw_response.headers
            else:
                # Convert to dict if it's another type (e.g., httpx.Headers)
                headers = dict(raw_response.headers)
    except Exception as e:
        print(f"Warning: Could not extract headers: {e}")
        headers = {}
    
    # Extract performance metrics from headers
    header_metrics = extract_metrics_from_headers(headers)
    
    # Get token usage from response
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
    
    # Use time-to-first-token from headers if available, otherwise use total time as fallback
    time_to_first_token = header_metrics.get("time_to_first_token") or time_to_complete
    
    # Record metrics
    record_llm_metrics(
        time_to_complete=time_to_complete,
        time_to_first_token=time_to_first_token,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model
    )
    
    # Extract and parse the response
    try:
        response_content = response.choices[0].message.content
        parsed_response = extract_json_from_response(response_content, SqlCompletion)
        return parsed_response.sql
    except Exception as e:
        error_msg = f"Error parsing SQL completion response: {str(e)}"
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
            error_msg += f"\nResponse content (first 500 chars): {content[:500]}"
        raise ValueError(error_msg) from e

def complete_sql(api_key, model, partial_sql, use_retrieval: bool = False, top_k: int = 5):
    if use_retrieval:
        # Retrieve schema context efficiently
        results = search_vector(partial_sql, top_k, api_key=api_key)
        
        # Build schema context string more efficiently using list comprehension
        # Only include attributes that exist to avoid KeyError
        context_parts = []
        for r in results:
            attrs = r.get('attributes', {})
            parts = [r['table_name']]
            
            # Only add attributes if they exist (for table_overview chunks)
            if 'row_count' in attrs:
                parts.append(f"Row count: {attrs['row_count']}")
            if 'primary_keys' in attrs:
                parts.append(f"Primary keys: {attrs['primary_keys']}")
            if 'unique_columns' in attrs:
                parts.append(f"Unique columns: {attrs['unique_columns']}")
            
            parts.append("")  # Empty line separator
            parts.append(r.get('content', ''))
            context_parts.append("\n".join(parts))
        
        schema_context = "\n\n".join(context_parts)
    else:
        schema_context = None
    return complete_partial_sql(api_key, model, partial_sql, schema_context)