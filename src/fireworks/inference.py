# TODO: ADD RETRIEVAL AND RERANKING BASED ON FAISS INDEX
# TODO: ADD PROMPT LIBRARY IN A YAML FILE
# TODO: EXPAND QUERY USING RETRIEVED SQL CONTEXT 
# TODO: CREATE A DEPLOYMENT ON FIREWORKS AI FOR BETTER PERFORMANCE
# TODO: ADD LOGGING AND ERROR HANDLING
# TODO: TRACK PERFORMANCE METRICS
# TODO: ADD A WEB APP WITH GRADIO TO INTERACT WITH THE MODEL
# NICE TO HAVE: TELEMETRY LOGGING TO CREATE RFT DATASET

import os
import re
from pydantic import BaseModel
import yaml
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
from pathlib import Path
import requests
from src.config import EMBEDDING_MODEL, LLM_MODEL, RERANKER_MODEL, INFERENCE_BASE_URL, RERANK_BASE_URL

load_dotenv()

_FILE_PATH = Path(__file__).parents[2]

# TODO: create a prompt library yaml file (if needed)
def load_prompt_library():
    """Load prompts from YAML configuration."""
    with open(_FILE_PATH / "configs" / "prompt_library.yaml", "r") as f:
        return yaml.safe_load(f)


def create_client() -> OpenAI:
    """
    Create client for FW inference
    """
    api_key = os.getenv("FIREWORKS_API_KEY")
    assert api_key is not None, "FIREWORKS_API_KEY not found in environment variables"
    return OpenAI(
        api_key=api_key,
        base_url=INFERENCE_BASE_URL,
    )


CLIENT = create_client()
PROMPT_LIBRARY = load_prompt_library()


def get_embedding(text: str) -> List[float]:
    """
    Get embedding for a given text using Fireworks AI embedding model.

    Args:
        text: Input text to embed

    Returns:
        List of float values representing the embedding vector
    """
    response = CLIENT.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def expand_query(query: str) -> str:
    """
    Expand a search query using LLM with few-shot prompting.

    Takes a user's search query and expands it with relevant terms, synonyms,
    and related concepts to improve search recall and relevance.

    Args:
        query: Original search query

    Returns:
        Expanded query string with additional relevant terms
    """
    # TODO: add the system prompt to the prompt library yaml file
    system_prompt = PROMPT_LIBRARY["query_expansion"]["system_prompt"]

    response = CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=100,
        reasoning_effort="none",
    )

    expanded = response.choices[0].message.content.strip()
    return expanded


def rerank_results(query: str, results: List[Dict], top_n: int = 5) -> List[Dict]:
    """
    Rerank search results using Fireworks AI reranker model.

    Takes search results and reranks them based on relevance to the query
    using a specialized reranking model that considers cross-attention between
    query and documents.

    Args:
        query: Original search query
        results: List of dictionaries containing product information and scores
        top_n: Number of top results to return after reranking (default: 5)

    Returns:
        List of dictionaries containing reranked product information with updated scores
    """
    # Prepare documents as text for reranker (product name + description)
    documents = [f"{r['product_name']}. {r['description']}" for r in results]

    payload = {
        "model": RERANKER_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": False,
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('FIREWORKS_API_KEY')}",
        "Content-Type": "application/json",
    }

    response = requests.post(RERANK_BASE_URL, json=payload, headers=headers)
    rerank_data = response.json()

    # Map reranked results back to original product data
    reranked_results = []
    for item in rerank_data.get("data", []):
        idx = item["index"]
        reranked_results.append({**results[idx], "score": item["relevance_score"]})

    return reranked_results

class SqlCompletion(BaseModel):
    sql: str

def extract_json_from_response(response: str, response_model: BaseModel) -> str:
    """Extract JSON from response that may contain reasoning tags."""
    # Extract reasoning (if present)
    reasoning_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None

    # Extract JSON
    json_match = re.search(r"</think>\s*(\{.*\})", response, re.DOTALL) if reasoning else re.search(r"(\{.*\})", response, re.DOTALL)
    json_str = json_match.group(1).strip() if json_match else "{}"

    # Parse into Pydantic model
    parsed_response = response_model.model_validate_json(json_str)
    return parsed_response

def complete_partial_sql(api_key: str, model: str, partial_sql: str, schema_context: str=None) -> str:
    if schema_context is None:
        schema_context = """
        """
    # 1) Build prompt (MOVE TO PROMPT LIBRARY YAML FILE)
    system_prompt = (
        "You are a SQL autocomplete assistant. "
        "Given a partial SQL query and database schema, "
        "you suggest the next tokens to complete the query. "
        "Return ONLY the completion, not the original input."
    )
    user_prompt = f"""SCHEMA:
# CALL 
{schema_context}

PARTIAL_SQL:
{partial_sql}

Return only the continuation of PARTIAL_SQL."""

    client = OpenAI(
        api_key=api_key, # hardcoded for now
        base_url=INFERENCE_BASE_URL,
    )
    response = client.chat.completions.create(
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
    parsed_response = extract_json_from_response(response.choices[0].message.content, SqlCompletion)
    return parsed_response.sql

def autocomplete(api_key, model, partial_sql):
    schema_context = open("schema_context.txt").read()
    return complete_partial_sql(api_key, model, partial_sql, schema_context)