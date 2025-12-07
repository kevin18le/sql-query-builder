import json
import os
import numpy as np
import faiss
import requests
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from src.config import FAISS_INDEX, EMBEDDING_MODEL, EMBEDDING_BASE_URL

load_dotenv()

# Load chunks metadata once at module level
_FILE_PATH = Path(__file__).parents[2]
_CHUNKS_METADATA_PATH = _FILE_PATH / "data" / "chunks_metadata.json"

def _load_chunks_metadata() -> List[Dict[str, Any]]:
    """Load chunks metadata from JSON file."""
    with open(_CHUNKS_METADATA_PATH, 'r') as f:
        return json.load(f)

_CHUNKS_METADATA = _load_chunks_metadata()

# Cache embeddings for queries (LRU cache with max size)
# Key: (query_text, api_key), Value: embedding vector
_EMBEDDING_CACHE: Dict[tuple, List[float]] = {}
_MAX_CACHE_SIZE = 100  # Limit cache size to prevent memory issues


### Embedding and Reranking Utilities ###
def get_embedding(text: str, api_key: str = None) -> List[float]:
    """
    Get embedding for a given text using Fireworks AI embedding model.
    Uses caching to avoid redundant API calls for the same query.

    Args:
        text: Input text to embed
        api_key: Optional API key. If not provided, uses FIREWORKS_API_KEY env var.

    Returns:
        List of float values representing the embedding vector
        
    Raises:
        ValueError: If API key is not found or response is invalid
        Exception: If embedding API call fails
    """
    if api_key is None:
        api_key = os.getenv("FIREWORKS_API_KEY")
    
    if api_key is None:
        raise ValueError("FIREWORKS_API_KEY not found. Please provide it as an argument or set it as an environment variable.")
    
    # Check cache first
    cache_key = (text, api_key)
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]
    
    # Create client and make API call
    client = OpenAI(
        api_key=api_key,
        base_url=EMBEDDING_BASE_URL,
    )
    
    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    except Exception as api_error:
        # Handle 404 errors (model not found)
        # Check for status_code attribute (OpenAI APIError)
        status_code = getattr(api_error, 'status_code', None)
        # Also check for NotFoundError exception type
        error_type = type(api_error).__name__
        
        if status_code == 404 or error_type == 'NotFoundError':
            raise ValueError(
                f"Embedding model not found: {EMBEDDING_MODEL}. "
                f"Please check that the model name is correct and available."
            ) from api_error
        # Re-raise other API errors with original exception
        raise
    
    # Validate response
    if not response or not response.data:
        raise ValueError("Invalid response from embedding API: no data returned")
    
    if len(response.data) == 0:
        raise ValueError("Invalid response from embedding API: empty data array")
    
    embedding = response.data[0].embedding
    
    if not embedding or len(embedding) == 0:
        raise ValueError("Invalid embedding returned: empty embedding vector")
    
    # Cache the result (with size limit using simple FIFO eviction)
    if len(_EMBEDDING_CACHE) >= _MAX_CACHE_SIZE:
        # Remove oldest entry (first key in dict for Python 3.7+)
        _EMBEDDING_CACHE.pop(next(iter(_EMBEDDING_CACHE)))
    
    _EMBEDDING_CACHE[cache_key] = embedding
    
    return embedding


# def rerank_results(query: str, results: List[Dict], top_n: int = 5, api_key: str = None) -> List[Dict]:
#     """
#     Rerank search results using Fireworks AI reranker model.

#     Takes search results and reranks them based on relevance to the query
#     using a specialized reranking model that considers cross-attention between
#     query and documents.

#     Args:
#         query: Original search query
#         results: List of dictionaries containing schema information
#         top_n: Number of top results to return after reranking (default: 5)
#         api_key: Optional API key. If not provided, uses FIREWORKS_API_KEY env var.

#     Returns:
#         List of dictionaries containing reranked schema information with updated scores
#     """
#     if api_key is None:
#         api_key = os.getenv("FIREWORKS_API_KEY")
    
#     if api_key is None:
#         raise ValueError("FIREWORKS_API_KEY not found. Please provide it as an argument or set it as an environment variable.")
    
#     # Prepare documents as text for reranker (table_name + content)
#     documents = [f"{r.get('table_name', '')}: {r.get('content', '')}" for r in results]

#     payload = {
#         "model": RERANKER_MODEL,
#         "query": query,
#         "documents": documents,
#         "top_n": top_n,
#         "return_documents": False,
#     }

#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json",
#     }

#     # Use session for connection pooling
#     response = _RERANK_SESSION.post(RERANK_BASE_URL, json=payload, headers=headers)
#     rerank_data = response.json()

#     # Map reranked results back to original schema data
#     reranked_results = []
#     for item in rerank_data.get("data", []):
#         idx = item["index"]
#         reranked_results.append({**results[idx], "score": item["relevance_score"]})

#     return reranked_results


### Vector Search Functions ###
def search_vector(query: str, top_k: int = 3, api_key: str = None) -> List[Dict[str, Any]]:
    """
    Search schema using vector embeddings and FAISS for semantic search.

    This is Stage 2: semantic search using vector embeddings to understand
    query meaning and intent beyond exact keyword matching.

    Args:
        query: Search query string
        top_k: Number of top results to return (default: 3)
        api_key: Optional API key. If not provided, uses FIREWORKS_API_KEY env var.

    Returns:
        List of dictionaries containing human-readable schema information and scores
    """
    query_embedding = get_embedding(query, api_key=api_key)
    query_vector = np.array([query_embedding], dtype=np.float32)

    faiss.normalize_L2(query_vector)
    faiss_index = FAISS_INDEX
    distances, indices = faiss_index.search(query_vector, top_k)

    # Convert L2 distances to similarity scores (0-1 range)
    # After normalization, L2 distance = 2 * (1 - cosine_similarity)
    # So cosine_similarity = 1 - (L2_distance / 2)
    similarity_scores = 1 - (distances[0] / 2)

    # Pre-allocate results list for better performance
    results = []
    results_append = results.append  # Cache method reference
    
    for idx, score in zip(indices[0], similarity_scores):
        chunk = _CHUNKS_METADATA[idx]
        metadata = chunk.get("metadata", {})
        
        # Build human-readable result based on chunk type
        result = {
            "table_name": chunk["table_name"],
            "chunk_type": chunk["type"],
            "content": chunk["content"],
            "score": float(score),
        }
        
        # Add structured metadata based on chunk type (optimized)
        chunk_type = chunk["type"]
        if chunk_type == "table_overview":
            result["attributes"] = {
                "row_count": metadata.get("row_count"),
                "primary_keys": metadata.get("primary_keys", []),
                "unique_columns": metadata.get("unique_columns", []),
            }
        elif chunk_type == "columns":
            result["attributes"] = {
                "columns": [
                    {
                        "name": col["name"],
                        "data_type": col["data_type"],
                        "nullable": col["nullable"]
                    }
                    for col in metadata.get("columns", [])
                ]
            }
        elif chunk_type == "foreign_keys":
            result["attributes"] = {
                "foreign_keys": [
                    {
                        "column": fk["column"],
                        "references_table": fk["references_table"],
                        "references_column": fk["references_column"]
                    }
                    for fk in metadata.get("foreign_keys", [])
                ]
            }
        elif chunk_type == "constraints_indexes":
            result["attributes"] = {
                "check_constraints": metadata.get("check_constraints", []),
                "indexes": [
                    {
                        "name": idx["name"],
                        "definition": idx["definition"]
                    }
                    for idx in metadata.get("indexes", [])
                ]
            }
        elif chunk_type == "relationship":
            result["attributes"] = {
                "from_table": metadata.get("from_table"),
                "from_column": metadata.get("from_column"),
                "to_table": metadata.get("to_table"),
                "to_column": metadata.get("to_column"),
            }
        
        results_append(result)
    
    return results


# def search_vector_with_reranking(query: str, top_k: int = 5, api_key: str = None) -> List[Dict[str, Any]]:
#     """
#     Search schema using vector embeddings and FAISS for semantic search with reranking.

#     This is Stage 3: semantic search using vector embeddings to understand
#     query meaning and intent beyond exact keyword matching, with reranking using LLM.

#     Args:
#         query: Search query string
#         top_k: Number of top results to return (default: 5)
#         api_key: Optional API key. If not provided, uses FIREWORKS_API_KEY env var.

#     Returns:
#         List of dictionaries containing schema information with preserved cosine scores
#     """
#     results = search_vector(query, top_k, api_key=api_key)
#     # Create a unique key for each result (table_name + chunk_type)
#     cosine_scores = {f"{r['table_name']}_{r['chunk_type']}": r["score"] for r in results}
#     reranked_results = rerank_results(query=query, results=results, api_key=api_key)

#     for r in reranked_results:
#         key = f"{r['table_name']}_{r['chunk_type']}"
#         if key in cosine_scores:
#             r["score"] = cosine_scores[key]

#     return reranked_results