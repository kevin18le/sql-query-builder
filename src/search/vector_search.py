"""Schema retrieval: embed the partial query and search the FAISS index.

The FAISS index and chunk metadata are loaded lazily on first use so that
importing this module (or ``src.config``) never requires the index to
exist. Build the index first with::

    python -m src.data_prep.build_schema_index
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from src.config import EMBEDDING_MODEL, INFERENCE_BASE_URL

load_dotenv()

_FILE_PATH = Path(__file__).parents[2]
_INDEX_PATH = _FILE_PATH / "data" / "schema_index.bin"
_CHUNKS_METADATA_PATH = _FILE_PATH / "data" / "chunks_metadata.json"

_faiss_index: Optional[faiss.Index] = None
_chunks_metadata: Optional[List[Dict[str, Any]]] = None


def _get_index() -> faiss.Index:
    """Load the FAISS index on first use."""
    global _faiss_index
    if _faiss_index is None:
        if not _INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {_INDEX_PATH}. "
                "Build it first with: python -m src.data_prep.build_schema_index"
            )
        _faiss_index = faiss.read_index(str(_INDEX_PATH))
        print(f"Loaded FAISS index with {_faiss_index.ntotal} vectors")
    return _faiss_index


def _get_chunks_metadata() -> List[Dict[str, Any]]:
    """Load chunk metadata on first use."""
    global _chunks_metadata
    if _chunks_metadata is None:
        if not _CHUNKS_METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Chunk metadata not found at {_CHUNKS_METADATA_PATH}. "
                "Build it first with: python -m src.data_prep.build_schema_index"
            )
        with open(_CHUNKS_METADATA_PATH, "r") as f:
            _chunks_metadata = json.load(f)
    return _chunks_metadata


def get_embedding(text: str, api_key: Optional[str] = None) -> List[float]:
    """Embed text with the Fireworks embedding model.

    Raises:
        ValueError: If no API key is available or the response is invalid.
    """
    if api_key is None:
        api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError(
            "FIREWORKS_API_KEY not found. Provide it as an argument or set it "
            "as an environment variable (see .env.example)."
        )

    client = OpenAI(api_key=api_key, base_url=INFERENCE_BASE_URL)

    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    except Exception as api_error:
        status_code = getattr(api_error, "status_code", None)
        if status_code == 404 or type(api_error).__name__ == "NotFoundError":
            raise ValueError(
                f"Embedding model not found: {EMBEDDING_MODEL}. "
                "Please check that the model name is correct and available."
            ) from api_error
        raise

    if not response or not response.data:
        raise ValueError("Invalid response from embedding API: no data returned")

    embedding = response.data[0].embedding
    if not embedding:
        raise ValueError("Invalid embedding returned: empty embedding vector")

    return embedding


def _format_result(chunk: Dict[str, Any], score: float) -> Dict[str, Any]:
    """Build a human-readable search result from a chunk and its score."""
    metadata = chunk.get("metadata", {})
    result = {
        "table_name": chunk["table_name"],
        "chunk_type": chunk["type"],
        "content": chunk["content"],
        "score": float(score),
    }

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
                    "nullable": col["nullable"],
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
                    "references_column": fk["references_column"],
                }
                for fk in metadata.get("foreign_keys", [])
            ]
        }
    elif chunk_type == "constraints_indexes":
        result["attributes"] = {
            "check_constraints": metadata.get("check_constraints", []),
            "indexes": [
                {"name": idx["name"], "definition": idx["definition"]}
                for idx in metadata.get("indexes", [])
            ],
        }
    elif chunk_type == "relationship":
        result["attributes"] = {
            "from_table": metadata.get("from_table"),
            "from_column": metadata.get("from_column"),
            "to_table": metadata.get("to_table"),
            "to_column": metadata.get("to_column"),
        }

    return result


def search_vector(
    query: str, top_k: int = 3, api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Semantic schema search over the FAISS index.

    Both stored embeddings and query vectors are L2-normalized, so the
    inner-product index returns cosine similarity scores.

    Args:
        query: Search query string (typically the partial SQL).
        top_k: Number of results; clamped to the index size.
        api_key: Optional Fireworks API key (defaults to env var).

    Returns:
        List of result dicts with table name, chunk type, content, score,
        and structured attributes. Empty when the index is empty or the
        query is blank.
    """
    if not query or not query.strip():
        return []

    index = _get_index()
    chunks_metadata = _get_chunks_metadata()

    if index.ntotal == 0:
        return []

    top_k = max(1, min(int(top_k), index.ntotal))

    query_embedding = get_embedding(query, api_key=api_key)
    query_vector = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        # FAISS returns -1 for missing results (e.g. top_k > ntotal)
        if idx < 0 or idx >= len(chunks_metadata):
            continue
        results.append(_format_result(chunks_metadata[idx], score))

    return results
