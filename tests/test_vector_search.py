"""Tests for FAISS retrieval edge cases (no API calls or database needed)."""

import faiss
import numpy as np
import pytest

from src.search import vector_search


@pytest.fixture
def small_index(monkeypatch):
    """A 3-vector normalized index with matching chunk metadata."""
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(3)
    index.add(vectors)

    chunks = [
        {"type": "table_overview", "table_name": "customers", "content": "Table: customers",
         "metadata": {"row_count": 50, "primary_keys": ["customer_id"], "unique_columns": []}},
        {"type": "columns", "table_name": "accounts", "content": "Table: accounts",
         "metadata": {"columns": [{"name": "account_id", "data_type": "int", "nullable": False}]}},
        {"type": "relationship", "table_name": "accounts", "content": "accounts -> customers",
         "metadata": {"from_table": "accounts", "from_column": "customer_id",
                      "to_table": "customers", "to_column": "customer_id"}},
    ]

    monkeypatch.setattr(vector_search, "_faiss_index", index)
    monkeypatch.setattr(vector_search, "_chunks_metadata", chunks)
    # Deterministic embedding: match the first stored vector
    monkeypatch.setattr(vector_search, "get_embedding", lambda text, api_key=None: [1.0, 0.0, 0.0])
    return index, chunks


class TestSearchVector:
    def test_returns_best_match(self, small_index):
        results = vector_search.search_vector("customers", top_k=1)
        assert len(results) == 1
        assert results[0]["table_name"] == "customers"
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
        assert results[0]["attributes"]["row_count"] == 50

    def test_top_k_clamped_to_index_size(self, small_index):
        results = vector_search.search_vector("query", top_k=100)
        assert len(results) == 3  # no -1 indices leak through

    def test_empty_query_returns_empty(self, small_index):
        assert vector_search.search_vector("") == []
        assert vector_search.search_vector("   ") == []

    def test_empty_index_returns_empty(self, monkeypatch):
        index = faiss.IndexFlatIP(3)
        monkeypatch.setattr(vector_search, "_faiss_index", index)
        monkeypatch.setattr(vector_search, "_chunks_metadata", [])
        monkeypatch.setattr(vector_search, "get_embedding", lambda text, api_key=None: [1.0, 0.0, 0.0])
        assert vector_search.search_vector("query") == []

    def test_missing_index_raises_clear_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(vector_search, "_faiss_index", None)
        monkeypatch.setattr(vector_search, "_INDEX_PATH", tmp_path / "nope.bin")
        with pytest.raises(FileNotFoundError, match="build_schema_index"):
            vector_search.search_vector("query")

class TestGetEmbedding:
    def test_missing_api_key_message(self, monkeypatch):
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        with pytest.raises(ValueError, match="FIREWORKS_API_KEY not found"):
            vector_search.get_embedding("text", api_key=None)
