"""Loaders for pre-computed retrieval artifacts (FAISS index + embeddings).

Note: the retrieval layer (``src.search.vector_search``) loads the index
lazily on first search; this loader is a convenience for scripts and tests.
"""

from pathlib import Path
from typing import Tuple

import faiss
import numpy as np

_FILE_PATH = Path(__file__).parents[2]


def load_faiss_index() -> Tuple[faiss.Index, np.ndarray]:
    """Load the pre-computed FAISS index and embeddings from disk.

    Returns:
        Tuple of (faiss_index, embeddings_array).

    Raises:
        FileNotFoundError: If the index or embeddings file is missing,
            with instructions for generating them.
    """
    index_path = _FILE_PATH / "data" / "schema_index.bin"
    embeddings_path = _FILE_PATH / "data" / "embeddings.npy"

    missing = [p.name for p in (index_path, embeddings_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required file(s): {', '.join(missing)} in {_FILE_PATH / 'data'}. "
            "Generate them with: python -m src.data_prep.build_schema_index"
        )

    index = faiss.read_index(str(index_path))
    embeddings = np.load(str(embeddings_path))
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    return index, embeddings
