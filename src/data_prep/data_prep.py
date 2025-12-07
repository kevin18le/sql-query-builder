import faiss
import numpy as np
from pathlib import Path

_FILE_PATH = Path(__file__).parents[2]

def load_faiss_index():
    """
    Load pre-computed FAISS index and embeddings from disk.
    Returns:
        Tuple of (faiss_index, embeddings_array)
    """
    index = faiss.read_index(str(_FILE_PATH / "data" / "schema_index.bin"))
    embeddings = np.load(_FILE_PATH / "data" / "embeddings.npy")
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    return index, embeddings