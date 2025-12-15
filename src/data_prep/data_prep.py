import faiss
import numpy as np
import sys
from pathlib import Path

_FILE_PATH = Path(__file__).parents[2]

def load_faiss_index():
    """
    Load pre-computed FAISS index and embeddings from disk.
    Returns:
        Tuple of (faiss_index, embeddings_array)
    """
    index_path = _FILE_PATH / "data" / "schema_index.bin"
    embeddings_path = _FILE_PATH / "data" / "embeddings.npy"
    
    # Check if files exist
    if not index_path.exists() or not embeddings_path.exists():
        missing_files = []
        if not index_path.exists():
            missing_files.append("schema_index.bin")
        if not embeddings_path.exists():
            missing_files.append("embeddings.npy")
        
        print("\n" + "="*70, file=sys.stderr)
        print("⚠️  WARNING: Required index and/or embeddings files not found!", file=sys.stderr)
        print("="*70, file=sys.stderr)
        print(f"\nMissing files: {', '.join(missing_files)}", file=sys.stderr)
        print(f"Expected location: {_FILE_PATH / 'data'}", file=sys.stderr)
        print("\nTo generate these files, please run:", file=sys.stderr)
        print("  python src/data_prep/build_schema_index.py", file=sys.stderr)
        print("  # or as a module:", file=sys.stderr)
        print("  python -m src.data_prep.build_schema_index", file=sys.stderr)
        print("\n" + "="*70 + "\n", file=sys.stderr)
        sys.exit(1)
    
    index = faiss.read_index(str(index_path))
    embeddings = np.load(str(embeddings_path))
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    return index, embeddings