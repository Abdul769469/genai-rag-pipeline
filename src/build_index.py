"""
Stage 2: Embedding + Indexing
-------------------------------
Turn each text chunk into a numeric vector (an "embedding") using a local
open-source model (no API key or cost needed for this step), then store
those vectors in a FAISS index for fast similarity search.

Why embeddings? Two pieces of text that mean similar things end up as
vectors that are close together in space, even if they don't share the
same words. That's what lets us find "battery life" chunks when the user
asks "how long does the robot run before charging?"
"""

import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import build_chunks

# A small, fast, well-regarded open-source embedding model.
# Runs locally on CPU — no API key required.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_DIR = Path(__file__).parent.parent / "index_store"
INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "chunks.pkl"


def build_and_save_index(data_dir: str = "data/sample_docs") -> None:
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads it)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Chunking documents...")
    chunks = build_chunks(data_dir)
    print(f"  -> {len(chunks)} chunks")

    print("Generating embeddings...")
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    # Inner product on normalized vectors == cosine similarity search
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved chunk metadata to {METADATA_PATH}")


if __name__ == "__main__":
    build_and_save_index()
