"""
Stage 1: Ingestion
-------------------
Load raw text documents from disk and split them into overlapping chunks.

Why chunk at all?
LLMs (and embedding models) work on limited-size windows of text. If you
embed a whole 10-page document as one vector, you lose fine-grained detail
and retrieval becomes vague. Splitting into small, overlapping chunks lets
us retrieve just the most relevant paragraph(s) for a given question.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str
    source: str
    text: str


def load_documents(data_dir: str) -> list[tuple[str, str]]:
    """Read every .txt file in data_dir. Returns list of (filename, full_text)."""
    docs = []
    for path in sorted(Path(data_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append((path.name, text))
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks by character count.

    chunk_size: max characters per chunk
    overlap: how many characters the end of one chunk repeats at the
             start of the next, so we don't cut a sentence in half and
             lose the surrounding context.
    """
    text = " ".join(text.split())  # normalize whitespace
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # step forward, but re-include the overlap
    return chunks


def build_chunks(data_dir: str, chunk_size: int = 500, overlap: int = 100) -> list[Chunk]:
    """Load all documents and return a flat list of Chunk objects, each tagged
    with a unique id and its source filename (used later for citations)."""
    all_chunks: list[Chunk] = []
    for filename, text in load_documents(data_dir):
        pieces = chunk_text(text, chunk_size, overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(id=f"{filename}::{i}", source=filename, text=piece)
            )
    return all_chunks


if __name__ == "__main__":
    # Quick manual test: python src/ingest.py
    chunks = build_chunks("data/sample_docs")
    print(f"Loaded {len(chunks)} chunks from data/sample_docs\n")
    for c in chunks[:3]:
        print(f"--- {c.id} ---")
        print(c.text[:150], "...\n")
