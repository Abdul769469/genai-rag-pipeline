"""
Stage 3: Retrieval + Generation
---------------------------------
Given a user question:
  1. Embed the question with the same model used to embed the chunks.
  2. Search the FAISS index for the most similar chunks (retrieval).
  3. Stuff those chunks into a prompt as context.
  4. Send the prompt to Claude and return the generated answer.

This is the "RAG" pattern (Retrieval-Augmented Generation): instead of
relying only on what the LLM memorized during training, we hand it fresh,
specific context at request time. It's the pattern behind most real-world
"chat with your docs" products.
"""

import os
import pickle

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from build_index import EMBEDDING_MODEL_NAME, INDEX_PATH, METADATA_PATH

load_dotenv()  # reads GROQ_API_KEY from a local .env file

from groq import Groq

# Groq hosts several open models with very fast inference.
# See https://console.groq.com/docs/models for the current list.
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5  # how many chunks to retrieve per question


class RagPipeline:
    def __init__(self):
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                "No index found. Run `python src/build_index.py` first."
            )
        print("Loading embedding model and index...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(METADATA_PATH, "rb") as f:
            self.chunks = pickle.load(f)

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
            )
        self.client = Groq(api_key=api_key)

    def retrieve(self, query: str, top_k: int = TOP_K):
        """Embed the query and return the top_k most similar chunks."""
        query_vec = self.embed_model.encode([query], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype="float32")
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append((chunk, float(score)))
        return results

    def build_prompt(self, query: str, retrieved_chunks) -> str:
        context_blocks = "\n\n".join(
            f"[Source: {c.source}]\n{c.text}" for c, _ in retrieved_chunks
        )
        return f"""You are a helpful assistant answering questions using only the
provided context. If the answer isn't in the context, say you don't know
rather than guessing.

Context:
{context_blocks}

Question: {query}

Answer concisely, and mention which source(s) you used."""

    def answer(self, query: str, top_k: int = TOP_K) -> dict:
        retrieved = self.retrieve(query, top_k)
        prompt = self.build_prompt(query, retrieved)

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        answer_text = response.choices[0].message.content
        return {
            "answer": answer_text,
            "sources": [c.source for c, _ in retrieved],
            "retrieved_chunks": retrieved,
        }
