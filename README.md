# GenAI RAG pipeline — practice project

A minimal but complete Retrieval-Augmented Generation (RAG) pipeline you can
run entirely in VS Code. It answers questions about a small set of sample
documents by retrieving relevant chunks and asking an LLM (served by
[Groq](https://console.groq.com), running Llama 3.3 70B) to answer using
that context.

```
Documents -> Chunking -> Embeddings -> Vector index (FAISS)
                                              |
User question -> Retrieve top-k chunks -> Prompt Groq LLM -> Answer
```

## What each stage teaches you

| Stage | File | Concept |
|---|---|---|
| 1. Ingest | `src/ingest.py` | Loading documents, chunking text with overlap |
| 2. Index | `src/build_index.py` | Embeddings, vector similarity, FAISS |
| 3. Retrieve + Generate | `src/rag_pipeline.py` | Semantic search, prompt construction, calling an LLM API |
| 4. Interface | `src/chat.py` | Tying it together into a usable CLI |

## Step-by-step setup in VS Code

### 1. Open the project
Unzip/copy this folder somewhere on your machine, then in VS Code:
`File > Open Folder...` and select `genai-rag-pipeline`.

If you don't have it yet, install the **Python extension** from the VS Code
Extensions panel (Ctrl+Shift+X, search "Python", by Microsoft).

### 2. Create a virtual environment
Open a terminal in VS Code (`` Ctrl+` ``) and run:

```bash
python -m venv .venv
```

Activate it:
- macOS/Linux: `source .venv/bin/activate`
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`

VS Code will usually prompt "Select this environment as your workspace
interpreter?" — click **Yes**. You can also pick it manually via
`Ctrl+Shift+P` -> "Python: Select Interpreter" -> choose the `.venv` one.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs `sentence-transformers` (local embeddings), `faiss-cpu`
(vector search), and `groq` (Groq API client). First install can take a
few minutes since it pulls in PyTorch.

### 4. Add your API key
Get a free key from https://console.groq.com/keys, then:

```bash
cp .env.example .env
```

Open `.env` and paste your key in place of `your-api-key-here`.

### 5. Build the vector index
This reads the sample documents in `data/sample_docs/`, chunks them,
generates embeddings, and saves a FAISS index to disk. It only needs to be
re-run when your source documents change.

```bash
python src/build_index.py
```

The first run downloads the embedding model (~90MB) — that's normal.

### 6. Chat with your documents

```bash
python src/chat.py
```

Try asking:
- "How long does the battery last?"
- "What safety certifications does the robot have?"
- "What's the return policy?"

Type `exit` to quit.

## How to extend this for more practice

Once the basic loop works, try these upgrades in order — each teaches a
different real-world GenAI skill:

1. **Swap in your own documents.** Drop `.txt` files into
   `data/sample_docs/` and rerun `build_index.py`.
2. **Support PDFs/Word docs.** Add a loader using `pypdf` or `python-docx`
   in `ingest.py`.
3. **Add streaming responses.** Use `client.messages.stream(...)` in
   `rag_pipeline.py` so answers appear token-by-token.
4. **Add a web UI.** Wrap `RagPipeline` in a `streamlit` app
   (`pip install streamlit`) for a chat interface in the browser.
5. **Add evaluation.** Write a small set of question/expected-answer pairs
   and score retrieval quality (did the right chunk get retrieved?) and
   answer quality (does the LLM's answer match the expected facts?).
6. **Try a hosted vector DB.** Swap FAISS for Chroma, Pinecone, or Qdrant
   to learn how production systems handle much larger document sets.
7. **Add conversation memory** so follow-up questions ("what about the
   larger model?") resolve correctly using chat history.

## Troubleshooting

- `GROQ_API_KEY not set` — make sure you copied `.env.example` to
  `.env` (not just edited the example) and that VS Code's terminal is in
  the project root when you run scripts.
- Rate limit errors — Groq's free tier has generous but finite requests-
  per-minute limits; wait a few seconds and retry, or check current limits
  at https://console.groq.com/settings/limits.
- `No index found` — run `python src/build_index.py` before `chat.py`.
- Import errors — confirm the `.venv` interpreter is selected
  (bottom-right corner of VS Code should show it) and dependencies
  installed without errors.
