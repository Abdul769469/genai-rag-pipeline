# GenAI RAG pipeline — practice project

A minimal but complete Retrieval-Augmented Generation (RAG) pipeline you can
run entirely in VS Code. It answers questions about a small set of sample
documents by retrieving relevant chunks and asking an LLM (served by
[Groq](https://console.groq.com), running Llama 3.3 70B) to answer using
that context — and includes an automated evaluation suite to measure how
well it actually works.

**Retrieval accuracy: 10/10 · Answer accuracy: 9/10** — see [Evaluation](#evaluation) below.

```
Documents -> Chunking -> Embeddings -> Vector index (FAISS)
                                              |
User question -> Retrieve top-k chunks -> Prompt Groq LLM -> Answer
                                              |
                                    Automated evaluation suite
```

## What each stage teaches you

| Stage | File | Concept |
|---|---|---|
| 1. Ingest | `src/ingest.py` | Loading documents, chunking text with overlap |
| 2. Index | `src/build_index.py` | Embeddings, vector similarity, FAISS |
| 3. Retrieve + Generate | `src/rag_pipeline.py` | Semantic search, prompt construction, calling an LLM API |
| 4. Interface (CLI) | `src/chat.py` | Tying it together into a usable command-line chat |
| 5. Interface (web) | `src/app.py` | Same pipeline behind a Streamlit browser UI |
| 6. Evaluation | `src/evaluate.py` | Automated scoring of retrieval and answer quality |

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

### 7. (Optional) Run the browser chat UI

```bash
pip install streamlit
streamlit run src/app.py
```

Opens the same pipeline as a chat interface at `http://localhost:8501`.

### 8. Run the evaluation suite

```bash
python src/evaluate.py
```

Runs 10 predefined test questions through the full pipeline automatically
and reports retrieval and answer accuracy. See [Evaluation](#evaluation)
below for details.

## Evaluation

Manually typing questions and eyeballing the answers doesn't scale, and it's
easy to miss regressions — a document can silently break retrieval and a
one-off chat session won't catch it. `src/evaluate.py` runs a fixed set of
10 question/answer pairs (`data/eval_questions.json`) through the full
pipeline and checks two things per question:

| Metric | What it checks |
|---|---|
| **Retrieval accuracy** | Did the expected source document show up among the retrieved chunks? |
| **Answer accuracy** | Does the generated answer contain the expected fact/keyword? |

**Latest run:**

| Metric | Score |
|---|---|
| Retrieval accuracy | **10 / 10 (100%)** |
| Answer accuracy | **10 / 10 (100%)** |

Run it yourself any time with:

```bash
python src/evaluate.py
```

Sample output:

```
[1/10] How long does the Acme Mover battery last?
    Retrieval: PASS  (expected company_faq.txt in [...])
    Answer:    PASS  (expected keyword '10 hours')
...
==================================================
SUMMARY
==================================================
Retrieval accuracy: 10/10 (100%)
Answer accuracy:    10/10 (100%)

All test cases passed.
```

This is a simplified, keyword-matching version of what's called "RAG
evaluation" in production systems. Real systems often use an LLM-as-judge
instead of exact keyword matching, and track additional metrics like
faithfulness, relevance, and latency — but the mechanics here (define
expected outcomes, run them automatically, get a pass/fail report) are the
same ones production eval pipelines are built on.

**To add your own test case**, add an entry to `data/eval_questions.json`:

```json
{
  "question": "Your question here",
  "expected_source": "the_file.txt",
  "expected_keyword": "a phrase that should appear in a correct answer"
}
```

## How to extend this for more practice

Once the basic loop works, try these upgrades in order — each teaches a
different real-world GenAI skill:

1. **Swap in your own documents.** Drop `.txt` files into
   `data/sample_docs/` and rerun `build_index.py`.
2. **Support PDFs/Word docs.** Add a loader using `pypdf` or `python-docx`
   in `ingest.py`.
3. **Add streaming responses.** Use streaming in `rag_pipeline.py` so
   answers appear token-by-token instead of all at once.
4. **Grow the evaluation suite.** Add more test cases to
   `data/eval_questions.json`, or replace exact keyword matching with an
   LLM-as-judge for more nuanced scoring.
5. **Try a hosted vector DB.** Swap FAISS for Chroma, Pinecone, or Qdrant
   to learn how production systems handle much larger document sets.
6. **Add conversation memory** so follow-up questions ("what about the
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
