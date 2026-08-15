"""
Entry point: a simple command-line chat loop over the RAG pipeline.

Run with:  python src/chat.py
"""

from rag_pipeline import RagPipeline


def main():
    pipeline = RagPipeline()
    print("\nRAG pipeline ready. Ask a question about Acme Robotics (or type 'exit').\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        result = pipeline.answer(query)

        print(f"\nAssistant: {result['answer']}")
        print(f"(sources: {', '.join(sorted(set(result['sources'])))})\n")


if __name__ == "__main__":
    main()
