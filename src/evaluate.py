"""
Stage 5: Evaluation
----------------------
Manually testing a RAG pipeline by typing questions and eyeballing the
answers doesn't scale, and it's easy to miss regressions (exactly what
happened earlier when a document silently got overwritten). This script
runs a fixed set of test questions automatically and checks two things
for each one:

  1. Retrieval accuracy: did the expected source file show up anywhere
     in the retrieved chunks?
  2. Answer accuracy: does the generated answer contain the expected
     keyword/fact?

This is a simplified version of what's called "RAG evaluation" in
production systems. Real systems often use an LLM-as-judge instead of
exact keyword matching, and track more nuanced metrics (faithfulness,
relevance, latency). This keyword-based version is intentionally simple
so the mechanics are easy to follow.

Run with:  python src/evaluate.py
"""

import json
from pathlib import Path

from rag_pipeline import RagPipeline

EVAL_FILE = Path(__file__).parent.parent / "data" / "eval_questions.json"


def load_test_cases() -> list[dict]:
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation():
    test_cases = load_test_cases()
    pipeline = RagPipeline()

    results = []
    print(f"\nRunning {len(test_cases)} test cases...\n")

    for i, case in enumerate(test_cases, start=1):
        question = case["question"]
        expected_source = case["expected_source"]
        expected_keyword = case["expected_keyword"]

        result = pipeline.answer(question)
        answer_text = result["answer"]
        retrieved_sources = set(result["sources"])

        retrieval_hit = expected_source in retrieved_sources
        # Case-insensitive keyword check against the generated answer
        answer_hit = expected_keyword.lower() in answer_text.lower()

        results.append(
            {
                "question": question,
                "retrieval_hit": retrieval_hit,
                "answer_hit": answer_hit,
                "expected_source": expected_source,
                "retrieved_sources": retrieved_sources,
                "expected_keyword": expected_keyword,
                "answer_text": answer_text,
            }
        )

        retrieval_mark = "PASS" if retrieval_hit else "FAIL"
        answer_mark = "PASS" if answer_hit else "FAIL"
        print(f"[{i}/{len(test_cases)}] {question}")
        print(f"    Retrieval: {retrieval_mark}  (expected {expected_source} in {sorted(retrieved_sources)})")
        print(f"    Answer:    {answer_mark}  (expected keyword '{expected_keyword}')")
        if not retrieval_hit or not answer_hit:
            print(f"    -> Answer was: {answer_text[:150]}")
        print()

    print_summary(results)


def print_summary(results: list[dict]):
    total = len(results)
    retrieval_passes = sum(r["retrieval_hit"] for r in results)
    answer_passes = sum(r["answer_hit"] for r in results)

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Retrieval accuracy: {retrieval_passes}/{total} ({100 * retrieval_passes / total:.0f}%)")
    print(f"Answer accuracy:    {answer_passes}/{total} ({100 * answer_passes / total:.0f}%)")

    failures = [r for r in results if not r["retrieval_hit"] or not r["answer_hit"]]
    if failures:
        print(f"\n{len(failures)} question(s) need attention:")
        for r in failures:
            print(f"  - {r['question']}")
    else:
        print("\nAll test cases passed.")


if __name__ == "__main__":
    run_evaluation()
