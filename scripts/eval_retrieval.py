from __future__ import annotations

import time
from pathlib import Path

from app.pipeline import RAGPipeline


EVAL_CASES = [
    {
        "question": "Why does RAG reduce hallucinations?",
        "expected_terms": ["hallucinations", "trusted documents", "groundedness"],
    },
    {
        "question": "What metrics should production RAG track?",
        "expected_terms": ["precision", "faithfulness", "latency", "cost"],
    },
]


def evaluate() -> None:
    pipeline = RAGPipeline()
    sample_dir = Path(__file__).resolve().parents[1] / "sample_data"
    stats = pipeline.ingest_folder(str(sample_dir))
    print(f"Ingested files={stats['files_processed']} chunks={stats['chunks_created']}")

    hit_count = 0
    elapsed = []
    for case in EVAL_CASES:
        start = time.perf_counter()
        results = pipeline.retrieve(case["question"], top_k=3)
        elapsed.append(time.perf_counter() - start)

        joined = " ".join(r.text.lower() for r in results)
        matched = any(term in joined for term in case["expected_terms"])
        hit_count += int(matched)
        print(
            f"Q: {case['question']}\n"
            f"  retrieved={len(results)} matched_expected={matched}\n"
        )

    accuracy = hit_count / len(EVAL_CASES) if EVAL_CASES else 0.0
    avg_ms = (sum(elapsed) / len(elapsed) * 1000) if elapsed else 0.0
    print(f"Top-k hit rate: {accuracy:.2%}")
    print(f"Avg retrieval latency: {avg_ms:.2f} ms")


if __name__ == "__main__":
    evaluate()

