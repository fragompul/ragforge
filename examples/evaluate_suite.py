"""Run RAGAS-style automatic evaluation over a small QA suite.

Run with: python examples/evaluate_suite.py
"""

from ragforge.chunking import SentenceChunker
from ragforge.evaluation import EvalCase, evaluate_pipeline
from ragforge.pipeline import RagPipeline

DOCS = {
    "eiffel_tower": (
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
        "It was completed in 1889 and stands 330 meters tall."
    ),
    "great_wall": (
        "The Great Wall of China is a series of fortifications built across "
        "northern China. It stretches for more than 20,000 kilometers."
    ),
}

CASES = [
    EvalCase(query="How tall is the Eiffel Tower?", relevant_doc_ids=["eiffel_tower"]),
    EvalCase(query="How long is the Great Wall of China?", relevant_doc_ids=["great_wall"]),
]


def main() -> list:
    pipeline = RagPipeline(chunker=SentenceChunker(max_chars=200, overlap_sentences=1))
    for doc_id, text in DOCS.items():
        pipeline.ingest(doc_id, text)

    results = evaluate_pipeline(pipeline, CASES, k=1)

    for result in results:
        print(
            f"Q: {result.query}\n"
            f"  faithfulness={result.faithfulness:.2f} "
            f"answer_relevancy={result.answer_relevancy:.2f} "
            f"context_precision={result.context_precision} "
            f"context_recall={result.context_recall} "
            f"overall={result.overall:.2f}"
        )
    return results


if __name__ == "__main__":
    main()
