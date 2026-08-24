"""Run RAGAS-style automatic evaluation over a small QA suite with markdown reporting.

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
    EvalCase(
        query="How tall is the Eiffel Tower?",
        relevant_doc_ids=["eiffel_tower"],
        ground_truth_answer="The Eiffel Tower is 330 meters tall.",
    ),
    EvalCase(
        query="How long is the Great Wall of China?",
        relevant_doc_ids=["great_wall"],
        ground_truth_answer="The Great Wall stretches for more than 20,000 kilometers.",
    ),
]


def main() -> list:
    pipeline = RagPipeline(chunker=SentenceChunker(max_chars=200, overlap_sentences=1))
    for doc_id, text in DOCS.items():
        pipeline.ingest(doc_id, text)

    summary = evaluate_pipeline(pipeline, CASES, k=1)

    print("=== Individual Case Results ===")
    for result in summary:
        print(
            f"Q: {result.query}\n"
            f"  faithfulness={result.faithfulness:.2f} "
            f"answer_relevancy={result.answer_relevancy:.2f} "
            f"context_precision={result.context_precision} "
            f"ranked_precision={result.ranked_context_precision} "
            f"context_recall={result.context_recall} "
            f"overall={result.overall:.2f}"
        )

    print("\n=== Summary Report ===")
    print(summary.to_markdown_table())

    return list(summary)


if __name__ == "__main__":
    main()
