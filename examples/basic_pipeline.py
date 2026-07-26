"""End-to-end pipeline: ingest documents, ask a question, inspect contexts.

Run with: python examples/basic_pipeline.py
"""

from ragforge.chunking import SentenceChunker
from ragforge.pipeline import RagPipeline
from ragforge.reranking import HeuristicReranker

DOCS = {
    "eiffel_tower": (
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
        "It was designed by Gustave Eiffel and completed in 1889. "
        "It stands 330 meters tall and was the world's tallest structure until 1930."
    ),
    "great_wall": (
        "The Great Wall of China is a series of fortifications built across "
        "northern China. Construction began over 2,000 years ago. "
        "It stretches for more than 20,000 kilometers."
    ),
}


def main() -> str:
    pipeline = RagPipeline(
        chunker=SentenceChunker(max_chars=200, overlap_sentences=1),
        reranker=HeuristicReranker(),
    )
    for doc_id, text in DOCS.items():
        pipeline.ingest(doc_id, text)

    rag_answer = pipeline.answer("How tall is the Eiffel Tower?", k=2)

    print(f"Query: {rag_answer.query}")
    print(f"Answer: {rag_answer.answer}\n")
    print("Retrieved contexts:")
    for context in rag_answer.contexts:
        print(f"  [{context.doc_id}] score={context.score:.3f} :: {context.text}")

    return rag_answer.answer


if __name__ == "__main__":
    main()
