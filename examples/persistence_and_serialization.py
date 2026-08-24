"""Demonstrates zero-dependency index serialization, persistence, and reloading.

In production RAG architectures, document chunking and vector embedding should happen
during offline indexing pipelines, with online query services loading pre-built indices.

Run with: python examples/persistence_and_serialization.py
"""

import tempfile
from pathlib import Path

from ragforge.chunking import SentenceChunker
from ragforge.pipeline import RagPipeline

DOCS = [
    ("arch", "Microservices communicate asynchronously via message queues and event buses."),
    ("tracing", "Distributed tracing captures latency metrics across service boundaries."),
    ("resilience", "Circuit breakers prevent cascading outages by shedding load when down."),
]


def main() -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        index_file = Path(tmpdir) / "rag_index.json"

        # 1. Build and persist index
        print("1. Indexing corpus and saving to disk...")
        indexing_pipeline = RagPipeline(chunker=SentenceChunker(max_chars=200))
        for doc_id, text in DOCS:
            indexing_pipeline.ingest(doc_id, text)

        indexing_pipeline.save(index_file)
        file_size = index_file.stat().st_size
        print(f"   Saved {indexing_pipeline.chunk_count} chunks ({file_size} bytes)")

        # 2. Reload index in a fresh pipeline without re-embedding
        print("\n2. Loading pre-built index into fresh serving pipeline...")
        serving_pipeline = RagPipeline.load(index_file)
        print(f"   Serving pipeline loaded with {serving_pipeline.chunk_count} chunks.")

        # 3. Query the loaded pipeline
        print("\n3. Querying loaded pipeline...")
        query = "How do circuit breakers and resilience mechanisms work?"
        answer = serving_pipeline.answer(query, k=1)

        print(f"   Query: {query}")
        print(f"   Answer: {answer.answer}")
        print(f"   Context Doc: {answer.contexts[0].doc_id}")

        return answer.answer


if __name__ == "__main__":
    main()
