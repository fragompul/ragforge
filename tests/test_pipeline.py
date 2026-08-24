import tempfile
from pathlib import Path

import pytest

from ragforge.chunking import FixedSizeChunker
from ragforge.pipeline import RagPipeline, default_prompt_formatter
from ragforge.reranking import HeuristicReranker


def _build_pipeline() -> RagPipeline:
    pipeline = RagPipeline(
        chunker=FixedSizeChunker(chunk_size=20, overlap=5),
        reranker=HeuristicReranker(),
    )
    pipeline.ingest(
        "doc1",
        "The Eiffel Tower is located in Paris, France, completed in 1889.",
        metadata={"category": "monuments", "public": True},
    )
    pipeline.ingest(
        "doc2",
        "The Great Wall of China stretches thousands of km across China.",
        metadata={"category": "monuments", "public": False},
    )
    return pipeline


def test_ingest_and_counts():
    pipeline = _build_pipeline()
    assert pipeline.document_count == 2
    assert pipeline.chunk_count > 0


def test_batch_ingestion():
    pipeline = RagPipeline()
    docs = [
        {"id": "d1", "text": "First batch document on computing.", "metadata": {"tag": "tech"}},
        {"doc_id": "d2", "text": "Second doc on physics.", "metadata": {"tag": "science"}},
    ]
    chunks = pipeline.ingest_batch(docs)
    assert len(chunks) == 2
    assert pipeline.document_count == 2


def test_batch_ingestion_missing_id():
    pipeline = RagPipeline()
    with pytest.raises(ValueError, match="must contain 'id' or 'doc_id'"):
        pipeline.ingest_batch([{"text": "no id"}])


def test_delete_and_clear():
    pipeline = _build_pipeline()
    removed = pipeline.delete("doc1")
    assert removed > 0
    assert pipeline.document_count == 1

    pipeline.clear()
    assert pipeline.document_count == 0
    assert pipeline.chunk_count == 0


def test_retrieve_with_metadata_filter():
    pipeline = _build_pipeline()
    # Retrieve only public docs
    results = pipeline.retrieve(
        "monuments and locations",
        k=5,
        filter_fn=lambda c: c.metadata.get("public") is True,
    )
    assert len(results) > 0
    assert all(r.metadata.get("public") is True for r in results)
    assert all(r.doc_id == "doc1" for r in results)

    # Test k <= 0
    assert pipeline.retrieve("test", k=0) == []


def test_answer_latencies_and_metadata():
    pipeline = _build_pipeline()
    answer = pipeline.answer("Where is the Eiffel Tower?", k=2)

    assert answer.query == "Where is the Eiffel Tower?"
    assert answer.answer
    assert len(answer.contexts) > 0
    assert answer.retrieval_latency_ms >= 0.0
    assert answer.generation_latency_ms >= 0.0
    assert answer.total_latency_ms >= 0.0
    assert "--- CONTEXTS ---" in answer.prompt

    ans_dict = answer.to_dict()
    assert ans_dict["query"] == answer.query
    assert len(ans_dict["contexts"]) == len(answer.contexts)


def test_custom_prompt_formatter_and_generator():
    def custom_formatter(query: str, contexts: list[str]) -> str:
        return f"CUSTOM_PROMPT: {query} with {len(contexts)} ctxs"

    def custom_generate(query: str, contexts: list[str]) -> str:
        return f"ANSWERED:{query}:{len(contexts)}"

    pipeline = RagPipeline(
        prompt_formatter=custom_formatter,
        generate_fn=custom_generate,
    )
    pipeline.ingest("doc1", "Some arbitrary text for custom generation testing.")

    answer = pipeline.answer("test query", k=1)
    assert answer.prompt.startswith("CUSTOM_PROMPT:")
    assert answer.answer.startswith("ANSWERED:test query:1")


def test_default_prompt_formatter_empty():
    prompt = default_prompt_formatter("What is RAG?", [])
    assert "No context provided" in prompt


def test_pipeline_serialization_roundtrip():
    pipeline = _build_pipeline()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "pipeline.json"
        pipeline.save(path)

        loaded = RagPipeline.load(path, reranker=HeuristicReranker())
        assert loaded.document_count == 2
        assert loaded.chunk_count == pipeline.chunk_count

        ans = loaded.answer("Eiffel Tower Paris", k=1)
        assert ans.contexts[0].doc_id == "doc1"
