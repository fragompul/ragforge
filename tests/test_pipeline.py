from ragforge.chunking import FixedSizeChunker
from ragforge.pipeline import RagPipeline
from ragforge.reranking import HeuristicReranker


def _build_pipeline() -> RagPipeline:
    pipeline = RagPipeline(
        chunker=FixedSizeChunker(chunk_size=20, overlap=5),
        reranker=HeuristicReranker(),
    )
    pipeline.ingest("doc1", "The Eiffel Tower is located in Paris, France, completed in 1889.")
    pipeline.ingest("doc2", "The Great Wall of China stretches thousands of km across China.")
    return pipeline


def test_ingest_returns_chunks_and_populates_indices():
    pipeline = _build_pipeline()
    assert len(pipeline.bm25_index._chunks) > 0
    assert len(pipeline.vector_index._chunks) > 0


def test_retrieve_finds_relevant_chunk_for_query():
    pipeline = _build_pipeline()
    results = pipeline.retrieve("Where is the Eiffel Tower?", k=2)
    assert any("Paris" in r.text for r in results)


def test_answer_uses_default_extractive_generator():
    pipeline = _build_pipeline()
    rag_answer = pipeline.answer("Where is the Eiffel Tower?", k=2)
    assert rag_answer.query == "Where is the Eiffel Tower?"
    assert rag_answer.answer  # non-empty
    assert len(rag_answer.contexts) > 0


def test_answer_falls_back_when_nothing_retrieved():
    pipeline = RagPipeline(chunker=FixedSizeChunker())
    rag_answer = pipeline.answer("anything", k=3)
    assert rag_answer.answer == "I don't have enough information to answer that."
    assert rag_answer.contexts == []


def test_custom_generate_fn_is_used():
    def echo_generate(query: str, contexts: list[str]) -> str:
        return f"custom:{len(contexts)}"

    pipeline = RagPipeline(
        chunker=FixedSizeChunker(chunk_size=20, overlap=5),
        generate_fn=echo_generate,
    )
    pipeline.ingest("doc1", "Some content about mountains and rivers and forests.")
    rag_answer = pipeline.answer("mountains", k=3)
    assert rag_answer.answer.startswith("custom:")
