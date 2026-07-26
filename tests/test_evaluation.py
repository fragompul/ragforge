from ragforge.chunking import FixedSizeChunker
from ragforge.evaluation import EvalCase, evaluate_pipeline
from ragforge.pipeline import RagPipeline


def _build_pipeline() -> RagPipeline:
    pipeline = RagPipeline(chunker=FixedSizeChunker(chunk_size=20, overlap=5))
    pipeline.ingest("doc1", "The Eiffel Tower is located in Paris, France, completed in 1889.")
    pipeline.ingest("doc2", "The Great Wall of China stretches thousands of km across China.")
    return pipeline


def test_evaluate_pipeline_returns_one_result_per_case():
    pipeline = _build_pipeline()
    cases = [
        EvalCase(query="Where is the Eiffel Tower?", relevant_doc_ids=["doc1"]),
        EvalCase(query="How long is the Great Wall?", relevant_doc_ids=["doc2"]),
    ]
    results = evaluate_pipeline(pipeline, cases, k=2)
    assert len(results) == 2
    assert all(0.0 <= r.faithfulness <= 1.0 for r in results)
    assert all(-1.0 <= r.answer_relevancy <= 1.0 for r in results)


def test_context_precision_and_recall_are_none_without_ground_truth():
    pipeline = _build_pipeline()
    results = evaluate_pipeline(pipeline, [EvalCase(query="Eiffel Tower")], k=2)
    assert results[0].context_precision is None
    assert results[0].context_recall is None


def test_context_precision_and_recall_computed_with_ground_truth():
    pipeline = _build_pipeline()
    results = evaluate_pipeline(
        pipeline, [EvalCase(query="Eiffel Tower Paris", relevant_doc_ids=["doc1"])], k=1
    )
    result = results[0]
    assert result.context_precision is not None
    assert result.context_recall is not None


def test_faithfulness_is_high_for_extractive_default_generator():
    # The default generator returns a context verbatim, so its content is
    # by construction fully grounded in the retrieved contexts.
    pipeline = _build_pipeline()
    results = evaluate_pipeline(pipeline, [EvalCase(query="Eiffel Tower")], k=1)
    assert results[0].faithfulness == 1.0


def test_overall_averages_available_metrics():
    pipeline = _build_pipeline()
    results = evaluate_pipeline(
        pipeline, [EvalCase(query="Eiffel Tower Paris", relevant_doc_ids=["doc1"])], k=1
    )
    result = results[0]
    parts = [
        result.faithfulness,
        result.answer_relevancy,
        result.context_precision,
        result.context_recall,
    ]
    assert result.overall == sum(parts) / len(parts)
