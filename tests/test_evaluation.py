import tempfile
from pathlib import Path

from ragforge.chunking import FixedSizeChunker
from ragforge.evaluation import (
    EvalCase,
    EvaluationSummary,
    _context_f1,
    _context_precision,
    _context_recall,
    _faithfulness,
    _ranked_context_precision,
    evaluate_pipeline,
)
from ragforge.pipeline import RagPipeline


def _build_pipeline() -> RagPipeline:
    pipeline = RagPipeline(chunker=FixedSizeChunker(chunk_size=20, overlap=5))
    pipeline.ingest("doc1", "The Eiffel Tower is located in Paris, France, completed in 1889.")
    pipeline.ingest("doc2", "The Great Wall of China stretches thousands of km across China.")
    return pipeline


def test_evaluate_pipeline_returns_summary():
    pipeline = _build_pipeline()
    cases = [
        EvalCase(
            query="Where is the Eiffel Tower?",
            relevant_doc_ids=["doc1"],
            ground_truth_answer="Paris, France.",
        ),
        EvalCase(
            query="How long is the Great Wall?",
            relevant_doc_ids=["doc2"],
            ground_truth_answer="Thousands of kilometers across China.",
        ),
    ]
    summary = evaluate_pipeline(pipeline, cases, k=2)

    assert len(summary) == 2
    assert all(0.0 <= r.faithfulness <= 1.0 for r in summary)
    assert all(-1.0 <= r.answer_relevancy <= 1.0 for r in summary)
    assert summary.mean_faithfulness >= 0.0
    assert summary.mean_answer_relevancy >= -1.0
    assert summary.mean_context_precision is not None
    assert summary.mean_ranked_context_precision is not None
    assert summary.mean_context_recall is not None
    assert summary.mean_context_f1 is not None
    assert summary.mean_answer_similarity is not None
    assert summary.mean_latency_ms >= 0.0


def test_context_precision_and_recall_are_none_without_ground_truth():
    pipeline = _build_pipeline()
    summary = evaluate_pipeline(pipeline, [EvalCase(query="Eiffel Tower")], k=2)
    assert summary[0].context_precision is None
    assert summary[0].ranked_context_precision is None
    assert summary[0].context_recall is None
    assert summary[0].context_f1 is None
    assert summary[0].answer_similarity is None


def test_metric_edge_cases():
    assert _context_precision([], ["doc1"]) == 0.0
    assert _context_precision(["doc1"], []) is None
    assert _ranked_context_precision([], ["doc1"]) == 0.0
    assert _ranked_context_precision(["doc1"], []) is None
    assert _ranked_context_precision(["doc2"], ["doc1"]) == 0.0
    assert _context_recall([], ["doc1"]) == 0.0
    assert _context_recall(["doc1"], []) is None
    assert _context_f1(None, 0.5) is None
    assert _context_f1(0.0, 0.0) == 0.0
    assert _faithfulness("", ["some context"]) == 1.0


def test_empty_evaluation_summary():
    summary = EvaluationSummary([])
    assert len(summary) == 0
    assert summary.mean_faithfulness == 0.0
    assert summary.mean_context_precision is None
    assert summary.mean_context_recall is None


def test_summary_markdown_table_and_serialization():
    pipeline = _build_pipeline()
    cases = [
        EvalCase(
            query="Where is Eiffel?",
            relevant_doc_ids=["doc1"],
            ground_truth_answer="Paris, France",
        )
    ]
    summary = evaluate_pipeline(pipeline, cases, k=1)

    table = summary.to_markdown_table()
    assert "| Metric | Mean Score |" in table
    assert "**Overall Score**" in table
    assert "**Ranked Precision (MAP@k)**" in table
    assert "**Context Precision**" in table
    assert "**Context Recall**" in table
    assert "**Context F1**" in table
    assert "**Answer Similarity**" in table

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "eval_report.json"
        summary.save(out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_overall_averages_available_metrics():
    pipeline = _build_pipeline()
    summary = evaluate_pipeline(
        pipeline,
        [
            EvalCase(
                query="Eiffel Tower Paris",
                relevant_doc_ids=["doc1"],
                ground_truth_answer="Eiffel Tower in Paris",
            )
        ],
        k=1,
    )
    result = summary[0]
    parts = [
        result.faithfulness,
        result.answer_relevancy,
        result.context_precision,
        result.context_recall,
        result.answer_similarity,
    ]
    expected = sum(p for p in parts if p is not None) / len([p for p in parts if p is not None])
    assert result.overall == expected
