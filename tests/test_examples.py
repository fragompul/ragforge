"""Runs the example scripts as integration tests so they can't silently rot."""

import importlib
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))


def _load(module_name: str):
    return importlib.import_module(module_name)


def test_basic_pipeline_example_answers_from_relevant_context():
    module = _load("basic_pipeline")
    answer = module.main()
    assert "330" in answer or "Eiffel" in answer


def test_hybrid_vs_single_example_shows_bm25_missing_the_paraphrase():
    module = _load("hybrid_vs_single")
    results = module.main()
    assert results["exact_match"]["bm25_only"] == "c1"
    assert results["paraphrase"]["bm25_only"] is None
    assert results["paraphrase"]["hybrid"] == "c2"


def test_evaluate_suite_example_produces_scored_results():
    module = _load("evaluate_suite")
    results = module.main()
    assert len(results) == 2
    assert all(r.context_precision == 1.0 for r in results)
