"""Runs all example scripts as integration tests to guarantee they never silently rot."""

import importlib
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))


def _load(module_name: str):
    return importlib.import_module(module_name)


def test_basic_pipeline_example():
    module = _load("basic_pipeline")
    answer = module.main()
    assert "330" in answer or "Eiffel" in answer


def test_hybrid_vs_single_example():
    module = _load("hybrid_vs_single")
    results = module.main()
    assert results["exact_match"]["bm25_only"] == "c1"
    assert results["paraphrase"]["bm25_only"] is None
    assert results["paraphrase"]["hybrid"] == "c2"


def test_evaluate_suite_example():
    module = _load("evaluate_suite")
    results = module.main()
    assert len(results) == 2
    assert all(r.context_precision == 1.0 for r in results)


def test_metadata_filtering_and_mmr_example():
    module = _load("metadata_filtering_and_mmr")
    results = module.main()
    assert results["finance_count"] == 2
    assert results["diverse_count"] == 2


def test_persistence_and_serialization_example():
    module = _load("persistence_and_serialization")
    answer = module.main()
    assert "Circuit breakers" in answer
