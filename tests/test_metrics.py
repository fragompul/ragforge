"""Metrics are checked against hand-computed values and known invariants."""

from __future__ import annotations

import math

import pytest

from ragforge.metrics import (
    average_precision,
    dcg_at_k,
    mean_metric,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

REL = {"a": 3, "b": 2, "c": 1}


def test_dcg_matches_hand_computation():
    # rel / log2(rank+1): 3/1 + 0 + 2/2 + 0 + 0
    assert dcg_at_k(["a", "x", "b", "y", "z"], REL, 5) == pytest.approx(3.0 + 1.0)


def test_ndcg_is_one_for_ideal_order_and_lower_otherwise():
    assert ndcg_at_k(["a", "b", "c"], REL, 3) == pytest.approx(1.0)
    swapped = ndcg_at_k(["c", "b", "a"], REL, 3)
    ideal = 3 + 2 / math.log2(3) + 1 / 2
    worst = 1 + 2 / math.log2(3) + 3 / 2
    assert swapped == pytest.approx(worst / ideal) and swapped < 1.0


def test_ndcg_respects_cutoff_and_binary_relevance():
    assert ndcg_at_k(["x", "a"], {"a": 1}, 1) == 0.0
    assert ndcg_at_k(["x", "a"], {"a": 1}, 2) == pytest.approx(1 / math.log2(3))


def test_recall_precision_mrr_ap():
    ranked = ["x", "a", "y", "b"]
    rel = {"a": 1, "b": 1, "c": 1}
    assert recall_at_k(ranked, rel, 4) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, rel, 1) == 0.0
    assert precision_at_k(ranked, rel, 4) == pytest.approx(0.5)
    assert mrr_at_k(ranked, rel, 10) == pytest.approx(0.5)
    assert mrr_at_k(ranked, rel, 1) == 0.0
    # AP = (1/2 + 2/4) / 3 relevant
    assert average_precision(ranked, rel, 4) == pytest.approx((0.5 + 0.5) / 3)


def test_duplicates_in_ranking_do_not_inflate_recall():
    assert recall_at_k(["a", "a", "a"], {"a": 1, "b": 1}, 3) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "fn", [ndcg_at_k, recall_at_k, precision_at_k, mrr_at_k, average_precision]
)
def test_queries_without_relevant_docs_score_zero(fn):
    assert fn(["a", "b"], {}, 5) == 0.0
    assert fn(["a", "b"], {"a": 0}, 5) == 0.0


@pytest.mark.parametrize("fn", [ndcg_at_k, recall_at_k, precision_at_k, mrr_at_k, dcg_at_k])
def test_non_positive_k_rejected(fn):
    with pytest.raises(ValueError, match="positive"):
        fn(["a"], {"a": 1}, 0)


def test_mean_metric_averages_over_qrels_and_penalizes_missing_runs():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    runs = {"q1": ["a"]}  # q2 has no run -> 0
    assert mean_metric(runs, qrels, "recall", 1) == pytest.approx(0.5)
    assert mean_metric(runs, qrels, "ndcg", 10) == pytest.approx(0.5)
    assert mean_metric({}, {}, "ndcg") == 0.0
    for metric in ("precision", "mrr", "map"):
        assert 0.0 <= mean_metric(runs, qrels, metric, 5) <= 1.0
    with pytest.raises(ValueError, match="metric"):
        mean_metric(runs, qrels, "bleu")


# ---- regression tests added after mutation testing (defaults, zero-gain entries, argument passing)


def test_default_cutoffs_are_ten_for_ranked_metrics():
    ranked = [f"x{i}" for i in range(10)] + ["hit"]  # first relevant doc sits at rank 11
    rel = {"hit": 1.0}
    assert recall_at_k(ranked, rel) == 0.0
    assert mrr_at_k(ranked, rel) == 0.0
    assert ndcg_at_k(ranked, rel) == 0.0
    assert precision_at_k(ranked, rel) == 0.0
    assert recall_at_k(ranked, rel, k=11) == 1.0


def test_average_precision_ignores_explicit_zero_relevance_entries():
    ranked = ["a", "x", "b"]
    with_zeros = {"a": 1, "b": 1, "z": 0, "y": 0}
    assert average_precision(ranked, with_zeros, 3) == pytest.approx(
        average_precision(ranked, {"a": 1, "b": 1}, 3)
    )


def test_mean_metric_defaults_and_k_are_forwarded():
    qrels = {"q": {"d11": 1.0}}
    runs = {"q": [f"d{i}" for i in range(1, 12)]}  # relevant doc at rank 11
    assert mean_metric(runs, qrels) == 0.0  # default metric ndcg, default k=10
    assert mean_metric(runs, qrels, "ndcg", k=11) > 0.0
    assert mean_metric(runs, qrels, "map", k=11) == pytest.approx(1 / 11)
    assert mean_metric(runs, qrels, "map", k=10) == 0.0
    assert mean_metric({"q": ["d11"]}, qrels, "mrr", k=5) == 1.0


def test_average_precision_default_cutoff_is_one_hundred():
    ranked = [f"x{i}" for i in range(100)] + ["hit"]  # relevant doc at rank 101
    assert average_precision(ranked, {"hit": 1.0}) == 0.0
    assert average_precision(ranked, {"hit": 1.0}, k=101) == pytest.approx(1 / 101)
