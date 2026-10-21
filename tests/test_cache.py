"""Tests for the semantic cache and the cached pipeline wrapper."""

from __future__ import annotations

import pytest

from ragforge import RagPipeline
from ragforge.cache import CachedPipeline, SemanticCache
from ragforge.chunking import RecursiveCharacterChunker


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_hit_above_threshold_and_miss_below():
    cache = SemanticCache(threshold=0.8)
    cache.put("how tall is the eiffel tower", "A")

    assert cache.get("how tall is the eiffel tower").value == "A"
    assert cache.get("tower eiffel tall how is the").value == "A"  # same bag of words
    assert cache.get("who created python") is None
    assert (cache.stats.hits, cache.stats.misses) == (2, 1)
    assert cache.stats.hit_rate == pytest.approx(2 / 3)


def test_partition_isolation():
    cache = SemanticCache()
    cache.put("same question", "tenant-a", partition=("a",))
    assert cache.get("same question", partition=("b",)) is None
    assert cache.get("same question", partition=("a",)).value == "tenant-a"


def test_lru_eviction_keeps_recently_used():
    cache = SemanticCache(max_size=2)
    cache.put("alpha alpha", 1)
    cache.put("beta beta", 2)
    assert cache.get("alpha alpha") is not None  # refresh alpha
    cache.put("gamma gamma", 3)  # evicts beta

    assert cache.get("beta beta") is None
    assert cache.get("alpha alpha") is not None
    assert cache.stats.evictions == 1 and len(cache) == 2


def test_ttl_expiry():
    clock = Clock()
    cache = SemanticCache(ttl_seconds=10, clock=clock)
    cache.put("question one", "A")
    clock.now = 9
    assert cache.get("question one") is not None
    clock.now = 20
    assert cache.get("question one") is None
    assert cache.stats.expirations == 1 and len(cache) == 0


def test_best_match_wins_among_candidates():
    cache = SemanticCache(threshold=0.5)
    cache.put("eiffel tower height", "tower")
    cache.put("python language creator", "python")
    assert cache.get("height of the eiffel tower").value == "tower"


@pytest.mark.parametrize(
    "kwargs", [{"threshold": 2}, {"max_size": 0}, {"ttl_seconds": 0}, {"ttl_seconds": -1}]
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SemanticCache(**kwargs)


def test_clear_keeps_stats():
    cache = SemanticCache()
    cache.put("q", 1)
    cache.get("q")
    cache.clear()
    assert len(cache) == 0 and cache.stats.hits == 1


def _pipeline():
    calls = {"n": 0}

    def generate(query: str, contexts: list[str]) -> str:
        calls["n"] += 1
        return f"answer-{calls['n']}"

    pipeline = RagPipeline(
        chunker=RecursiveCharacterChunker(chunk_size=200, chunk_overlap=0), generate_fn=generate
    )
    pipeline.ingest("eiffel", "The Eiffel Tower is in Paris. It was completed in 1889.")
    return pipeline, calls


def test_cached_pipeline_serves_repeat_without_generation():
    pipeline, calls = _pipeline()
    cached = CachedPipeline(pipeline)

    first = cached.answer("When was the Eiffel Tower completed?", k=2)
    second = cached.answer("When was the Eiffel Tower completed?", k=2)

    assert calls["n"] == 1
    assert first.metadata["cache"] == "miss" and second.metadata["cache"] == "hit"
    assert second.answer == first.answer and second.total_latency_ms == 0.0
    assert second.metadata["cached_query"] == "When was the Eiffel Tower completed?"


def test_different_k_is_a_different_cache_partition():
    pipeline, calls = _pipeline()
    cached = CachedPipeline(pipeline)
    cached.answer("eiffel tower", k=1)
    cached.answer("eiffel tower", k=3)
    assert calls["n"] == 2


def test_filtered_calls_bypass_unless_keyed():
    pipeline, calls = _pipeline()
    cached = CachedPipeline(pipeline)
    only_eiffel = lambda chunk: chunk.doc_id == "eiffel"  # noqa: E731

    cached.answer("eiffel tower", filter_fn=only_eiffel)
    cached.answer("eiffel tower", filter_fn=only_eiffel)
    assert calls["n"] == 2 and cached.cache.stats.bypasses == 2

    cached.answer("eiffel tower", filter_fn=only_eiffel, cache_key="only-eiffel")
    cached.answer("eiffel tower", filter_fn=only_eiffel, cache_key="only-eiffel")
    assert calls["n"] == 3


def test_corpus_mutation_invalidates_cache():
    pipeline, calls = _pipeline()
    cached = CachedPipeline(pipeline)
    cached.answer("eiffel tower")

    cached.ingest("louvre", "The Louvre is a museum in Paris.")
    cached.answer("eiffel tower")
    assert calls["n"] == 2

    cached.ingest_batch([{"id": "d", "text": "More text about towers."}])
    assert len(cached.cache) == 0
    cached.answer("eiffel tower")
    cached.delete("d")
    assert len(cached.cache) == 0
    cached.answer("eiffel tower")
    cached.clear()
    assert len(cached.cache) == 0 and cached.document_count == 0


def test_namespaces_do_not_share_answers():
    pipeline, calls = _pipeline()
    shared = SemanticCache()
    CachedPipeline(pipeline, shared, namespace="t1").answer("eiffel tower")
    CachedPipeline(pipeline, shared, namespace="t2").answer("eiffel tower")
    assert calls["n"] == 2
