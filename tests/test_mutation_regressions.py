"""Boundary and contract tests added after mutation testing of cache, routing and citations.

Each test pins behaviour a mutant (``<`` -> ``<=``, dropped argument, changed default) used to get
away with: validation edges and messages, equality at thresholds, exact serialized shapes.
"""

from __future__ import annotations

import pytest

from ragforge import RagPipeline
from ragforge.cache import CachedPipeline, SemanticCache
from ragforge.citations import attribute
from ragforge.index import BM25Index
from ragforge.routing import QueryRouter

# ------------------------------------------------------------------------------- cache


def axis(query: str) -> list[float]:
    """'a' -> (1, 0), 'b' -> (0, 1), 'ab' -> 45 degrees (cosine 0.7071 with either axis)."""
    return {"a": [1.0, 0.0], "b": [0.0, 1.0], "ab": [1.0, 1.0]}[query]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": 1.5}, r"threshold must be in \[-1, 1\], got 1.5"),
        ({"threshold": -1.5}, r"threshold must be in \[-1, 1\], got -1.5"),
        ({"max_size": 0}, r"max_size must be >= 1, got 0"),
        ({"ttl_seconds": 0}, r"ttl_seconds must be positive, got 0"),
    ],
)
def test_cache_validation_messages(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SemanticCache(**kwargs)


def test_cache_validation_accepts_the_inclusive_boundaries():
    SemanticCache(threshold=1.0)
    SemanticCache(threshold=-1.0)
    SemanticCache(max_size=1)
    SemanticCache(ttl_seconds=0.001)


def test_cache_hit_at_exactly_the_threshold_and_hit_counter():
    cache = SemanticCache(embed_fn=axis, threshold=1.0)
    cache.put("a", "value")
    entry = cache.get("a")
    assert entry is not None and entry.hits == 1  # sim == threshold is a hit
    cache.get("a")
    assert entry.hits == 2 and cache.stats.hits == 2
    assert cache.get("b") is None and cache.stats.misses == 1


def test_cache_picks_the_first_of_equally_similar_entries():
    cache = SemanticCache(embed_fn=axis, threshold=0.5)
    first = cache.put("a", "first")
    cache.put("a", "second")
    assert cache.get("a") is first


def test_cache_ttl_boundary_expiry_counter_and_eviction_counter():
    now = [0.0]
    cache = SemanticCache(embed_fn=axis, ttl_seconds=10, max_size=1, clock=lambda: now[0])
    cache.put("a", 1)
    now[0] = 10.0  # age == ttl: still alive
    assert cache.get("a") is not None
    now[0] = 10.5
    assert cache.get("a") is None and cache.stats.expirations == 1 and len(cache) == 0
    cache.put("a", 1)
    cache.put("b", 2)  # max_size 1: LRU eviction
    assert cache.stats.evictions == 1 and cache.get("b") is not None and cache.get("a") is None


def test_cache_skips_incompatible_vectors_and_partitions_but_keeps_scanning():
    def mixed(q: str) -> list[float]:
        return [1.0, 0.0, 0.0] if q == "long" else axis(q)

    cache = SemanticCache(embed_fn=mixed, threshold=0.9)
    cache.put("long", "other-dim")
    cache.put("a", "right", partition=("x",))
    cache.put("a", "wrong-partition", partition=("y",))
    got = cache.get("a", partition=("x",))
    assert got is not None and got.value == "right"


def test_cache_expired_entries_do_not_stop_the_scan():
    now = [0.0]
    cache = SemanticCache(embed_fn=axis, ttl_seconds=5, clock=lambda: now[0])
    cache.put("a", "old")
    now[0] = 4.0
    cache.put("b", "fresh")
    now[0] = 6.0  # "a" expired, "b" alive and listed after it
    assert cache.get("b") is not None


class Recorder:
    """Pipeline stand-in recording the arguments the cache forwards."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.pipeline = RagPipeline()
        self.pipeline.ingest("d", "alpha beta gamma delta")

    def __getattr__(self, name):
        return getattr(self.pipeline, name)

    def answer(self, query, **kwargs):
        self.calls.append(kwargs)
        return self.pipeline.answer(query, **kwargs)


def test_cached_pipeline_forwards_arguments_and_serves_hits_without_latency():
    inner = Recorder()
    cached = CachedPipeline(inner)  # type: ignore[arg-type]
    first = cached.answer("alpha beta", k=3, candidate_pool=7)
    assert inner.calls == [{"k": 3, "candidate_pool": 7, "filter_fn": None}]
    again = cached.answer("alpha beta", k=3, candidate_pool=7)
    assert len(inner.calls) == 1  # served from the cache
    assert again.query == "alpha beta" and again.answer == first.answer
    assert again.contexts == first.contexts and again.prompt == first.prompt
    assert again.retrieval_latency_ms == 0.0 and again.generation_latency_ms == 0.0
    assert again.total_latency_ms >= 0.0


def test_cached_pipeline_bypasses_the_cache_for_filtered_queries_and_forwards_the_filter():
    inner = Recorder()
    cached = CachedPipeline(inner)  # type: ignore[arg-type]
    only_d = lambda chunk: chunk.doc_id == "d"  # noqa: E731
    cached.answer("alpha beta", k=3, candidate_pool=7, filter_fn=only_d)
    cached.answer("alpha beta", k=3, candidate_pool=7, filter_fn=only_d)
    assert inner.calls == [{"k": 3, "candidate_pool": 7, "filter_fn": only_d}] * 2
    assert cached.cache.stats.bypasses == 2


def test_cached_pipeline_defaults_are_k5_pool20():
    inner = Recorder()
    CachedPipeline(inner).answer("alpha")  # type: ignore[arg-type]
    assert inner.calls[0]["k"] == 5 and inner.calls[0]["candidate_pool"] == 20


# ----------------------------------------------------------------------------- routing


def test_router_decisions_are_exact():
    router = QueryRouter()
    assert (
        router.route('find "exact phrase here"').weight_bm25,
        router.route('find "exact phrase here"').weight_vector,
    ) == (2.0, 1.0)
    d = router.route("What does ERR-4012 mean")
    assert (d.weight_bm25, d.weight_vector, d.reason) == (2.0, 1.0, "identifier")
    long_q = "how can I make the retrieval pipeline answer questions about long documents faster"
    d = router.route(long_q)
    assert (d.weight_bm25, d.weight_vector, d.reason) == (
        1.0,
        1.5,
        "long natural-language question",
    )
    d = router.route("what is a vector")
    assert (d.weight_bm25, d.weight_vector, d.reason) == (1.0, 1.0, "balanced")


def test_router_requires_a_wh_start_and_enough_content_terms():
    router = QueryRouter()
    # Long but not a question: stays balanced.
    assert (
        router.route("retrieval pipeline latency documents faster longer answers questions").reason
        == "balanced"
    )
    # Question starting with a stopword-heavy prefix: only content terms count.
    assert router.route("is the in of to a an and or").reason == "balanced"


def test_router_rare_term_uses_idf_threshold_inclusively():
    bm25 = BM25Index()
    from ragforge.chunking import Chunk

    chunks = [
        Chunk(
            id=f"c{i}",
            doc_id=f"d{i}",
            text=f"common filler words {'rareword' if i == 0 else ''}",
            position=0,
        )
        for i in range(40)
    ]
    bm25.add(chunks)
    idf = bm25._idf("rareword")
    assert (
        QueryRouter(rare_idf=idf).route("rareword", bm25).reason.startswith("rare term")
    )  # >= threshold
    assert QueryRouter(rare_idf=idf + 0.01).route("rareword", bm25).reason == "balanced"
    # An unmatched term contributes no handle (default 0, never a spurious boost):
    assert QueryRouter(rare_idf=0.5).route("zzzunknown", bm25).reason == "balanced"


# -------------------------------------------------------------------------- citations


class Ctx:
    def __init__(
        self, text: str, chunk_id: str = "c0", doc_id: str = "d0", start_char: int = 0
    ) -> None:
        self.text, self.chunk_id, self.doc_id, self.start_char = text, chunk_id, doc_id, start_char


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": 1.1}, r"threshold must be in \[0, 1\], got 1.1"),
        ({"threshold": -0.1}, r"threshold must be in \[0, 1\], got -0.1"),
        ({"max_citations": 0}, r"max_citations must be >= 1, got 0"),
        ({"semantic_weight": 1.5}, r"semantic_weight must be in \[0, 1\], got 1.5"),
        ({"nli_weight": 1.5}, r"nli_weight must be in \[0, 1\], got 1.5"),
        ({"nli_candidates": 0}, r"nli_candidates must be >= 1, got 0"),
    ],
)
def test_attribute_validation_messages(kwargs, message):
    with pytest.raises(ValueError, match=message):
        attribute("Some answer.", [Ctx("Some answer.")], **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0.0},
        {"threshold": 1.0},
        {"semantic_weight": 0.0},
        {"semantic_weight": 1.0},
        {"nli_candidates": 1},
        {"max_citations": 1},
    ],
)
def test_attribute_validation_accepts_inclusive_boundaries(kwargs):
    attribute("Some answer.", [Ctx("Some answer.")], **kwargs)


def test_attribute_defaults_cap_at_two_citations():
    contexts = [Ctx("The tower stands in Paris.", f"c{i}", f"d{i}") for i in range(4)]
    cited = attribute("The tower stands in Paris.", contexts)
    assert len(cited.sentences[0].citations) == 2  # max_citations default
    assert cited.answer == "The tower stands in Paris."


def test_attribute_support_equal_to_threshold_counts_as_supported():
    cited = attribute("alpha beta.", [Ctx("alpha beta.")], threshold=1.0)
    assert cited.sentences[0].supported and cited.sentences[0].citations


def test_attribute_unsupported_sentence_gets_no_citations_even_below_best_score():
    cited = attribute(
        "Completely unrelated zebra statement.", [Ctx("alpha beta gamma.")], threshold=0.6
    )
    sentence = cited.sentences[0]
    assert not sentence.supported and sentence.citations == []


def test_attribute_markers_equal_to_threshold_are_not_invalid():
    ctx = Ctx("alpha beta gamma delta.")
    cited = attribute("alpha beta gamma delta [1].", [ctx], threshold=1.0)
    assert cited.sentences[0].claimed_markers == [1] and cited.sentences[0].invalid_markers == []


def test_attribute_citation_spans_and_context_offsets():
    ctx = Ctx("Intro sentence. The tower is in Paris.", start_char=100)
    cited = attribute("The tower is in Paris.", [ctx])
    c = cited.sentences[0].citations[0]
    assert (c.start_char, c.end_char) == (100 + 16, 100 + 38)
    assert c.quote == "The tower is in Paris."
    sentence = cited.sentences[0]
    assert (sentence.start, sentence.end) == (0, 22)


def test_attribute_skips_blank_sentences_without_ending_the_scan():
    cited = attribute(
        "First fact here. [1]  Second fact here.", [Ctx("First fact here. Second fact here.")]
    )
    assert [s.text.strip() for s in cited.sentences if "fact" in s.text][-1].endswith(
        "Second fact here."
    )


def test_attribute_picks_earliest_span_on_equal_score():
    ctx = Ctx("Paris is big. Paris is big.")
    cited = attribute("Paris is big.", [ctx])
    assert cited.sentences[0].citations[0].start_char == 0


def test_attribute_semantic_blend_never_goes_below_lexical_floor():
    vectors = iter([[1.0, 0.0], [-1.0, 0.0]])  # sentence, then its candidate span: cosine -1

    def anti(text: str) -> list[float]:
        return next(vectors)

    ctx = Ctx("alpha beta gamma delta.")
    plain = attribute("alpha beta gamma delta.", [ctx])
    blended = attribute("alpha beta gamma delta.", [ctx], embed_fn=anti, semantic_weight=0.5)
    # negative cosine is clamped to 0: blended = 0.5 * coverage, strictly below the lexical score
    assert blended.sentences[0].support == pytest.approx(0.5 * plain.sentences[0].support)


def test_citation_and_sentence_dict_shapes():
    cited = attribute("The tower is in Paris.", [Ctx("The tower is in Paris.", "c1", "d1")])
    data = cited.to_dict()
    assert set(data) == {"answer", "groundedness", "sentences"}
    s = data["sentences"][0]
    assert set(s) == {
        "text",
        "start",
        "end",
        "support",
        "supported",
        "citations",
        "claimed_markers",
        "invalid_markers",
    }
    c = s["citations"][0]
    assert set(c) == {"marker", "chunk_id", "doc_id", "start_char", "end_char", "quote", "score"}
    assert (c["marker"], c["chunk_id"], c["doc_id"]) == (1, "c1", "d1")
    assert data["groundedness"] == round(cited.groundedness, 4) and c["score"] == round(
        cited.sentences[0].citations[0].score, 4
    )


def test_to_markdown_exact_rendering():
    ctx = Ctx("The tower is in Paris. It opened in 1889.", "c1", "eiffel")
    cited = attribute("The tower is in Paris . Zebras fly north.", [ctx])
    md = cited.to_markdown()
    first, *_ = md.splitlines()
    assert first == "The tower is in Paris. [1] Zebras fly north. (unverified)"
    assert (
        md.splitlines()[-1]
        == "[1] eielfel".replace("eielfel", "eiffel") + " (0-22): “The tower is in Paris.”"
    )
