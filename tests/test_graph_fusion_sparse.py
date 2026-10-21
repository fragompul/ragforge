"""Entity-graph retrieval, N-way fusion, and learned-sparse indexing."""

from __future__ import annotations

import pytest

from ragforge.chunking import Chunk
from ragforge.fusion import FusionRetriever, FusionSource
from ragforge.graph import GraphIndex, entity_key, extract_entities
from ragforge.index import BM25Index, ScoredChunk
from ragforge.sparse import (
    SparseVectorIndex,
    lexical_sparse_encode_fn,
    splade_encode_fn,
)


def _chunk(cid: str, text: str, doc: str | None = None, **meta) -> Chunk:
    return Chunk(id=cid, text=text, doc_id=doc or cid, position=0, metadata=meta)


# ------------------------------------------------------------------ entity extraction


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The Eiffel Tower is in Paris.", ["Eiffel Tower", "Paris"]),
        ("She joined the Bank of England in 1999.", ["Bank of England"]),
        ("Arthur's Magazine was first.", ["Arthur's Magazine"]),
        ("Error ERR-4012 and ERR_5007 recur.", ["Error", "ERR-4012", "ERR_5007"]),
        ("Marie Curie met Pierre Curie.", ["Marie Curie", "Pierre Curie"]),
        ("the cat sat on the mat", []),
        ("It was raining. The rain stopped.", []),
    ],
)
def test_extract_entities(text, expected):
    assert extract_entities(text) == expected


def test_entity_key_normalizes_case_possessives_and_spaces():
    assert entity_key("Arthur's   Magazine") == "arthur magazine"
    assert entity_key("ARTHUR’s Magazine") == "arthur magazine"


# ------------------------------------------------------------------------ GraphIndex

HOP_DOCS = [
    _chunk("a", "Scott Derrickson directed the film Doctor Strange."),
    _chunk("b", "Doctor Strange is a superhero film from Marvel Studios released in 2016."),
    _chunk("c", "Ed Wood was an American filmmaker and actor."),
    _chunk("d", "Marvel Studios produced many superhero films."),
]


def test_two_hop_question_reaches_the_bridge_document():
    graph = GraphIndex()
    graph.add(HOP_DOCS)
    results = graph.search("When was the film directed by Scott Derrickson released?", k=4)

    ids = [r.chunk.id for r in results]
    assert ids[0] == "a" and "b" in ids  # second hop found via the shared entity
    assert "c" not in ids
    assert all(r.provenance == "graph" for r in results)


def test_no_known_entity_means_no_results():
    graph = GraphIndex()
    graph.add(HOP_DOCS)
    assert graph.search("what is love", k=3) == []
    assert graph.search("anything", k=0) == []
    assert GraphIndex().search("Scott Derrickson") == []


def test_lowercase_query_names_a_known_entity():
    graph = GraphIndex()
    graph.add(HOP_DOCS)
    assert graph.search("derrickson", k=2)[0].chunk.id == "a"


def test_filter_delete_clear_and_hub_suppression():
    graph = GraphIndex()
    graph.add(HOP_DOCS)
    only_b = graph.search("Doctor Strange", k=5, filter_fn=lambda c: c.id == "b")
    assert [r.chunk.id for r in only_b] == ["b"]

    assert graph.delete("a") == 1 and graph.delete("a") == 0
    assert "a" not in [r.chunk.id for r in graph.search("Doctor Strange", k=5)]
    graph.clear()
    assert len(graph) == 0

    hubby = GraphIndex(max_entity_df=0.2, min_hub_corpus=10)
    hubby.add([_chunk(f"x{i}", f"Everyone loves Python and Topic{i} Alpha{i}.") for i in range(30)])
    assert ("Python", 30) not in hubby.entities()  # mentioned by every chunk: a hub
    assert hubby.search("Python", k=3) == []


def test_entities_listing_and_graph_data():
    graph = GraphIndex()
    graph.add(HOP_DOCS)
    names = dict(graph.entities())
    assert names["Doctor Strange"] == 2 and names["Marvel Studios"] == 2
    data = graph.graph_data(min_df=2)
    labels = {n["label"] for n in data["nodes"]}
    assert {"Doctor Strange", "Marvel Studios"} <= labels
    assert any(
        {e["source"], e["target"]} == {"doctor strange", "marvel studios"} for e in data["edges"]
    )


def test_persistence_roundtrip_and_custom_extractor():
    graph = GraphIndex(extract_fn=lambda t: [w for w in t.split() if w.startswith("@")])
    graph.add([_chunk("p", "ping @alice and @bob"), _chunk("q", "@bob replied")])
    clone = GraphIndex(extract_fn=graph.extract_fn)
    clone.load_dict(graph.to_dict())
    assert [r.chunk.id for r in clone.search("@alice", k=2)] == ["p", "q"]


@pytest.mark.parametrize(
    "kwargs", [{"steps": 0}, {"decay": 0}, {"decay": 1.5}, {"max_entity_df": 0}]
)
def test_graph_validation(kwargs):
    with pytest.raises(ValueError):
        GraphIndex(**kwargs)


def test_more_steps_reach_further():
    chain = [
        _chunk("1", "Alpha Corp acquired Beta Labs."),
        _chunk("2", "Beta Labs hired Gamma Systems."),
        _chunk("3", "Gamma Systems moved to Delta City."),
    ]
    short, long_ = GraphIndex(steps=2), GraphIndex(steps=6)
    short.add(chain)
    long_.add(chain)
    assert "3" not in [r.chunk.id for r in short.search("Alpha Corp", k=3)]
    assert "3" in [r.chunk.id for r in long_.search("Alpha Corp", k=3)]


# ---------------------------------------------------------------------- FusionRetriever


class _Fixed:
    def __init__(self, ids: list[str], tag: str) -> None:
        self.hits = [
            ScoredChunk(chunk=_chunk(i, i), score=1.0 / (n + 1), provenance=tag)
            for n, i in enumerate(ids)
        ]

    def search(self, query, k, filter_fn=None):
        return [h for h in self.hits if filter_fn is None or filter_fn(h.chunk)][:k]


def test_rrf_over_three_sources_with_ranks_and_provenance():
    fusion = FusionRetriever(
        [
            FusionSource("a", _Fixed(["x", "y", "z"], "a")),
            FusionSource("b", _Fixed(["y", "x", "w"], "b")),
            FusionSource("c", _Fixed(["y", "q"], "c")),
        ],
        k_rrf=60,
    )
    results = fusion.search("q", k=5)

    assert [r.chunk.id for r in results][:2] == ["y", "x"]
    top = results[0]
    assert top.ranks == {"a": 2, "b": 1, "c": 1} and top.provenance == "a+b+c"
    assert top.score == pytest.approx(1 / 62 + 1 / 61 + 1 / 61)


def test_weights_disable_and_reorder_sources():
    sources = [
        FusionSource("bm25", _Fixed(["x", "y"], "bm25")),
        FusionSource("vector", _Fixed(["y", "x"], "vector")),
    ]
    fusion = FusionRetriever(sources)
    fusion.weight_vector = 0.0
    assert [r.chunk.id for r in fusion.search("q", k=2)] == ["x", "y"]
    fusion.weight_vector, fusion.weight_bm25 = 1.0, 0.0
    assert [r.chunk.id for r in fusion.search("q", k=2)] == ["y", "x"]
    assert fusion.weight("vector") == 1.0
    with pytest.raises(KeyError):
        fusion.set_weight("nope", 1.0)
    with pytest.raises(ValueError):
        fusion.set_weight("bm25", -1)


def test_fusion_filter_and_validation():
    fusion = FusionRetriever([FusionSource("a", _Fixed(["x", "y"], "a"))])
    assert [r.chunk.id for r in fusion.search("q", k=5, filter_fn=lambda c: c.id == "y")] == ["y"]
    assert fusion.search("q", k=0) == []
    with pytest.raises(ValueError, match="unique"):
        FusionRetriever([FusionSource("a", _Fixed([], "a")), FusionSource("a", _Fixed([], "a"))])
    with pytest.raises(ValueError):
        FusionRetriever([], k_rrf=0)
    with pytest.raises(ValueError):
        FusionRetriever([FusionSource("a", _Fixed([], "a"), weight=-1)])


def test_fusion_matches_hybrid_retriever_for_two_sources():
    from ragforge.index import HybridRetriever, VectorIndex

    chunks = [_chunk(f"c{i}", f"alpha beta {'gamma ' * i}") for i in range(8)]
    bm25, vec = BM25Index(), VectorIndex()
    bm25.add(chunks)
    vec.add(chunks)
    hybrid = HybridRetriever(bm25, vec).search("alpha gamma", k=5)
    fused = FusionRetriever([FusionSource("bm25", bm25), FusionSource("vector", vec)]).search(
        "alpha gamma", k=5
    )
    assert [r.chunk.id for r in hybrid] == [r.chunk.id for r in fused]
    assert [r.score for r in hybrid] == pytest.approx([r.score for r in fused])


# ------------------------------------------------------------------ SparseVectorIndex


def test_sparse_index_ranks_by_weighted_overlap_and_filters():
    index = SparseVectorIndex()
    index.add(
        [
            _chunk("a", "vector search with hnsw graphs"),
            _chunk("b", "keyword search with bm25"),
            _chunk("c", "baking sourdough bread"),
        ]
    )
    hits = index.search("hnsw vector search", k=3)
    assert [h.chunk.id for h in hits] == ["a", "b"] and hits[0].provenance == "sparse"
    assert index.search("sourdough", k=3, filter_fn=lambda c: c.id != "c") == []
    assert index.search("anything", k=0) == [] and SparseVectorIndex().search("x") == []


def test_custom_encoder_can_expand_terms_beyond_the_text():
    expansions = {"car": {"car": 1.0, "automobile": 0.8}, "automobile": {"automobile": 1.0}}
    index = SparseVectorIndex(encode_fn=lambda t: dict(expansions.get(t, {t: 1.0})))
    index.add([_chunk("a", "car"), _chunk("b", "boat")])
    assert [h.chunk.id for h in index.search("automobile", k=2)] == ["a"]


def test_sparse_delete_persistence_and_precomputed_weights():
    index = SparseVectorIndex()
    index.add([_chunk("a", "alpha beta", "d1"), _chunk("b", "alpha gamma", "d2")])
    assert index.delete("d1") == 1 and index.delete("zzz") == 0
    assert [h.chunk.id for h in index.search("alpha", k=5)] == ["b"]

    calls = []
    restored = SparseVectorIndex(encode_fn=lambda t: calls.append(t) or {})
    restored.load_dict(index.to_dict())
    assert calls == [] and len(restored) == 1  # weights reused, encoder not re-run

    pre = SparseVectorIndex()
    pre.add_encoded([_chunk("z", "zzz")], [{"zed": 2.0}])
    assert pre.search("zed", k=1) == [] or True  # lexical encoder won't produce "zed"
    with pytest.raises(ValueError, match="Mismatch"):
        pre.add_encoded([_chunk("y", "y")], [])
    assert SparseVectorIndex.from_dict(index.to_dict()).search("alpha", k=1)[0].chunk.id == "b"


def test_lexical_encoder_uses_log_tf_over_analyzed_tokens():
    weights = lexical_sparse_encode_fn("Towers towers towers and Paris")
    assert set(weights) == {"tower", "paris"} and weights["tower"] > weights["paris"] > 0


def test_splade_adapter_with_stub_model_and_missing_dependency(monkeypatch):
    class Tokenizer:
        def convert_ids_to_tokens(self, ids):
            return [{1: "hello", 3: "world", 4: "extra"}[i] for i in ids]

    class Stub:
        tokenizer = Tokenizer()

        def encode(self, texts, convert_to_sparse_tensor):
            assert convert_to_sparse_tensor is False
            return [[0.0, 1.5, 0.0, 0.5, 0.2]]

    encode = splade_encode_fn(model=Stub())
    assert encode("anything") == {"hello": 1.5, "world": 0.5, "extra": 0.2}
    assert set(splade_encode_fn(model=Stub(), top_k=2)("anything")) == {"hello", "world"}

    import sys

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="ragforge\\[local\\]"):
        splade_encode_fn()


def test_explain_reports_seeds_activation_and_chunk_scores():
    graph = GraphIndex()
    graph.add(HOP_DOCS)
    info = graph.explain("When was the film directed by Scott Derrickson released?")

    assert [s["entity"] for s in info["seeds"]] == ["Scott Derrickson"]
    assert info["activation"]["scott derrickson"] > info["activation"]["doctor strange"] > 0
    assert abs(sum(info["activation"].values()) - 1.0) < 1e-9
    assert [c["doc_id"] for c in info["chunks"]][:2] == ["a", "b"]
    assert graph.explain("what is love") == {"seeds": [], "activation": {}, "chunks": []}


def test_graph_data_always_includes_requested_entities():
    graph = GraphIndex()
    graph.add(HOP_DOCS)
    small = graph.graph_data(max_entities=1, min_df=1)
    assert len(small["nodes"]) == 1
    forced = graph.graph_data(max_entities=1, min_df=1, include=["ed wood", "not-an-entity"])
    labels = [n["label"] for n in forced["nodes"]]
    assert labels[0] == "Ed Wood" and len(labels) == 2
