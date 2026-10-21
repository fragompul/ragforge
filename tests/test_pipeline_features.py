"""Pipeline integration: extra indexes (sparse, graph), contextual retrieval, routing, HyDE."""

from __future__ import annotations

import json

import pytest

from ragforge import RagPipeline
from ragforge.chunking import Chunk, RecursiveCharacterChunker
from ragforge.contextual import (
    combine_context_fns,
    heading_context_fn,
    lead_context_fn,
    llm_context_fn,
    title_context_fn,
)
from ragforge.fusion import FusionRetriever
from ragforge.graph import GraphIndex
from ragforge.quantization import QuantizedVectorIndex
from ragforge.query_expansion import hyde_expansion_fn
from ragforge.routing import QueryRouter
from ragforge.sparse import SparseVectorIndex

DOCS = [
    ("derrickson", "Scott Derrickson directed the film Doctor Strange."),
    ("strange", "Doctor Strange is a superhero film from Marvel Studios released in 2016."),
    ("wood", "Ed Wood was an American filmmaker."),
]


def _pipeline(**kwargs) -> RagPipeline:
    p = RagPipeline(chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=0), **kwargs)
    for doc_id, text in DOCS:
        p.ingest(doc_id, text)
    return p


# ------------------------------------------------------------------- extra indexes


def test_graph_and_sparse_extras_join_the_fusion():
    pipeline = _pipeline(extra_indexes={"graph": GraphIndex(), "sparse": SparseVectorIndex()})
    assert isinstance(pipeline.retriever, FusionRetriever)

    top = pipeline.retrieve("When was the film directed by Scott Derrickson released?", k=3)

    assert {"derrickson", "strange"} <= {c.doc_id for c in top}
    assert any("graph" in c.ranks for c in top) and any("sparse" in c.ranks for c in top)
    assert any(c.ranks.keys() >= {"bm25", "vector"} for c in top)


def test_extras_follow_ingest_delete_and_clear():
    graph = GraphIndex()
    pipeline = _pipeline(extra_indexes={"graph": graph})
    assert len(graph) == 3
    pipeline.delete("wood")
    assert len(graph) == 2
    pipeline.clear()
    assert len(graph) == 0


def test_extras_persist_and_require_a_configured_instance_on_load(tmp_path):
    pipeline = _pipeline(
        extra_indexes={"sparse": SparseVectorIndex()}, extra_weights={"sparse": 2.0}
    )
    path = tmp_path / "idx.json"
    pipeline.save(path)
    data = json.loads(path.read_text())
    assert "sparse" in data["extras"] and data["extra_weights"]["sparse"] == 2.0

    loaded = RagPipeline.load(path, extra_indexes={"sparse": SparseVectorIndex()})
    assert len(loaded.extra_indexes["sparse"]) == 3
    assert loaded._base_retriever.weight("sparse") == 2.0
    with pytest.raises(ValueError, match="extra_indexes"):
        RagPipeline.load(path)


def test_pipeline_without_extras_serializes_without_extras_key():
    assert "extras" not in _pipeline().to_dict()


def test_extra_weight_zero_disables_a_source():
    pipeline = _pipeline(extra_indexes={"graph": GraphIndex()}, extra_weights={"graph": 0.0})
    for chunk in pipeline.retrieve("Scott Derrickson", k=3):
        assert "graph" not in chunk.ranks


# ---------------------------------------------------------- quantized vector backends


@pytest.mark.parametrize("backend", ["int8", "binary"])
def test_quantized_backends_roundtrip(tmp_path, backend):
    pipeline = _pipeline(vector_backend=backend)
    assert isinstance(pipeline.vector_index, QuantizedVectorIndex)
    path = tmp_path / "q.json"
    pipeline.save(path)
    loaded = RagPipeline.load(path)
    assert loaded.vector_backend == backend and isinstance(
        loaded.vector_index, QuantizedVectorIndex
    )
    assert loaded.retrieve("Ed Wood filmmaker", k=1)[0].doc_id == "wood"


# ------------------------------------------------------------ contextual retrieval


def test_contextualizer_improves_findability_but_keeps_original_text():
    plain = RagPipeline(chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=0))
    contextual = RagPipeline(
        chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=0),
        contextualizer=lambda doc, chunk: (
            "ACME Corp Q2 2023 earnings report" if chunk.doc_id == "acme" else ""
        ),
    )
    for p in (plain, contextual):
        p.ingest(
            "acme", "Revenue grew by 3% over the previous quarter.", metadata={"title": "ACME"}
        )
        p.ingest("other", "Revenue declined sharply; costs rose.")

    query = "ACME Corp Q2 2023 revenue growth"
    assert plain.bm25_index.search(query, k=1)[0].chunk.doc_id == "other"  # the gap is real
    top = contextual.retrieve(query, k=1)[0]
    assert top.doc_id == "acme"
    assert top.text == "Revenue grew by 3% over the previous quarter."  # citations use the original
    assert contextual.bm25_index._chunks[0].context.startswith("ACME")


def test_context_is_persisted_with_the_chunk(tmp_path):
    p = RagPipeline(contextualizer=lambda d, c: "situating context")
    p.ingest("d", "Some text about things.")
    path = tmp_path / "c.json"
    p.save(path)
    loaded = RagPipeline.load(path)
    assert loaded.bm25_index._chunks[0].context == "situating context"
    assert Chunk.from_dict(Chunk(id="a", text="t", doc_id="d", position=0).to_dict()).context == ""


def test_chunk_index_text_and_serialization():
    plain = Chunk(id="a", text="body", doc_id="d", position=0)
    assert plain.index_text == "body" and "context" not in plain.to_dict()
    with_ctx = Chunk(id="a", text="body", doc_id="d", position=0, context="where")
    assert with_ctx.index_text == "where\nbody" and with_ctx.to_dict()["context"] == "where"


def test_heuristic_contexts():
    doc = (
        "# Guide\n\nIntro.\n\n## Install\n\nRun pip.\n\n### Linux\n\nUse apt.\n\n"
        "## Usage\n\nCall it."
    )
    chunk_at = lambda needle: Chunk(  # noqa: E731
        id="c", text=needle, doc_id="d", position=0, start_char=doc.index(needle), end_char=0
    )
    assert heading_context_fn(doc, chunk_at("Use apt.")) == "Guide > Install > Linux"
    assert heading_context_fn(doc, chunk_at("Call it.")) == "Guide > Usage"
    bare = Chunk(id="c", text="no", doc_id="d", position=0)
    assert heading_context_fn("no headings here", bare) == ""

    titled = Chunk(
        id="c", text="x", doc_id="my-doc_id", position=0, metadata={"title": "Real Title"}
    )
    assert title_context_fn()("", titled) == "Real Title"
    assert (
        title_context_fn()("", Chunk(id="c", text="x", doc_id="my-doc_id", position=0))
        == "my doc id"
    )

    lead = lead_context_fn(max_chars=20)
    body = Chunk(id="c", text="later", doc_id="d", position=1, start_char=50)
    assert lead("The opening sentence is long. More.", body) == "The opening sentence"
    first = Chunk(id="c", text="The opening", doc_id="d", position=0, start_char=0)
    assert lead("The opening sentence.", first) == ""

    both = combine_context_fns(lambda d, c: "A", lambda d, c: "", lambda d, c: "B")
    assert both("", body) == "A | B"


def test_llm_context_fn_prompts_with_document_and_chunk():
    seen = []
    fn = llm_context_fn(
        lambda prompt: seen.append(prompt) or "  Section about taxes.  ", max_document_chars=20
    )
    chunk = Chunk(id="c", text="Rates rose.", doc_id="d", position=0)
    assert fn("D" * 100, chunk) == "Section about taxes."
    assert "D" * 20 in seen[0] and "D" * 21 not in seen[0] and "Rates rose." in seen[0]


# -------------------------------------------------------------------------- routing


def test_router_rules_and_trace_attribute():
    router = QueryRouter()
    assert router.route('find "exact phrase here"').reason == "quoted phrase"
    assert router.route("error ERR-4012 happened").reason == "identifier"
    assert router.route("sku A1B2C3 price").weight_bm25 == 2.0
    assert (
        router.route(
            "how does the embedding model decide which documents are similar to a query"
        ).weight_vector
        == 1.5
    )
    assert router.route("hello").reason == "balanced"


def test_router_uses_corpus_idf_for_rare_terms():
    pipeline = _pipeline()
    for i in range(40):
        pipeline.ingest(f"filler{i}", f"common words appear everywhere number {i}")
    pipeline.ingest("rare", "zeppelin hangar")
    router = QueryRouter(rare_idf=3.0)
    assert router.route("zeppelin", pipeline.bm25_index).reason.startswith("rare term")
    assert router.route("common words", pipeline.bm25_index).reason == "balanced"


def test_pipeline_applies_routing_per_query_without_drifting():
    from ragforge.telemetry import InMemoryExporter, Tracer

    exporter = InMemoryExporter()
    pipeline = _pipeline(router=QueryRouter(), tracer=Tracer(exporters=[exporter]), weight_bm25=1.5)

    pipeline.retrieve("ERR-4012 timeout", k=2)
    base = pipeline._base_retriever
    assert (base.weight_bm25, base.weight_vector) == (1.5 * 2.0, 1.0)
    pipeline.retrieve("hello", k=2)
    assert (base.weight_bm25, base.weight_vector) == (
        1.5,
        1.0,
    )  # reset from the pipeline's own weights
    routes = [s.attributes["route"] for s in exporter.spans if s.name == "fusion_search"]
    assert routes == ["identifier", "balanced"]


def test_routing_with_fusion_retriever():
    pipeline = _pipeline(router=QueryRouter(), extra_indexes={"graph": GraphIndex()})
    pipeline.retrieve('"Doctor Strange"', k=2)
    assert pipeline._base_retriever.weight("bm25") == 2.0


# ----------------------------------------------------------------------------- HyDE


def test_hyde_adds_a_generated_passage_as_a_query_variant():
    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "The tower is 330 metres tall and stands in Paris."

    expand = hyde_expansion_fn(fake_llm)
    assert expand("How tall is the Eiffel Tower?") == [
        "The tower is 330 metres tall and stands in Paris."
    ]
    assert "How tall is the Eiffel Tower?" in calls[0]
    assert hyde_expansion_fn(lambda p: "   ")("q") == []
    assert len(hyde_expansion_fn(lambda p: "x y", include_question_style=True)("q")) == 2


def test_hyde_finds_documents_a_vocabulary_gap_would_miss():
    pipeline = RagPipeline(
        chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=0),
        query_expansion_fn=hyde_expansion_fn(lambda p: "The Eiffel Tower is 330 metres tall."),
    )
    pipeline.ingest("tower", "The Eiffel Tower is 330 metres tall and completed in 1889.")
    pipeline.ingest("noise", "Bananas are yellow fruit grown in tropical climates.")
    top = pipeline.retrieve("what is the altitude of that famous landmark", k=1)
    assert top[0].doc_id == "tower"
