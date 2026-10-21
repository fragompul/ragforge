"""Multi-hop support: forced hops, bridge-entity rewrites, graded contexts."""

from __future__ import annotations

import pytest

from ragforge import AgenticRag, RagPipeline
from ragforge.agent import Grade, bridge_rewrite_fn, lexical_grade_fn
from ragforge.chunking import RecursiveCharacterChunker
from ragforge.fusion import FusionRetriever


def _pipeline() -> RagPipeline:
    p = RagPipeline(
        chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=0), bm25_analyzer="english"
    )
    p.ingest("film", "Scott Derrickson directed the film Doctor Strange in 2016.")
    p.ingest("studio", "Doctor Strange was produced by Marvel Studios, a company based in Burbank.")
    p.ingest("noise", "Bananas grow in tropical regions and are rich in potassium.")
    p.ingest("noise2", "Mount Everest is the tallest mountain above sea level.")
    return p


QUESTION = "Where is the studio that produced the film directed by Scott Derrickson based?"


def test_bridge_rewrite_appends_entities_from_graded_contexts():
    rewrite = bridge_rewrite_fn(max_entities=2)
    grade = Grade(1.0, contexts=["Scott Derrickson directed Doctor Strange in 2016.", "Other"])
    new = rewrite(QUESTION, grade, [QUESTION])
    assert (
        new.startswith(QUESTION) and "Doctor Strange" in new and "Scott" not in new[len(QUESTION) :]
    )
    assert rewrite(QUESTION, Grade(1.0, contexts=["nothing capitalised here"]), [QUESTION]) == ""
    assert rewrite(QUESTION, Grade(1.0), [QUESTION]) == ""  # no contexts -> nothing to bridge


def test_min_iterations_forces_a_second_hop_even_when_first_looks_sufficient():
    pipeline = _pipeline()
    always_sufficient = lambda q, c: Grade(1.0)  # noqa: E731
    single = AgenticRag(pipeline, grade_fn=always_sufficient, rewrite_fn=bridge_rewrite_fn()).run(
        QUESTION, k=1
    )
    multi = AgenticRag(
        pipeline,
        grade_fn=always_sufficient,
        rewrite_fn=bridge_rewrite_fn(),
        min_iterations=2,
        max_iterations=3,
    ).run(QUESTION, k=1)

    assert single.retrievals == 1 and multi.retrievals >= 2
    assert {"film", "studio"} <= {c.doc_id for c in multi.answer.contexts}
    assert "studio" not in {c.doc_id for c in single.answer.contexts}
    kinds = [s.kind for s in multi.steps]
    assert "rewrite" in kinds and kinds[-1] == "answer"


def test_hop_stops_early_when_rewrite_has_nothing_new():
    pipeline = _pipeline()
    agent = AgenticRag(
        pipeline, rewrite_fn=lambda q, g, prev: "", min_iterations=3, max_iterations=3
    )
    result = agent.run("Doctor Strange film", k=1)
    assert result.retrievals == 1
    assert any(s.kind == "stop" and "no new query" in s.detail for s in result.steps)


def test_grade_carries_the_graded_contexts():
    seen = []

    def spy_grade(question, contexts):
        grade = lexical_grade_fn(question, contexts)
        seen.append(grade)
        return grade

    AgenticRag(_pipeline(), grade_fn=spy_grade).run("Scott Derrickson film", k=2)
    assert seen and seen[0].contexts and "Derrickson" in seen[0].contexts[0]


@pytest.mark.parametrize("kwargs", [{"min_iterations": 0}, {"min_iterations": 4}])
def test_min_iterations_validation(kwargs):
    with pytest.raises(ValueError, match="min_iterations"):
        AgenticRag(_pipeline(), max_iterations=3, **kwargs)


def test_agent_works_over_a_fusion_retriever_with_graph_source():
    from ragforge.graph import GraphIndex

    pipeline = RagPipeline(
        chunker=RecursiveCharacterChunker(chunk_size=300, chunk_overlap=0),
        extra_indexes={"graph": GraphIndex()},
    )
    for doc_id, text in [
        ("film", "Scott Derrickson directed the film Doctor Strange in 2016."),
        ("studio", "Doctor Strange was produced by Marvel Studios, a company based in Burbank."),
        ("noise", "Bananas grow in tropical regions."),
    ]:
        pipeline.ingest(doc_id, text)
    assert isinstance(pipeline.retriever, FusionRetriever)
    result = AgenticRag(pipeline).run(QUESTION, k=3)
    assert {"film", "studio"} <= {c.doc_id for c in result.answer.contexts}
