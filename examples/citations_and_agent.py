"""Verifiable citations and an agentic loop that knows when to abstain.

1. Answer a question and attribute every sentence to exact character offsets
   in the source, flagging a hallucinated sentence.
2. Run the agentic retrieve -> grade -> rewrite loop on a multi-part question
   and on an unanswerable one.

Run with: python examples/citations_and_agent.py
"""

from ragforge import AgenticRag, RagPipeline, attribute
from ragforge.agent import heuristic_decompose_fn
from ragforge.chunking import RecursiveCharacterChunker
from ragforge.extractive import extractive_generate_fn
from ragforge.sample import SAMPLE_DOCUMENTS


def main() -> dict[str, object]:
    pipeline = RagPipeline(
        chunker=RecursiveCharacterChunker(chunk_size=500, chunk_overlap=0),
        bm25_analyzer="english",
        generate_fn=extractive_generate_fn(),
    )
    pipeline.ingest_batch(SAMPLE_DOCUMENTS)

    response = pipeline.answer("Why does BM25 fail on synonyms?", k=3)
    cited = attribute(
        response.answer + " BM25 was invented in 2031 by a chatbot.", response.contexts
    )
    print(cited.to_markdown(), f"\n\ngroundedness = {cited.groundedness:.2f}\n")

    agent = AgenticRag(pipeline, decompose_fn=heuristic_decompose_fn)
    multi = agent.run("What does error ERR-4012 mean? How does reciprocal rank fusion work?", k=2)
    nothing = agent.run("What is the capital of Atlantis and its quantum zeppelin fleet?")
    for result in (multi, nothing):
        print(f"{result.question}\n  " + " -> ".join(s.kind for s in result.steps))

    return {
        "groundedness": cited.groundedness,
        "unsupported": [s.text for s in cited.unsupported],
        "multi_hop_docs": sorted({c.doc_id for c in multi.answer.contexts}),
        "abstained": nothing.abstained,
    }


if __name__ == "__main__":
    main()
