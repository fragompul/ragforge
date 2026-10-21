"""Semantic caching and LLM-as-judge evaluation.

* ``CachedPipeline`` answers a rephrased repeat question without re-running the
  pipeline, and invalidates itself when the corpus changes.
* ``evaluate_with_judge`` grades answers on a rubric. The offline
  ``lexical_judge_fn`` stands in for a real model here; swap in
  ``anthropic_complete_fn()`` (pip install ragforge[anthropic]) for the real thing.

Run with: python examples/cache_and_judge.py
"""

from ragforge import CachedPipeline, RagPipeline, SemanticCache
from ragforge.chunking import RecursiveCharacterChunker
from ragforge.evaluation import EvalCase
from ragforge.extractive import extractive_generate_fn
from ragforge.judge import evaluate_with_judge, lexical_judge_fn
from ragforge.sample import SAMPLE_DOCUMENTS, SAMPLE_EVAL_CASES


def main() -> dict[str, object]:
    pipeline = RagPipeline(
        chunker=RecursiveCharacterChunker(chunk_size=500, chunk_overlap=0),
        bm25_analyzer="english",
        generate_fn=extractive_generate_fn(),
    )
    pipeline.ingest_batch(SAMPLE_DOCUMENTS)

    cached = CachedPipeline(pipeline, SemanticCache(embed_fn=pipeline.embed_fn, threshold=0.85))
    first = cached.answer("How does reciprocal rank fusion combine rankings?")
    again = cached.answer("how does reciprocal rank fusion combine the rankings")
    print(f"first : {first.metadata['cache']:<5} {first.total_latency_ms:.2f} ms")
    print(f"again : {again.metadata['cache']:<5} {again.total_latency_ms:.2f} ms")
    cached.ingest("extra", "A new document invalidates every cached answer.")
    print("cache entries after ingest:", len(cached.cache))

    cases = [
        EvalCase(c["query"], c["relevant_doc_ids"], c["ground_truth_answer"])
        for c in SAMPLE_EVAL_CASES
    ]
    summary = evaluate_with_judge(
        pipeline, cases, lexical_judge_fn, rubrics=("faithfulness", "correctness")
    )
    print("\n" + summary.to_markdown_table())
    return {
        "first": first.metadata["cache"],
        "second": again.metadata["cache"],
        "entries_after_ingest": len(cached.cache),
        "faithfulness": summary.means["faithfulness"],
    }


if __name__ == "__main__":
    main()
