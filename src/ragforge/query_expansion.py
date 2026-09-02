"""Multi-query retrieval: query rewriting fan-out fused via Reciprocal Rank Fusion.

A single query representation -- one BM25 tokenization, one embedding vector
-- captures only one phrasing of the user's intent. Hybrid search already
fuses *lexical* and *semantic* signals for a given phrasing (see
:class:`ragforge.index.HybridRetriever`), but a vocabulary mismatch between
the query and the corpus ("uptime guarantee" vs. "SLA", "cancel my plan" vs.
"terminate subscription") can cause a miss *before* fusion even runs, since
both retrievers are scoring the same single phrasing.

Multi-query retrieval addresses this at the query side: rewrite the query
into several paraphrases, retrieve independently for each variant (including
the original, unmodified query), and fuse the resulting rankings with the
same RRF algorithm ragforge already uses to combine BM25 and vector search
(see ``docs/math.md#2``). This is the modern, LLM-rewriting analogue of
classical pseudo-relevance feedback / query expansion (Rocchio, 1971).

Because the original query is always included as one of the variants, a
low-quality rewrite can dilute the fused ranking with irrelevant candidates
but can never *remove* results the single-query path would have found.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ragforge.chunking import Chunk
from ragforge.index import FilterFn, ScoredChunk

QueryExpansionFn = Callable[[str], list[str]]


class Searchable(Protocol):
    """Structural type for anything ``MultiQueryRetriever`` can wrap.

    Matches both ``ragforge.index.HybridRetriever`` and the individual
    ``BM25Index`` / ``VectorIndex`` / ``ApproxVectorIndex`` classes, so
    multi-query fan-out can sit in front of any retrieval stage.
    """

    def search(
        self, query: str, *, k: int, filter_fn: FilterFn | None = None
    ) -> list[ScoredChunk]: ...


def identity_expansion(query: str) -> list[str]:
    """No-op expansion: retrieve using only the original query."""
    return []


def llm_query_expansion_fn(
    generate_fn: Callable[[str, list[str]], str], num_variants: int = 3
) -> QueryExpansionFn:
    """Build a ``QueryExpansionFn`` that asks an LLM to paraphrase the query.

    ``generate_fn`` should be the same ``(query, contexts) -> answer``
    callable used elsewhere in ragforge (e.g. an OpenAI/Anthropic chat
    completion wrapper, as shown in the README's "Production Integration"
    example) -- it is called here with an empty context list and an
    instruction prompt in place of a question, since we want paraphrases,
    not an answer grounded in retrieved chunks.

    Expects one rewritten query per line in the response; blank lines and
    common list-numbering prefixes (``1.``, ``-``, ``*``) are stripped.
    """

    def _expand(query: str) -> list[str]:
        instruction = (
            f"Rewrite the following search query into {num_variants} alternative "
            "phrasings that preserve its meaning but vary vocabulary and sentence "
            "structure. Output exactly one rewritten query per line, with no "
            f"numbering, bullets, or extra commentary.\n\nQuery: {query}"
        )
        response = generate_fn(instruction, [])
        variants = []
        for line in response.splitlines():
            cleaned = line.strip().lstrip("-*0123456789.() ").strip()
            if cleaned:
                variants.append(cleaned)
        return variants[:num_variants]

    return _expand


class MultiQueryRetriever:
    """Fans a query out into rewritten variants and fuses results via RRF.

    Args:
        base_retriever: Any object exposing ``search(query, k, filter_fn=None)``,
            typically a :class:`ragforge.index.HybridRetriever`.
        expand_fn: Produces additional query phrasings for a given query.
            The original query is always searched too, regardless of what
            ``expand_fn`` returns.
        k_rrf: RRF rank-damping constant, matching
            :class:`ragforge.index.HybridRetriever`'s parameter of the same
            name (see ``docs/math.md#2`` for the fusion formula).
        max_expansions: Upper bound on how many rewritten variants are
            actually searched, capping retrieval cost regardless of how many
            ``expand_fn`` returns.
    """

    def __init__(
        self,
        base_retriever: Searchable,
        expand_fn: QueryExpansionFn = identity_expansion,
        k_rrf: int = 60,
        max_expansions: int = 3,
    ) -> None:
        if k_rrf <= 0:
            raise ValueError(f"k_rrf must be positive, got {k_rrf}")
        if max_expansions < 0:
            raise ValueError(f"max_expansions must be non-negative, got {max_expansions}")

        self.base_retriever = base_retriever
        self.expand_fn = expand_fn
        self.k_rrf = k_rrf
        self.max_expansions = max_expansions

    def search(
        self,
        query: str,
        k: int = 5,
        candidate_pool: int = 20,
        filter_fn: FilterFn | None = None,
    ) -> list[ScoredChunk]:
        """Retrieve for the original query and its expansions, fused via RRF."""
        if k <= 0:
            return []

        variants = [query]
        for variant in self.expand_fn(query)[: self.max_expansions]:
            cleaned = variant.strip()
            if cleaned and cleaned not in variants:
                variants.append(cleaned)

        pool = max(k, candidate_pool)
        rrf_scores: dict[str, float] = {}
        chunks_by_id: dict[str, Chunk] = {}
        provenance_by_id: dict[str, set[str]] = {}

        for variant in variants:
            results = self.base_retriever.search(variant, k=pool, filter_fn=filter_fn)
            for rank, scored in enumerate(results, start=1):
                cid = scored.chunk.id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.k_rrf + rank)
                chunks_by_id[cid] = scored.chunk
                provenance_by_id.setdefault(cid, set()).add(scored.provenance or "query_variant")

        fused = [
            ScoredChunk(
                chunk=chunks_by_id[cid],
                score=score,
                provenance=f"multi_query({'+'.join(sorted(provenance_by_id[cid]))})",
            )
            for cid, score in rrf_scores.items()
        ]
        fused.sort(key=lambda sc: sc.score, reverse=True)
        return fused[:k]
