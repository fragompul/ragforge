"""Query routing: choose how much to trust lexical vs. dense retrieval for each query.

Fixed fusion weights are a compromise. Some queries are *lexical* -- error codes, SKUs,
quoted phrases, rare proper nouns -- and BM25 nails them while embeddings blur them.
Others are *semantic* -- long natural-language questions or paraphrases with little
vocabulary overlap -- where dense retrieval wins. :class:`QueryRouter` inspects the query
(and, optionally, how strongly BM25 matches the corpus) and returns per-query RRF weights.

The default rules are deliberately transparent and tuning-free:

* **identifier / quoted-phrase present** -> lexical weight up;
* **a rare term matches the corpus** (high IDF) -> lexical weight up (BM25 has a precise
  handle on the query);
* **long wh-question, no specific term matches** -> dense weight up.

Whether routing beats fixed equal weights is an empirical question that depends on the
corpus; ``scripts/eval_beir.py`` measures it on public datasets and ``docs/evaluation.md``
reports the honest outcome.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragforge.index import ENGLISH_STOPWORDS, BM25Index, english_analyzer

_QUOTED = re.compile(r'"[^"]{3,}"|“[^”]{3,}”')
_IDENTIFIER = re.compile(r"\b(?=[\w.-]*\d)(?=[\w.-]*[A-Za-z])[\w.-]{3,}\b|\b[A-Z]{2,}[-_]\w+\b")
_WH = re.compile(r"^\s*(who|what|when|where|why|how|which|does|do|is|are|can)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RoutingDecision:
    """Per-query fusion weights and the reason they were chosen."""

    weight_bm25: float
    weight_vector: float
    reason: str


@dataclass
class QueryRouter:
    """Rule-based router; all knobs are plain attributes.

    Args:
        lexical_boost: Multiplier applied to the BM25 weight when the query is lexical.
        semantic_boost: Multiplier applied to the vector weight when it is semantic.
        rare_idf: IDF above which a matching query term counts as a precise lexical handle
            (only used when a BM25 index is passed to :meth:`route`).
        long_query_terms: Content-term count from which a wh-question counts as semantic.
    """

    lexical_boost: float = 2.0
    semantic_boost: float = 1.5
    rare_idf: float = 6.0
    long_query_terms: int = 6

    def route(self, query: str, bm25: BM25Index | None = None) -> RoutingDecision:
        if _QUOTED.search(query):
            return RoutingDecision(self.lexical_boost, 1.0, "quoted phrase")
        if _IDENTIFIER.search(query):
            return RoutingDecision(self.lexical_boost, 1.0, "identifier")

        terms = english_analyzer(query)
        if bm25 is not None and terms:
            best = max((bm25._idf(t) for t in terms if t in bm25._postings), default=0.0)
            if best >= self.rare_idf:
                return RoutingDecision(self.lexical_boost, 1.0, f"rare term (idf {best:.1f})")
        content = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in ENGLISH_STOPWORDS]
        if _WH.match(query) and len(content) >= self.long_query_terms:
            return RoutingDecision(1.0, self.semantic_boost, "long natural-language question")
        return RoutingDecision(1.0, 1.0, "balanced")
