"""Shows why hybrid (BM25 + vector) retrieval beats BM25 alone: BM25 nails
exact keyword/code matches, but returns nothing for a paraphrase that
shares no vocabulary with the target chunk. A real embedding model
captures that kind of semantic match; this example uses a small
synonym-aware toy embedding (not the library's default `hashing_embed`,
which is purely lexical) to illustrate what a production embedding model
provides, without requiring a network call or an API key.

Run with: python examples/hybrid_vs_single.py
"""

import re
import zlib

from ragforge.chunking import Chunk
from ragforge.index import BM25Index, HybridRetriever, VectorIndex

_SYNONYMS = {"vehicle": "car", "automobile": "car", "turn": "activate", "start": "activate"}


def toy_semantic_embed(text: str, dims: int = 4096) -> list[float]:
    """A hand-built stand-in for a real embedding model: maps a few known
    synonyms to the same token before hashing, so paraphrases that use
    those synonyms produce similar vectors. A real deployment would call
    an actual embedding API instead of hand-coding synonym pairs.
    """

    vector = [0.0] * dims
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        canonical = _SYNONYMS.get(word, word)
        token_hash = zlib.crc32(canonical.encode("utf-8"))
        vector[token_hash % dims] += 1.0
    return vector


CHUNKS = [
    Chunk(
        id="c1",
        text="Error code E4021 indicates a network timeout during checkout.",
        doc_id="d1",
        position=0,
    ),
    Chunk(
        id="c2",
        text="The automobile would not start because the battery was depleted.",
        doc_id="d2",
        position=0,
    ),
    Chunk(
        id="c3",
        text="Our return policy allows refunds within 30 days of purchase.",
        doc_id="d3",
        position=0,
    ),
]

# Shares the exact code "E4021" with c1 -- BM25's strength.
EXACT_MATCH_QUERY = "What does error E4021 mean?"
# Paraphrases c2 ("vehicle"/"automobile", "turn on"/"start") with zero
# literal token overlap -- BM25 returns nothing; a semantic embedding
# still finds it.
PARAPHRASE_QUERY = "my vehicle won't turn on"


def main() -> dict:
    bm25 = BM25Index()
    bm25.add(CHUNKS)
    vector = VectorIndex(embed_fn=toy_semantic_embed)
    vector.add(CHUNKS)
    hybrid = HybridRetriever(bm25, vector)

    results = {}
    for label, query in [("exact_match", EXACT_MATCH_QUERY), ("paraphrase", PARAPHRASE_QUERY)]:
        bm25_top = bm25.search(query, k=1)
        hybrid_top = hybrid.search(query, k=1)
        results[label] = {
            "bm25_only": bm25_top[0].chunk.id if bm25_top else None,
            "hybrid": hybrid_top[0].chunk.id if hybrid_top else None,
        }
        entry = results[label]
        print(f"{label}: BM25-only={entry['bm25_only']}  Hybrid={entry['hybrid']}")

    return results


if __name__ == "__main__":
    main()
