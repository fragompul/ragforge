"""Entity-graph retrieval ("GraphRAG-lite"): find chunks by following shared entities.

Dense and lexical retrievers rank chunks one at a time, so a **multi-hop** question -- one
whose answer needs facts from two documents linked by a bridge entity -- often finds the
first document and misses the second: its text shares little vocabulary with the question.
Graph retrieval fixes that by indexing *entities* and walking the links:

1. every chunk is connected to the entities it mentions (capitalized names, acronyms,
   identifiers such as ``ERR-4012``), weighted by how rare the entity is (IDF);
2. a query activates the entities it mentions;
3. activation diffuses through the bipartite chunk/entity graph for a few steps, as
   personalized PageRank with restart. A chunk about the bridge entity receives mass from
   the first-hop chunk that mentions it, even if the query never names it.

The result is just another ranked list: fuse it with BM25 and vectors through
:class:`ragforge.fusion.FusionRetriever`. Extraction is a plain heuristic by default
(no model, no network); pass ``extract_fn`` -- e.g. an LLM-backed one -- for better recall.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from ragforge.chunking import Chunk
from ragforge.index import ENGLISH_STOPWORDS, FilterFn, ScoredChunk

ExtractFn = Callable[[str], list[str]]

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’_.\-]*[A-Za-z0-9]|[A-Za-z0-9]")
_CONNECTORS = frozenset({"of", "the", "de", "la", "von", "van", "for", "du", "del"})


def _is_capitalized(token: str) -> bool:
    return token[0].isupper() or (token[0].isdigit() and any(c.isalpha() for c in token))


def _is_identifier(token: str) -> bool:
    return (
        any(c.isdigit() for c in token)
        and any(c.isalpha() for c in token)
        and ("-" in token or "_" in token or token.isupper())
    )


def extract_entities(text: str) -> list[str]:
    """Heuristic entity mentions: capitalized runs, acronyms and identifiers (display form).

    Sentence-initial common words are ignored; ``of/the/de/...`` may connect two
    capitalized tokens ("Bank of England"); identifiers always stand alone. Mentions come
    in order of first appearance, duplicates removed.
    """
    tokens = list(_TOKEN.finditer(text))
    entities: list[str] = []
    i = 0
    while i < len(tokens):
        word = tokens[i].group()
        sentence_start = i == 0 or text[: tokens[i].start()].rstrip().endswith(
            (".", "!", "?", "\n")
        )
        if _is_identifier(word):
            entities.append(word)
            i += 1
            continue
        if not _is_capitalized(word) or (
            sentence_start and word.lower() in ENGLISH_STOPWORDS and word.isalpha()
        ):
            i += 1
            continue
        j = i + 1
        while j < len(tokens):
            nxt = tokens[j].group()
            gap = text[tokens[j - 1].end() : tokens[j].start()]
            if gap.strip(" \t'’-") != "" or _is_identifier(nxt):
                break
            if _is_capitalized(nxt):
                j += 1
            elif (
                nxt.lower() in _CONNECTORS
                and j + 1 < len(tokens)
                and _is_capitalized(tokens[j + 1].group())
                and text[tokens[j].end() : tokens[j + 1].start()].strip() == ""
            ):
                j += 2
            else:
                break
        phrase = text[tokens[i].start() : tokens[j - 1].end()]
        phrase = re.sub(r"['’]s$", "", phrase)
        if not (
            len(phrase) == 1 or (len(phrase.split()) == 1 and phrase.lower() in ENGLISH_STOPWORDS)
        ):
            entities.append(phrase)
        i = j
    seen: set[str] = set()
    unique = []
    for entity in entities:
        key = entity_key(entity)
        if key not in seen:
            seen.add(key)
            unique.append(entity)
    return unique


def entity_key(entity: str) -> str:
    """Canonical form used to match mentions: lowercased, possessives and extra spaces removed."""
    return re.sub(r"\s+", " ", re.sub(r"['’]s\b", "", entity.lower())).strip()


class GraphIndex:
    """Bipartite chunk/entity graph with personalized-PageRank retrieval.

    Args:
        extract_fn: Text -> entity mentions (default :func:`extract_entities`).
        steps: Diffusion steps; each hop alternates entity -> chunk -> entity (4 reaches the
            second document of a two-hop question).
        decay: Mass retained per step (the rest returns to the seed distribution).
        max_entity_df: Entities mentioned by more than this fraction of chunks are ignored as
            uninformative hubs (enforced once the corpus has ``min_hub_corpus`` chunks).
    """

    def __init__(
        self,
        extract_fn: ExtractFn = extract_entities,
        steps: int = 4,
        decay: float = 0.6,
        max_entity_df: float = 0.2,
        min_hub_corpus: int = 50,
    ) -> None:
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        if not 0.0 < max_entity_df <= 1.0:
            raise ValueError(f"max_entity_df must be in (0, 1], got {max_entity_df}")
        self.extract_fn = extract_fn
        self.steps = steps
        self.decay = decay
        self.max_entity_df = max_entity_df
        self.min_hub_corpus = min_hub_corpus
        self._chunks: list[Chunk] = []
        self._entities_of: list[list[str]] = []  # chunk position -> entity keys
        self._chunks_of: dict[str, list[int]] = defaultdict(list)  # entity key -> positions
        self._display: dict[str, str] = {}
        self._token_index: dict[str, set[str]] | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    # -- building ----------------------------------------------------------------------

    def _append(self, chunk: Chunk, mentions: list[str]) -> None:
        position = len(self._chunks)
        keys = []
        for mention in mentions:
            key = entity_key(mention)
            if key and key not in keys:
                keys.append(key)
                self._display.setdefault(key, mention)
                self._chunks_of[key].append(position)
        self._chunks.append(chunk)
        self._entities_of.append(keys)
        self._token_index = None

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._append(chunk, self.extract_fn(chunk.index_text))

    def delete(self, doc_id: str) -> int:
        kept = [
            (c, [self._display[k] for k in keys])
            for c, keys in zip(self._chunks, self._entities_of, strict=True)
            if c.doc_id != doc_id
        ]
        removed = len(self._chunks) - len(kept)
        if removed:
            self.clear()
            for chunk, mentions in kept:
                self._append(chunk, mentions)
        return removed

    def clear(self) -> None:
        self._chunks, self._entities_of = [], []
        self._chunks_of, self._display = defaultdict(list), {}
        self._token_index = None

    # -- inspection --------------------------------------------------------------------

    def _usable(self, key: str) -> bool:
        df = len(self._chunks_of.get(key, ()))
        if df == 0:
            return False
        too_common = df / len(self._chunks) > self.max_entity_df
        return not (len(self._chunks) >= self.min_hub_corpus and too_common)

    def _idf(self, key: str) -> float:
        return math.log(1.0 + len(self._chunks) / len(self._chunks_of[key]))

    def entities(self, min_df: int = 1) -> list[tuple[str, int]]:
        """``(display name, number of chunks)`` for usable entities, most connected first."""
        rows = [
            (self._display[k], len(v))
            for k, v in self._chunks_of.items()
            if len(v) >= min_df and self._usable(k)
        ]
        return sorted(rows, key=lambda r: (-r[1], r[0]))

    def graph_data(
        self, max_entities: int = 60, min_df: int = 2, include: Iterable[str] = ()
    ) -> dict[str, Any]:
        """Entities and entity-entity links (co-mentions) for visualization.

        The ``max_entities`` most connected entities are returned, plus every key in ``include``
        (e.g. the entities a query activated) so they are never cut off.
        """
        forced = [k for k in include if k in self._chunks_of]
        by_degree = sorted(self._chunks_of.items(), key=lambda kv: -len(kv[1]))
        top = [k for k, v in by_degree if len(v) >= min_df and self._usable(k) and k not in forced][
            :max_entities
        ]
        keep = set(top) | set(forced)
        edges: dict[tuple[str, str], int] = defaultdict(int)
        for keys in self._entities_of:
            present = sorted(k for k in keys if k in keep)
            for a in range(len(present)):
                for b in range(a + 1, len(present)):
                    edges[(present[a], present[b])] += 1
        ordered = [*forced, *top]
        return {
            "nodes": [
                {"id": k, "label": self._display[k], "degree": len(self._chunks_of[k]),
                 "docs": sorted({self._chunks[p].doc_id for p in self._chunks_of[k]})[:6]}
                for k in ordered
            ],
            "edges": [
                {"source": a, "target": b, "weight": w} for (a, b), w in sorted(edges.items())
            ],
        }  # fmt: skip

    # -- retrieval ---------------------------------------------------------------------

    def _seeds(self, query: str) -> dict[str, float]:
        seeds: dict[str, float] = {}
        for mention in self.extract_fn(query):
            key = entity_key(mention)
            if key in self._chunks_of and self._usable(key):
                seeds[key] = self._idf(key)
        if (
            not seeds
        ):  # no recognizable mention: fall back to words that appear inside an entity name
            if self._token_index is None:
                index: dict[str, set[str]] = defaultdict(set)
                for key in self._chunks_of:
                    for token in key.split():
                        index[token].add(key)
                self._token_index = index
            for word in re.findall(r"[A-Za-z0-9][\w'’.\-]*", query.lower()):
                token = entity_key(word)
                if token in ENGLISH_STOPWORDS:
                    continue
                for key in self._token_index.get(token, ()):
                    if self._usable(key):
                        seeds[key] = max(seeds.get(key, 0.0), self._idf(key))
        return seeds

    def _diffuse(self, query: str) -> tuple[dict[str, float], dict[int, float], dict[str, float]]:
        """Run the diffusion; returns ``(seeds, chunk_scores, entity_activation)``."""
        seeds = self._seeds(query)
        if not seeds:
            return {}, {}, {}
        total = sum(seeds.values())
        restart = {key: w / total for key, w in seeds.items()}

        entity_mass = dict(restart)
        activation: dict[str, float] = defaultdict(float)
        chunk_score: dict[int, float] = defaultdict(float)
        for step in range(self.steps):
            for key, mass in entity_mass.items():
                activation[key] += mass * (self.decay**step)
            # entity -> chunk: split mass over the entity's chunks
            chunk_mass: dict[int, float] = defaultdict(float)
            for key, mass in entity_mass.items():
                positions = self._chunks_of[key]
                share = mass / len(positions)
                for position in positions:
                    chunk_mass[position] += share
            for position, mass in chunk_mass.items():
                chunk_score[position] += mass * (self.decay**step)
            if step == self.steps - 1:
                break
            # chunk -> entity: split mass over the chunk's usable entities, then restart
            next_mass: dict[str, float] = defaultdict(float)
            for position, mass in chunk_mass.items():
                usable = [k for k in self._entities_of[position] if self._usable(k)]
                if not usable:
                    continue
                share = mass / len(usable)
                for key in usable:
                    next_mass[key] += share * self.decay
            for key, mass in restart.items():
                next_mass[key] += (1.0 - self.decay) * mass
            entity_mass = next_mass
        return seeds, chunk_score, dict(activation)

    def search(
        self, query: str, k: int = 5, filter_fn: FilterFn | None = None
    ) -> list[ScoredChunk]:
        """Chunks ranked by activation after diffusing the query's entities through the graph."""
        if k <= 0 or not self._chunks:
            return []
        _, chunk_score, _ = self._diffuse(query)
        candidates = (
            (-score, position)
            for position, score in chunk_score.items()
            if score > 1e-12 and (filter_fn is None or filter_fn(self._chunks[position]))
        )
        return [
            ScoredChunk(chunk=self._chunks[pos], score=-neg, provenance="graph")
            for neg, pos in heapq.nsmallest(k, candidates)
        ]

    def explain(self, query: str) -> dict[str, Any]:
        """Which entities the query activates and how strongly the diffusion reached each one.

        Returns ``{"seeds": [{"entity", "key", "weight"}], "activation": {key: mass},
        "chunks": [{"chunk_id", "doc_id", "score"}]}`` -- the data behind a graph visualization.
        """
        seeds, chunk_score, activation = self._diffuse(query)
        total = sum(activation.values()) or 1.0
        return {
            "seeds": [
                {"entity": self._display[key], "key": key, "weight": weight}
                for key, weight in seeds.items()
            ],
            "activation": {key: mass / total for key, mass in activation.items()},
            "chunks": [
                {"chunk_id": self._chunks[p].id, "doc_id": self._chunks[p].doc_id, "score": score}
                for p, score in sorted(chunk_score.items(), key=lambda kv: -kv[1])
            ],
        }

    # -- persistence -------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": [c.to_dict() for c in self._chunks],
            "entities": [[self._display[k] for k in keys] for keys in self._entities_of],
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self.clear()
        for chunk_data, mentions in zip(
            data.get("chunks", []), data.get("entities", []), strict=True
        ):
            self._append(Chunk.from_dict(chunk_data), list(mentions))
