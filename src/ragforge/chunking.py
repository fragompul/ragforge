"""Chunking strategies for splitting source documents into retrievable units.

Chunk size and overlap are the first lever that determines RAG quality in
production: chunks too large dilute retrieval precision (a query matches
a chunk that only partially answers it); chunks too small lose context
needed to answer the question at all. Two strategies are provided so the
tradeoff is explicit rather than hidden behind a single default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


class FixedSizeChunker:
    """Splits text into fixed-size, word-count-based chunks with overlap.

    Simple and predictable -- a reasonable default for homogeneous,
    unstructured text where sentence boundaries don't carry much meaning
    (logs, transcripts, code).
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 40) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        words = text.split()
        if not words:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        position = 0
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                Chunk(id=f"{doc_id}::{position}", text=chunk_text, doc_id=doc_id, position=position)
            )
            position += 1
            if end == len(words):
                break
            start += step
        return chunks


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class SentenceChunker:
    """Groups whole sentences into chunks up to ``max_chars``, with a
    sentence-count overlap between consecutive chunks.

    Preferred for prose (docs, articles, support content) where cutting a
    chunk mid-sentence loses meaning that a fixed word count would
    otherwise discard.
    """

    def __init__(self, max_chars: int = 800, overlap_sentences: int = 1) -> None:
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]

    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        position = 0
        i = 0
        while i < len(sentences):
            current: list[str] = []
            length = 0
            j = i
            while j < len(sentences) and (
                length + len(sentences[j]) <= self.max_chars or not current
            ):
                current.append(sentences[j])
                length += len(sentences[j]) + 1
                j += 1

            chunks.append(
                Chunk(
                    id=f"{doc_id}::{position}",
                    text=" ".join(current),
                    doc_id=doc_id,
                    position=position,
                )
            )
            position += 1

            if j >= len(sentences):
                break
            i = max(i + 1, j - self.overlap_sentences)
        return chunks
