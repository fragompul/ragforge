"""Chunking strategies for splitting source documents into retrievable units.

Chunk size and overlap are the primary levers determining retrieval quality in production RAG:
- Chunks that are too large dilute retrieval precision (a query matches a chunk that only
  incidentally mentions the topic amidst irrelevant context).
- Chunks that are too small discard necessary context required for faithful generation.

Three distinct, production-grade strategies are provided:
1. ``FixedSizeChunker``: Token/word-count splitting with overlap for homogeneous text.
2. ``SentenceChunker``: Sentence-boundary-aware chunking preserving semantic sentences.
3. ``RecursiveCharacterChunker``: Hierarchical multi-separator splitting
   (paragraphs -> sentences -> words) which is the gold standard for arbitrary structure.
4. ``MarkdownChunker``: Header-aware chunking for structured markdown documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Chunk:
    """A retrievable unit of text with provenance and metadata."""

    id: str
    text: str
    doc_id: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "doc_id": self.doc_id,
            "position": self.position,
            "metadata": dict(self.metadata),
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            id=data["id"],
            text=data["text"],
            doc_id=data["doc_id"],
            position=data.get("position", 0),
            metadata=data.get("metadata", {}),
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0),
        )


class Chunker(Protocol):
    """Protocol for document chunking strategies."""

    def chunk(
        self, text: str, doc_id: str, metadata: dict[str, Any] | None = None
    ) -> list[Chunk]: ...


class FixedSizeChunker:
    """Splits text into fixed-size word-count chunks with sliding window overlap."""

    def __init__(self, chunk_size: int = 200, overlap: int = 40) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        clean_text = text.strip()
        if not clean_text:
            return []

        words = clean_text.split()
        if not words:
            return []

        meta = dict(metadata or {})
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        position = 0
        start = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append(
                Chunk(
                    id=f"{doc_id}::{position}",
                    text=chunk_text,
                    doc_id=doc_id,
                    position=position,
                    metadata=meta.copy(),
                    start_char=0,
                    end_char=len(chunk_text),
                )
            )
            position += 1
            if end == len(words):
                break
            start += step

        return chunks


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class SentenceChunker:
    """Groups whole sentences into chunks up to ``max_chars`` with sentence overlap."""

    def __init__(self, max_chars: int = 800, overlap_sentences: int = 1) -> None:
        if max_chars <= 0:
            raise ValueError(f"max_chars must be positive, got {max_chars}")
        if overlap_sentences < 0:
            raise ValueError(f"overlap_sentences must be >= 0, got {overlap_sentences}")

        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]

    def chunk(self, text: str, doc_id: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        meta = dict(metadata or {})
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

            chunk_text = " ".join(current)
            chunks.append(
                Chunk(
                    id=f"{doc_id}::{position}",
                    text=chunk_text,
                    doc_id=doc_id,
                    position=position,
                    metadata=meta.copy(),
                    start_char=0,
                    end_char=len(chunk_text),
                )
            )
            position += 1

            if j >= len(sentences):
                break
            i = max(i + 1, j - self.overlap_sentences)

        return chunks


class RecursiveCharacterChunker:
    """Hierarchical text splitter that recursively splits text across multiple separators.

    Separators are tried in priority order (e.g. paragraphs -> lines -> sentences
    -> words -> characters).
    This ensures that natural semantic boundaries (paragraphs, sentences) are preserved
    whenever possible before breaking down into smaller sub-segments.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []

        separator = separators[-1]
        new_separators: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)

        final_chunks: list[str] = []
        good_splits: list[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if new_separators:
                    sub_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(s)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        merged: list[str] = []
        current: list[str] = []
        total_len = 0

        for split in splits:
            split_len = len(split)
            sep_len = len(separator) if current else 0
            if total_len + sep_len + split_len > self.chunk_size and current:
                doc = separator.join(current).strip()
                if doc:
                    merged.append(doc)

                # Keep overlap items from the end of current
                while current and total_len > self.chunk_overlap:
                    total_len -= len(current[0]) + (len(separator) if len(current) > 1 else 0)
                    current.pop(0)

            current.append(split)
            total_len += split_len + (len(separator) if len(current) > 1 else 0)

        if current:
            doc = separator.join(current).strip()
            if doc:
                merged.append(doc)

        return merged

    def chunk(self, text: str, doc_id: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        clean_text = text.strip()
        if not clean_text:
            return []

        raw_chunks = self._split_text(clean_text, self.separators)
        meta = dict(metadata or {})
        chunks: list[Chunk] = []

        start_search_idx = 0
        for position, chunk_text in enumerate(raw_chunks):
            start_pos = clean_text.find(chunk_text, start_search_idx)
            if start_pos == -1:
                start_pos = start_search_idx
            end_pos = start_pos + len(chunk_text)
            start_search_idx = max(0, end_pos - self.chunk_overlap)

            chunks.append(
                Chunk(
                    id=f"{doc_id}::{position}",
                    text=chunk_text,
                    doc_id=doc_id,
                    position=position,
                    metadata=meta.copy(),
                    start_char=start_pos,
                    end_char=end_pos,
                )
            )

        return chunks


class MarkdownChunker:
    """Header-aware Markdown chunker that preserves section headers in chunk metadata."""

    _HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(self, max_chars: int = 1000, min_chunk_chars: int = 50) -> None:
        if max_chars <= 0:
            raise ValueError(f"max_chars must be positive, got {max_chars}")
        self.max_chars = max_chars
        self.min_chunk_chars = min_chunk_chars
        self._fallback_chunker = RecursiveCharacterChunker(chunk_size=max_chars, chunk_overlap=100)

    def chunk(self, text: str, doc_id: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        clean_text = text.strip()
        if not clean_text:
            return []

        base_meta = dict(metadata or {})
        matches = list(self._HEADER_PATTERN.finditer(clean_text))
        if not matches:
            return self._fallback_chunker.chunk(clean_text, doc_id, metadata=base_meta)

        sections: list[tuple[str, str, int, int]] = []
        # Handle intro before first header
        if matches[0].start() > 0:
            intro_text = clean_text[: matches[0].start()].strip()
            if intro_text:
                sections.append(("Introduction", intro_text, 0, matches[0].start()))

        for i, match in enumerate(matches):
            header_title = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(clean_text)
            section_text = clean_text[start:end].strip()
            if section_text:
                sections.append((header_title, section_text, start, end))

        chunks: list[Chunk] = []
        position = 0

        for header, sec_text, start_idx, end_idx in sections:
            sec_meta = base_meta.copy()
            sec_meta["section"] = header

            if len(sec_text) <= self.max_chars:
                chunks.append(
                    Chunk(
                        id=f"{doc_id}::{position}",
                        text=sec_text,
                        doc_id=doc_id,
                        position=position,
                        metadata=sec_meta,
                        start_char=start_idx,
                        end_char=end_idx,
                    )
                )
                position += 1
            else:
                sub_chunks = self._fallback_chunker.chunk(sec_text, doc_id, metadata=sec_meta)
                for sub in sub_chunks:
                    sub.id = f"{doc_id}::{position}"
                    sub.position = position
                    sub.start_char += start_idx
                    sub.end_char += start_idx
                    chunks.append(sub)
                    position += 1

        return chunks
