"""Contextual retrieval: situate each chunk in its document before indexing it.

A chunk like "The company's revenue grew by 3% over the previous quarter." is unfindable
for the question "What was ACME Corp's Q2 2023 revenue growth?" -- the chunk never says
which company or quarter. *Contextual retrieval* (popularized by Anthropic, 2024) fixes
this at indexing time by prepending a short description of where the chunk sits in its
document, to the text that BM25 and the embedder see (``Chunk.index_text``) while answers
and citations keep using the original text.

A *contextualizer* is any ``(document_text, chunk) -> str``. Provided here:

* :func:`title_context_fn` / :func:`heading_context_fn` / :func:`lead_context_fn`:
  deterministic, free heuristics (document title, markdown heading path, opening sentence);
* :func:`llm_context_fn`: the LLM variant -- one call per chunk, using the whole document.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ragforge.chunking import Chunk

ContextFn = Callable[[str, Chunk], str]
CompleteFn = Callable[[str], str]

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def title_context_fn(title_key: str = "title") -> ContextFn:
    """Context = the document title, read from chunk metadata (``title_key``) or the doc id."""

    def _context(document: str, chunk: Chunk) -> str:
        title = chunk.metadata.get(title_key)
        return f"{title}" if title else chunk.doc_id.replace("-", " ").replace("_", " ")

    return _context


def heading_context_fn(document: str, chunk: Chunk) -> str:
    """Context = the markdown heading path above the chunk (``Guide > Install > Linux``)."""
    path: list[tuple[int, str]] = []
    for match in _HEADING.finditer(document):
        if match.start() > chunk.start_char:
            break
        level = len(match.group(1))
        path = [(lv, t) for lv, t in path if lv < level]
        path.append((level, match.group(2)))
    return " > ".join(title for _, title in path)


def lead_context_fn(max_chars: int = 160) -> ContextFn:
    """Context = the document's opening sentence (skipped for the chunk that contains it)."""

    def _context(document: str, chunk: Chunk) -> str:
        first = re.split(r"(?<=[.!?])\s+|\n", document.strip(), maxsplit=1)[0][:max_chars]
        return "" if chunk.start_char == 0 or first in chunk.text else first

    return _context


def combine_context_fns(*fns: ContextFn) -> ContextFn:
    """Join several contexts with ``" | "``, skipping empty ones."""

    def _context(document: str, chunk: Chunk) -> str:
        return " | ".join(c for fn in fns if (c := fn(document, chunk)))

    return _context


def llm_context_fn(complete_fn: CompleteFn, max_document_chars: int = 12000) -> ContextFn:
    """Model-backed context: "give a short context to situate this chunk in the document".

    Costs one model call per chunk (use prompt caching on the document if your provider
    supports it). The document is truncated to ``max_document_chars``.
    """

    def _context(document: str, chunk: Chunk) -> str:
        prompt = (
            "<document>\n"
            f"{document[:max_document_chars]}\n"
            "</document>\n"
            "Here is a chunk from that document:\n"
            f"<chunk>\n{chunk.text}\n</chunk>\n"
            "Write a short, succinct context (one or two sentences) that situates this chunk "
            "within the whole document to improve search retrieval of the chunk. "
            "Answer with the context only."
        )
        return complete_fn(prompt).strip()

    return _context
