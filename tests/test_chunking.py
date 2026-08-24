import pytest

from ragforge.chunking import (
    Chunk,
    FixedSizeChunker,
    MarkdownChunker,
    RecursiveCharacterChunker,
    SentenceChunker,
)


def test_chunk_serialization():
    chunk = Chunk(
        id="doc1::0",
        text="Sample text content",
        doc_id="doc1",
        position=0,
        metadata={"category": "ai", "score": 0.95},
        start_char=0,
        end_char=19,
    )
    data = chunk.to_dict()
    restored = Chunk.from_dict(data)

    assert restored.id == chunk.id
    assert restored.text == chunk.text
    assert restored.doc_id == chunk.doc_id
    assert restored.position == chunk.position
    assert restored.metadata == chunk.metadata
    assert restored.start_char == chunk.start_char
    assert restored.end_char == chunk.end_char


def test_fixed_size_chunker_splits_by_word_count():
    text = " ".join(f"word{i}" for i in range(25))
    chunker = FixedSizeChunker(chunk_size=10, overlap=2)

    chunks = chunker.chunk(text, doc_id="doc1", metadata={"source": "test"})

    assert len(chunks) > 1
    assert all(c.doc_id == "doc1" for c in chunks)
    assert all(c.metadata["source"] == "test" for c in chunks)
    assert len(chunks[0].text.split()) == 10
    # consecutive chunks overlap by `overlap` words
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]


def test_fixed_size_chunker_handles_short_text():
    chunker = FixedSizeChunker(chunk_size=100, overlap=10)
    chunks = chunker.chunk("just a few words", doc_id="doc1")
    assert len(chunks) == 1
    assert chunks[0].text == "just a few words"


def test_fixed_size_chunker_empty_text_returns_no_chunks():
    chunker = FixedSizeChunker()
    assert chunker.chunk("", doc_id="doc1") == []
    assert chunker.chunk("   ", doc_id="doc1") == []


def test_fixed_size_chunker_parameter_validations():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        FixedSizeChunker(chunk_size=0, overlap=0)
    with pytest.raises(ValueError, match="overlap must be non-negative"):
        FixedSizeChunker(chunk_size=10, overlap=-1)
    with pytest.raises(ValueError, match="overlap .* must be smaller than chunk_size"):
        FixedSizeChunker(chunk_size=10, overlap=10)


def test_sentence_chunker_keeps_sentences_intact():
    text = "First sentence here. Second sentence here. Third sentence here."
    chunker = SentenceChunker(max_chars=30, overlap_sentences=0)

    chunks = chunker.chunk(text, doc_id="doc1")

    for chunk in chunks:
        assert chunk.text.strip().endswith(".")


def test_sentence_chunker_single_giant_sentence():
    giant = "A" * 500 + "."
    chunker = SentenceChunker(max_chars=100, overlap_sentences=1)
    chunks = chunker.chunk(giant, doc_id="doc1")
    assert len(chunks) == 1
    assert chunks[0].text == giant


def test_sentence_chunker_overlaps_by_sentence_count():
    text = "Aa. Bb. Cc. Dd. Ee. Ff."
    chunker = SentenceChunker(max_chars=7, overlap_sentences=1)

    chunks = chunker.chunk(text, doc_id="doc1")

    assert len(chunks) == 5
    for prev, curr in zip(chunks, chunks[1:], strict=False):
        last_sentence = prev.text.rsplit(" ", 1)[-1]
        first_sentence = curr.text.split(" ", 1)[0]
        assert last_sentence == first_sentence


def test_sentence_chunker_empty_text_returns_no_chunks():
    chunker = SentenceChunker()
    assert chunker.chunk("   ", doc_id="doc1") == []


def test_sentence_chunker_validations():
    with pytest.raises(ValueError, match="max_chars must be positive"):
        SentenceChunker(max_chars=0)
    with pytest.raises(ValueError, match="overlap_sentences must be >= 0"):
        SentenceChunker(overlap_sentences=-1)


def test_recursive_character_chunker_splits_hierarchically():
    text = (
        "Paragraph 1 contains introductory concepts and design ideas.\n\n"
        "Paragraph 2 discusses distributed architectures and consensus mechanisms.\n\n"
        "Paragraph 3 covers latency optimization and memory caching."
    )
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(text, doc_id="doc_recursive", metadata={"tier": "production"})

    assert len(chunks) >= 3
    assert all(c.doc_id == "doc_recursive" for c in chunks)
    assert all(c.metadata["tier"] == "production" for c in chunks)
    assert all(len(c.text) <= 120 for c in chunks)
    assert all(c.start_char >= 0 and c.end_char > c.start_char for c in chunks)


def test_recursive_character_chunker_deep_recursion_and_character_split():
    # Long text with no spaces to force character-level splitting
    text = "abcdefghijklmnopqrstuvwxyz" * 10
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk(text, doc_id="doc_chars")
    assert len(chunks) > 1
    assert all(len(c.text) <= 50 for c in chunks)


def test_recursive_character_chunker_empty_and_validations():
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.chunk("", doc_id="doc1") == []

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        RecursiveCharacterChunker(chunk_size=-5)
    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=-1)
    with pytest.raises(ValueError, match="chunk_overlap .* must be smaller than chunk_size"):
        RecursiveCharacterChunker(chunk_size=50, chunk_overlap=50)


def test_markdown_chunker_preserves_section_headers():
    doc = (
        "Introductory preamble before any header appears.\n\n"
        "# Overview\n"
        "This is the general introduction to the library.\n\n"
        "## Architecture\n"
        "The architecture contains ingestion, indexing, and reranking stages.\n\n"
        "## Large Section\n" + ("Detailed explanation of distributed indexing and search. " * 30)
    )
    chunker = MarkdownChunker(max_chars=200)
    chunks = chunker.chunk(doc, doc_id="readme", metadata={"repo": "ragforge"})

    assert len(chunks) >= 4
    headers = [c.metadata.get("section") for c in chunks]
    assert "Introduction" in headers
    assert "Overview" in headers
    assert "Architecture" in headers
    assert "Large Section" in headers


def test_markdown_chunker_fallback_and_validation():
    plain = "Just plain prose without any markdown headers.\n\nSecond paragraph."
    chunker = MarkdownChunker(max_chars=200)
    chunks = chunker.chunk(plain, doc_id="plain")
    assert len(chunks) >= 1
    assert chunks[0].doc_id == "plain"
    assert chunker.chunk("", doc_id="empty") == []

    with pytest.raises(ValueError, match="max_chars must be positive"):
        MarkdownChunker(max_chars=0)
