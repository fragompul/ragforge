import pytest

from ragforge.chunking import FixedSizeChunker, SentenceChunker


def test_fixed_size_chunker_splits_by_word_count():
    text = " ".join(f"word{i}" for i in range(25))
    chunker = FixedSizeChunker(chunk_size=10, overlap=2)

    chunks = chunker.chunk(text, doc_id="doc1")

    assert all(c.doc_id == "doc1" for c in chunks)
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


def test_fixed_size_chunker_rejects_overlap_gte_chunk_size():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=10, overlap=10)


def test_sentence_chunker_keeps_sentences_intact():
    text = "First sentence here. Second sentence here. Third sentence here."
    chunker = SentenceChunker(max_chars=30, overlap_sentences=0)

    chunks = chunker.chunk(text, doc_id="doc1")

    for chunk in chunks:
        assert chunk.text.strip().endswith(".")


def test_sentence_chunker_overlaps_by_sentence_count():
    # Six equal-length sentences and a max_chars sized for exactly two
    # sentences per chunk means the overlap is never clipped, so every
    # chunk boundary overlaps by exactly one sentence.
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


def test_sentence_chunker_rejects_negative_overlap():
    with pytest.raises(ValueError):
        SentenceChunker(overlap_sentences=-1)
