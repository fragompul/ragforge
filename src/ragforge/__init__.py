"""ragforge: production-hardened, dependency-minimal RAG pipeline."""

from ragforge.chunking import (
    Chunk,
    Chunker,
    FixedSizeChunker,
    MarkdownChunker,
    RecursiveCharacterChunker,
    SentenceChunker,
)
from ragforge.embeddings import (
    BatchEmbedFn,
    EmbedFn,
    cosine_similarity,
    hashing_embed,
    normalize_vector,
)
from ragforge.evaluation import (
    EvalCase,
    EvalResult,
    EvaluationSummary,
    evaluate_pipeline,
)
from ragforge.index import (
    BM25Index,
    FilterFn,
    HybridRetriever,
    ScoredChunk,
    VectorIndex,
)
from ragforge.pipeline import (
    GenerateFn,
    PromptFormatter,
    RagAnswer,
    RagPipeline,
    RetrievedChunk,
    default_prompt_formatter,
)
from ragforge.reranking import (
    CrossEncoderReranker,
    HeuristicReranker,
    MaxMarginalRelevanceReranker,
    NoopReranker,
    Reranker,
)

__version__ = "0.1.0"

__all__ = [
    # Chunking
    "Chunk",
    "Chunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveCharacterChunker",
    "MarkdownChunker",
    # Embeddings
    "EmbedFn",
    "BatchEmbedFn",
    "hashing_embed",
    "cosine_similarity",
    "normalize_vector",
    # Indices & Retrieval
    "BM25Index",
    "VectorIndex",
    "HybridRetriever",
    "ScoredChunk",
    "FilterFn",
    # Reranking
    "Reranker",
    "NoopReranker",
    "HeuristicReranker",
    "CrossEncoderReranker",
    "MaxMarginalRelevanceReranker",
    # Pipeline & Answers
    "RagPipeline",
    "RetrievedChunk",
    "RagAnswer",
    "GenerateFn",
    "PromptFormatter",
    "default_prompt_formatter",
    # Evaluation
    "EvalCase",
    "EvalResult",
    "EvaluationSummary",
    "evaluate_pipeline",
]
