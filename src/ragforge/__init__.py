"""ragforge: production-hardened, dependency-minimal RAG pipeline."""

from ragforge.ann import HNSWIndex
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
from ragforge.embeddings_providers import (
    cached_embed_fn,
    cohere_embed_fn,
    ollama_embed_fn,
    openai_embed_fn,
    sentence_transformers_embed_fn,
)
from ragforge.evaluation import (
    EvalCase,
    EvalResult,
    EvaluationSummary,
    evaluate_pipeline,
)
from ragforge.index import (
    ApproxVectorIndex,
    BM25Index,
    FilterFn,
    HybridRetriever,
    ScoredChunk,
    VectorIndex,
    VectorSearchable,
)
from ragforge.pipeline import (
    GenerateFn,
    PromptFormatter,
    RagAnswer,
    RagPipeline,
    RetrievedChunk,
    default_prompt_formatter,
)
from ragforge.query_expansion import (
    MultiQueryRetriever,
    QueryExpansionFn,
    identity_expansion,
    llm_query_expansion_fn,
)
from ragforge.reranking import (
    CrossEncoderReranker,
    HeuristicReranker,
    MaxMarginalRelevanceReranker,
    NoopReranker,
    Reranker,
)
from ragforge.server import serve, serve_forever_blocking
from ragforge.telemetry import (
    InMemoryExporter,
    Span,
    SpanExporter,
    Tracer,
    console_exporter,
    otel_exporter,
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
    # Real embedding provider adapters (optional dependencies)
    "openai_embed_fn",
    "cohere_embed_fn",
    "sentence_transformers_embed_fn",
    "ollama_embed_fn",
    "cached_embed_fn",
    # Indices & Retrieval
    "BM25Index",
    "VectorIndex",
    "ApproxVectorIndex",
    "VectorSearchable",
    "HNSWIndex",
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
    # Query Expansion
    "MultiQueryRetriever",
    "QueryExpansionFn",
    "identity_expansion",
    "llm_query_expansion_fn",
    # Telemetry
    "Tracer",
    "Span",
    "SpanExporter",
    "console_exporter",
    "InMemoryExporter",
    "otel_exporter",
    # Serving
    "serve",
    "serve_forever_blocking",
]
