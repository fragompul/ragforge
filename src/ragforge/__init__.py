"""ragforge: production-hardened, dependency-minimal RAG pipeline."""

from ragforge._version import __version__
from ragforge.agent import AgenticRag, AgentResult, AgentStep, Grade
from ragforge.aio import AsyncRagPipeline
from ragforge.ann import HNSWIndex
from ragforge.asgi import create_app
from ragforge.cache import CachedPipeline, SemanticCache
from ragforge.chunking import (
    Chunk,
    Chunker,
    FixedSizeChunker,
    MarkdownChunker,
    RecursiveCharacterChunker,
    SentenceChunker,
)
from ragforge.citations import CitedAnswer, attribute, nli_cross_encoder_fn
from ragforge.contextual import (
    heading_context_fn,
    lead_context_fn,
    llm_context_fn,
    title_context_fn,
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
from ragforge.fast import NumpyVectorIndex
from ragforge.fusion import FusionRetriever, FusionSource
from ragforge.graph import GraphIndex, extract_entities
from ragforge.guardrails import Guardrails, InjectionDetector, PIIRedactor, spotlight
from ragforge.index import (
    ApproxVectorIndex,
    BM25Index,
    FilterFn,
    HybridRetriever,
    ScoredChunk,
    VectorIndex,
    VectorSearchable,
    english_analyzer,
)
from ragforge.judge import JudgeVerdict, evaluate_with_judge, judge
from ragforge.llm_providers import (
    anthropic_complete_fn,
    anthropic_generate_fn,
    anthropic_stream_fn,
    ollama_complete_fn,
    ollama_generate_fn,
    openai_complete_fn,
    openai_generate_fn,
    openai_stream_fn,
)
from ragforge.mcp_server import MCPServer, serve_stdio
from ragforge.pipeline import (
    GenerateFn,
    PromptFormatter,
    RagAnswer,
    RagPipeline,
    RetrievedChunk,
    default_prompt_formatter,
)
from ragforge.quantization import BinaryQuantizer, QuantizedVectorIndex, ScalarQuantizer
from ragforge.query_expansion import (
    MultiQueryRetriever,
    QueryExpansionFn,
    hyde_expansion_fn,
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
from ragforge.routing import QueryRouter
from ragforge.server import serve, serve_forever_blocking
from ragforge.sparse import SparseVectorIndex, lexical_sparse_encode_fn, splade_encode_fn
from ragforge.storage import DiskIndex
from ragforge.telemetry import (
    InMemoryExporter,
    Span,
    SpanExporter,
    Tracer,
    console_exporter,
    otel_exporter,
)
from ragforge.tenancy import MultiTenantRag, ScopedPipeline, TenantError

__all__ = [
    "__version__",
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
    "NumpyVectorIndex",
    "english_analyzer",
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
    # Quantization, sparse, graph, fusion, routing, contextual retrieval
    "QuantizedVectorIndex",
    "ScalarQuantizer",
    "BinaryQuantizer",
    "SparseVectorIndex",
    "lexical_sparse_encode_fn",
    "splade_encode_fn",
    "GraphIndex",
    "extract_entities",
    "FusionRetriever",
    "FusionSource",
    "QueryRouter",
    "hyde_expansion_fn",
    "title_context_fn",
    "heading_context_fn",
    "lead_context_fn",
    "llm_context_fn",
    # Storage, async, multi-tenancy
    "DiskIndex",
    "AsyncRagPipeline",
    "create_app",
    "MultiTenantRag",
    "ScopedPipeline",
    "TenantError",
    "nli_cross_encoder_fn",
    # Guardrails (prompt-injection screening, PII redaction)
    "Guardrails",
    "InjectionDetector",
    "PIIRedactor",
    "spotlight",
    # Citations, agents, caching
    "attribute",
    "CitedAnswer",
    "AgenticRag",
    "AgentResult",
    "AgentStep",
    "Grade",
    "SemanticCache",
    "CachedPipeline",
    # LLM-as-judge and provider adapters (optional dependencies)
    "judge",
    "JudgeVerdict",
    "evaluate_with_judge",
    "anthropic_complete_fn",
    "anthropic_generate_fn",
    "anthropic_stream_fn",
    "openai_complete_fn",
    "openai_generate_fn",
    "openai_stream_fn",
    "ollama_complete_fn",
    "ollama_generate_fn",
    # MCP
    "MCPServer",
    "serve_stdio",
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
