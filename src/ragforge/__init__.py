"""ragforge: a production-hardened, dependency-minimal RAG pipeline."""

from ragforge.chunking import Chunk, FixedSizeChunker, SentenceChunker
from ragforge.embeddings import hashing_embed
from ragforge.evaluation import EvalCase, EvalResult, evaluate_pipeline
from ragforge.index import BM25Index, HybridRetriever, VectorIndex
from ragforge.pipeline import RagPipeline, RetrievedChunk
from ragforge.reranking import HeuristicReranker, NoopReranker, Reranker

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "FixedSizeChunker",
    "SentenceChunker",
    "hashing_embed",
    "BM25Index",
    "VectorIndex",
    "HybridRetriever",
    "Reranker",
    "HeuristicReranker",
    "NoopReranker",
    "RagPipeline",
    "RetrievedChunk",
    "EvalCase",
    "EvalResult",
    "evaluate_pipeline",
]
