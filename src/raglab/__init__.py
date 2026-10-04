"""grounded-rag-lab: citation-first RAG with guardrails and an offline eval harness."""

from .answer import GroundedAnswerer
from .embedding import OpenAIEmbedder, TfidfEmbedder, default_embedder
from .evaluate import EvalReport, GoldenCase, evaluate, load_golden, render_markdown
from .llm import ExtractiveLLM, OpenAICompatLLM, ScriptedLLM, Source, default_llm
from .pipeline import RagPipeline, load_documents
from .retrieval import HybridIndex
from .types import Answer, Chunk, Citation, Retrieved

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "EvalReport",
    "ExtractiveLLM",
    "GoldenCase",
    "GroundedAnswerer",
    "HybridIndex",
    "OpenAICompatLLM",
    "OpenAIEmbedder",
    "RagPipeline",
    "Retrieved",
    "ScriptedLLM",
    "Source",
    "TfidfEmbedder",
    "__version__",
    "default_embedder",
    "default_llm",
    "evaluate",
    "load_documents",
    "load_golden",
    "render_markdown",
]
