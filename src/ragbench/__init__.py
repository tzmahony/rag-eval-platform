"""ragbench: a framework for rigorously evaluating RAG configurations and LLMs."""

from ragbench.types import Chunk, Document, EvalExample, LLMResponse, ScoredChunk

__version__ = "0.1.0"

__all__ = ["Chunk", "Document", "EvalExample", "LLMResponse", "ScoredChunk", "__version__"]
