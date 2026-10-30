from typing import Dict, Any, Optional, Callable, List
from app.chunking.base import BaseChunker
from app.chunking.fixed_chunker import FixedChunker
from app.chunking.recursive_chunker import RecursiveChunker
from app.chunking.semantic_chunker import SemanticChunker


class ChunkerFactory:
    """Factory to instantiate chunking strategies with given parameters."""

    @staticmethod
    def create(
        strategy: str,
        params: Dict[str, Any] | None = None,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None
    ) -> BaseChunker:
        params = params or {}
        strategy_lower = strategy.strip().lower()

        if strategy_lower == "fixed":
            return FixedChunker(
                chunk_size=params.get("chunk_size", 500),
                chunk_overlap=params.get("chunk_overlap", 100),
                **{k: v for k, v in params.items() if k not in ["chunk_size", "chunk_overlap"]}
            )
        elif strategy_lower == "recursive":
            return RecursiveChunker(
                chunk_size=params.get("chunk_size", 500),
                chunk_overlap=params.get("chunk_overlap", 100),
                separators=params.get("separators", None),
                **{k: v for k, v in params.items() if k not in ["chunk_size", "chunk_overlap", "separators"]}
            )
        elif strategy_lower == "semantic":
            chunker = SemanticChunker(
                similarity_threshold=params.get("similarity_threshold", 0.70),
                min_chunk_size=params.get("min_chunk_size", 150),
                max_chunk_size=params.get("max_chunk_size", 1200),
                embedding_fn=embedding_fn,
                **{k: v for k, v in params.items() if k not in ["similarity_threshold", "min_chunk_size", "max_chunk_size"]}
            )
            return chunker
        else:
            raise ValueError(f"Unknown chunking strategy: '{strategy}'. Supported: 'fixed', 'recursive', 'semantic'.")
