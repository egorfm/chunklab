import logging
import numpy as np
from typing import List, Dict, Optional, Any
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages text embedding generation using SentenceTransformers models with lazy loading."""

    _model_cache: Dict[str, Any] = {}

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device or self._detect_device()
        self._model = None
        self._dimension = None

    def _detect_device(self) -> str:
        if settings.EMBEDDING_DEVICE and settings.EMBEDDING_DEVICE != "auto":
            return settings.EMBEDDING_DEVICE
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    @property
    def model(self):
        if self.model_name not in EmbeddingService._model_cache:
            try:
                from sentence_transformers import SentenceTransformer
                # Try local files first for speed and sandboxed environments
                try:
                    model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
                except Exception:
                    # If not locally cached and allowed, try standard load
                    model = SentenceTransformer(self.model_name, device=self.device)
                EmbeddingService._model_cache[self.model_name] = model
                logger.info(f"Loaded embedding model '{self.model_name}' on device '{self.device}'")
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer '{self.model_name}': {e}. Using deterministic fallback.")
                EmbeddingService._model_cache[self.model_name] = None


        return EmbeddingService._model_cache.get(self.model_name)


    def get_dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension

        loaded_model = self.model
        if loaded_model is not None:
            dim = loaded_model.get_sentence_embedding_dimension()
            self._dimension = int(dim)
            return self._dimension

        # Fallback dimension (e.g. for all-MiniLM-L6-v2 it is 384, for bge-m3 / e5-base it's 768 or 1024)
        if "bge-m3" in self.model_name.lower():
            self._dimension = 1024
        elif "e5-base" in self.model_name.lower():
            self._dimension = 768
        else:
            self._dimension = 384
        return self._dimension

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []

        loaded_model = self.model
        if loaded_model is not None:
            try:
                embeddings = loaded_model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Error encoding texts with model {self.model_name}: {e}")

        # Deterministic fallback hashing embeddings for offline / fast unit testing
        dim = self.get_dimension()
        result = []
        for text in texts:
            vec = np.zeros(dim, dtype=np.float32)
            for idx, char in enumerate(text[:dim]):
                vec[idx % dim] += ord(char)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            result.append(vec.tolist())
        return result

    def embed_query(self, query: str) -> List[float]:
        # Some models like multilingual-e5 require "query: " prefix for asymmetric search
        query_text = query
        if "e5" in self.model_name.lower() and not query.lower().startswith("query:"):
            query_text = f"query: {query}"
        return self.embed_texts([query_text])[0]
