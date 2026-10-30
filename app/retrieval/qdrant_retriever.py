import uuid
import logging
from typing import List, Dict, Any, Optional
from app.config import settings
from app.chunking.base import Chunk

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest_models
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    QdrantClient = None
    rest_models = None
    Distance = None
    VectorParams = None
    PointStruct = None


class QdrantRetriever:
    """Manages Qdrant vector collections, point indexing, and similarity retrieval."""

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.client = self._init_client()

    def _init_client(self):
        if QdrantClient is None:
            logger.warning("qdrant_client not available.")
            return None

        # Try remote Qdrant if URL provided
        if self.url and self.url.strip() and self.url != ":memory:":
            try:
                client = QdrantClient(url=self.url, api_key=self.api_key, timeout=5)
                # Quick healthcheck
                client.get_collections()
                logger.info(f"Connected to Qdrant at {self.url}")
                return client
            except Exception as e:
                logger.warning(f"Could not connect to Qdrant at {self.url}: {e}. Falling back to local storage.")

        # Fallback to local persistent directory or in-memory
        try:
            settings.ensure_directories()
            client = QdrantClient(path=str(settings.QDRANT_STORAGE_PATH))
            logger.info(f"Initialized local Qdrant at {settings.QDRANT_STORAGE_PATH}")
            return client
        except Exception as e:
            logger.warning(f"Could not initialize local disk Qdrant: {e}. Falling back to in-memory Qdrant.")
            return QdrantClient(":memory:")

    def get_collection_name(self, document_id: int, strategy: str) -> str:
        clean_strat = strategy.lower().replace("-", "_")
        return f"doc_{document_id}_{clean_strat}"

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        if self.client is None:
            return

        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if not exists:
            logger.info(f"Creating Qdrant collection: {collection_name} (dim: {vector_size})")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

    def index_chunks(
        self,
        document_id: int,
        strategy: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        recreate: bool = True
    ) -> List[str]:
        if self.client is None or not chunks or not embeddings:
            return []

        collection_name = self.get_collection_name(document_id, strategy)
        vector_size = len(embeddings[0])

        if recreate:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass

        self.ensure_collection(collection_name, vector_size)

        points = []
        point_ids = []
        for chunk, emb in zip(chunks, embeddings):
            # Deterministic UUID for chunk point
            pt_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_{strategy}_{chunk.chunk_index}"))
            point_ids.append(pt_id)

            payload = {
                "document_id": document_id,
                "strategy": strategy,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "page_number": chunk.page_number,
                "bbox": list(chunk.bbox),
                "char_length": chunk.char_length,
                "token_count": chunk.token_count,
            }

            points.append(
                PointStruct(
                    id=pt_id,
                    vector=emb,
                    payload=payload
                )
            )

        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=collection_name,
                points=points[i:i + batch_size]
            )

        logger.info(f"Indexed {len(points)} chunks into {collection_name}")
        return point_ids

    def search(
        self,
        document_id: int,
        strategy: str,
        query_vector: List[float],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if self.client is None:
            return []

        collection_name = self.get_collection_name(document_id, strategy)

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            if not any(c.name == collection_name for c in collections):
                logger.warning(f"Collection {collection_name} does not exist.")
                return []

            # Search in Qdrant (using query_points for qdrant-client >= 1.10 or search)
            if hasattr(self.client, "query_points"):
                results = self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=top_k,
                    with_payload=True
                ).points
            else:
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True
                )

            retrieved = []
            for rank, r in enumerate(results, start=1):
                payload = r.payload or {}
                retrieved.append({
                    "rank": rank,
                    "point_id": str(r.id),
                    "score": float(r.score) if hasattr(r, "score") else 0.0,
                    "chunk_index": payload.get("chunk_index", -1),
                    "page_number": payload.get("page_number", 1),
                    "bbox": tuple(payload.get("bbox", [0.0, 0.0, 0.0, 0.0])),
                    "text": payload.get("text", ""),
                    "char_length": payload.get("char_length", 0),
                    "token_count": payload.get("token_count", 0),
                })
            return retrieved

        except Exception as e:
            logger.error(f"Search failed for {collection_name}: {e}")
            return []

    def delete_collection(self, document_id: int, strategy: str) -> None:
        if self.client is None:
            return
        collection_name = self.get_collection_name(document_id, strategy)
        try:
            self.client.delete_collection(collection_name)
        except Exception as e:
            logger.warning(f"Could not delete collection {collection_name}: {e}")
