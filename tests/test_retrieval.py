from qdrant_client import QdrantClient
from app.retrieval.qdrant_retriever import QdrantRetriever
from app.chunking.base import Chunk


def test_qdrant_retriever_indexing_and_search():
    retriever = QdrantRetriever(url=":memory:")
    # Force in-memory client
    retriever.client = QdrantClient(":memory:")

    chunks = [
        Chunk(
            chunk_index=0,
            text="First chunk about machine learning and neural networks.",
            page_number=1,
            bbox=(50.0, 100.0, 500.0, 150.0),
            char_length=56,
            token_count=8,
            strategy="fixed"
        ),
        Chunk(
            chunk_index=1,
            text="Second chunk detailing recursive character text splitters.",
            page_number=1,
            bbox=(50.0, 160.0, 500.0, 210.0),
            char_length=58,
            token_count=7,
            strategy="fixed"
        )
    ]

    # Deterministic 4D embeddings for testing
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ]

    point_ids = retriever.index_chunks(
        document_id=99,
        strategy="fixed",
        chunks=chunks,
        embeddings=embeddings,
        recreate=True
    )

    assert len(point_ids) == 2

    # Query matching vector 0
    query_vec_0 = [1.0, 0.0, 0.0, 0.0]
    results = retriever.search(document_id=99, strategy="fixed", query_vector=query_vec_0, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_index"] == 0
    assert results[0]["score"] > 0.99
    assert results[0]["page_number"] == 1
    assert results[0]["bbox"] == (50.0, 100.0, 500.0, 150.0)

    # Query matching vector 1
    query_vec_1 = [0.0, 1.0, 0.0, 0.0]
    results_1 = retriever.search(document_id=99, strategy="fixed", query_vector=query_vec_1, top_k=1)
    assert len(results_1) == 1
    assert results_1[0]["chunk_index"] == 1
