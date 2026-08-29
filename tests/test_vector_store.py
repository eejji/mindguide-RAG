from qdrant_client import QdrantClient

from src.models import Chunk, Document
from src.vector_store import (
    recreate_collection,
    search_vector_store,
    store_chunks,
)


def test_qdrant_stores_payload_and_returns_nearest_chunk():
    client = QdrantClient(":memory:")
    collection_name = "test_chunks"

    document = Document(
        document_id="test_document",
        title="테스트 문서",
        source_url="https://example.org/test",
        language="ko",
    )

    chunks = [
        Chunk("c001", "test_document", 1, 1, "관련 있는 문장"),
        Chunk("c002", "test_document", 2, 2, "관련 없는 문장"),
    ]
    chunk_vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    try:
        recreate_collection(
            client=client,
            collection_name=collection_name,
            vector_size=2,
        )
        store_chunks(
            client=client,
            collection_name=collection_name,
            document=document,
            chunks=chunks,
            chunk_vectors=chunk_vectors,
        )

        hits = search_vector_store(
            client=client,
            collection_name=collection_name,
            query_vector=[1.0, 0.0],
            top_k=1,
        )

        assert len(hits) == 1
        assert hits[0].payload is not None
        assert hits[0].payload["chunk_id"] == "c001"
        assert hits[0].payload["title"] == "테스트 문서"
        assert hits[0].score == 1.0
    finally:
        client.close()
