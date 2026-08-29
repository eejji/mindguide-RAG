from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

from src.models import Chunk, Document

def open_local_vector_store(
        storage_path: Path,
) -> QdrantClient:

    """
    지정한 폴더를 사용하는 로컬 Qdrant
    """

    storage_path.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path = str(storage_path))

    return client


def recreate_collection(
        client: QdrantClient,
        collection_name: str,
        vector_size: int
) -> None:
    """
    기존 컬렉션을 지우고 새로운 컬렉션을 만듬
    """

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )


def store_chunks(
        client: QdrantClient,
        collection_name: str,
        document: Document,
        chunks: list[Chunk],
        chunk_vectors: list[list[float]]
) -> None:

    """
    청크 벡터와 출처 정보를 Qdrant에 저장한다.
    """
    if len(chunks) != len(chunk_vectors):
        raise ValueError(
            "청크 수와 청크 벡터 수가 같아야 합니다."
        )

    points: list[PointStruct] = []

    for index in range(len(chunks)):
        chunk = chunks[index]
        vector = chunk_vectors[index]

        if chunk.document_id != document.document_id:
            raise ValueError(
                "청크와 문서의 document_id가 일치해야 합니다."
            )

        payload = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "title": document.title,
            "source_url": document.source_url,
            "language": document.language,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "text": chunk.text,
        }

        point = PointStruct( # 벡터 데이터 한 건을 표현하는 객체
            id = index,
            vector = vector,
            payload = payload
        )

        points.append(point)

    if points:
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )

def search_vector_store(
        client: QdrantClient,
        collection_name: str,
        query_vector: list[float],
        top_k: int =3
) -> list[ScoredPoint]:

    """
    질문 벡터와 가장 가까운 Qdrant Point를 반환
    """

    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return response.points
