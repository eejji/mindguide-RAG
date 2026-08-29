import pytest

from src.models import Chunk
from src.search import cosine_similarity, search_chunks


def test_cosine_similarity_for_same_and_orthogonal_vectors():
    same_score = cosine_similarity(
        vector_a=[1.0, 0.0],
        vector_b=[1.0, 0.0],
    )
    orthogonal_score = cosine_similarity(
        vector_a=[1.0, 0.0],
        vector_b=[0.0, 1.0],
    )

    assert same_score == pytest.approx(1.0)
    assert orthogonal_score == pytest.approx(0.0)


def test_search_chunks_returns_highest_scores_first():
    chunks = [
        Chunk("c001", "test", 1, 1, "가장 관련 있는 문장"),
        Chunk("c002", "test", 1, 1, "관련 없는 문장"),
        Chunk("c003", "test", 1, 1, "조금 관련 있는 문장"),
    ]
    chunk_vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.2],
    ]

    results = search_chunks(
        chunks=chunks,
        chunk_vectors=chunk_vectors,
        query_vector=[1.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c001"
    assert results[1].chunk.chunk_id == "c003"
    assert results[0].score > results[1].score
