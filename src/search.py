from src.models import Chunk, SearchResult

def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float]
) -> float:

    """
    두 벡터의 코사인 유사도를 계산
    """

    if len(vector_a) != len(vector_b):
        raise ValueError("두 벡터의 차원이 같아야합니다.")
    
    if not vector_a:
        raise ValueError("빈 벡터는 비교할 수 없습니다.")

    dot_product = 0.0
    vector_a_square_sum = 0.0
    vector_b_square_sum = 0.0

    for index in range(len(vector_a)):
        value_a = vector_a[index]
        value_b = vector_b[index]

        # 두 벡터 내적 구하기
        dot_product += value_a * value_b

        # 각 벡터 길이 계산에 사용할 제곱 합
        vector_a_square_sum += value_a ** 2
        vector_b_square_sum += value_b ** 2

    vector_a_length = vector_a_square_sum ** 0.5
    vector_b_length = vector_b_square_sum ** 0.5

    denominator = vector_a_length * vector_b_length

    if denominator == 0:
        raise ValueError("길이가 0인 벡터는 비교할 수 없습니다.")

    similarity = dot_product / denominator

    return similarity


def get_result_score(result: SearchResult) -> float:
    """
    검색 결과를 점수순으로 정렬할 때 사용할 값을 반환한다.
    """
    return result.score

def search_chunks(
    chunks: list[Chunk],
    chunk_vectors: list[list[float]],
    query_vector: list[float],
    top_k: int = 3,
) -> list[SearchResult]:
    """
    질문과 가장 유사한 청크를 점수순으로 반환한다.
    """    

    if len(chunks) != len(chunk_vectors):
        raise ValueError("청크 수와 청크 벡터 수가 같아야 합니다.")

    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    results: list[SearchResult] = []

    for index in range(len(chunks)):
        chunk = chunks[index]
        chunk_vector = chunk_vectors[index]

        score = cosine_similarity(
            vector_a = query_vector,
            vector_b = chunk_vector
        )

        result = SearchResult(
            chunk=chunk,
            score=score
        )

        results.append(result)

    results.sort(key = get_result_score, reverse=True)

    top_results = results[:top_k]

    return top_results
