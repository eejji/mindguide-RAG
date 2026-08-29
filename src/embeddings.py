from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"

def load_embedding_model() -> SentenceTransformer:
    """
    E5 임베딩 모델을 CPU에 로드

    Text --> 384차원의 벡터로 변환
    """

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    return model


def embed_passages( # 기존 PDF 내용들을 txt --> vector
    model: SentenceTransformer,
    texts: list[str]) -> list[list[float]]:

    """
    문서 청크 목록을 임베딩 벡터 목록으로 변환한다.
    Text --> e5 model -->  Vector
    """

    if not texts: # text가 없는 경우
        return []

    passages: list[str] = []

    for text in texts:
        passage = f"passage: {text}"
        passages.append(passage)

    vectors = model.encode(
        passages,
        normalize_embeddings=True, # 정규화 --> 차후 코사인 유사도 계산 위해
        show_progress_bar=True)

    return vectors.tolist()

def embed_query( # 사용자 질문을 받아서 query 가져오는 것
        model:SentenceTransformer,
        question:str
) -> list[float]:

    """
    사용자 질문 하나를 384차원 벡터로 변환
    """

    query = f"query: {question}"

    vector = model.encode(
        query,
        normalize_embeddings=True
    )

    return vector.tolist()
