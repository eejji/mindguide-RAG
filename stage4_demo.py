from pathlib import Path

from src.chunking import chunk_page
from src.embeddings import (
    embed_passages,
    embed_query,
    load_embedding_model
)

from src.pdf_parser import extract_page

# 1. 실제 PDF 페이지 추출
pdf_path = Path("data/documents/우울증임상진료지침_대한의학회.pdf")

page = extract_page(
    pdf_path=pdf_path,
    document_id="depression_cpg_2022",
    page_number=18
)

# 2. 페이지를 검색용 청크로 분할

chunks = chunk_page(
    page=page,
    chunk_size=300,
    overlap=50
)

# 3. 임베딩 모델 로드
print("Embedding model E5 loading...")
model = load_embedding_model()

# 4. Chunk 객체에서 실제 본문 문자열만 꺼낸다.
chunk_texts: list[str] = []

for chunk in chunks:
    chunk_texts.append(chunk.text)

chunk_vectors = embed_passages(model=model, texts=chunk_texts)

# 5. 사용자 질문도 벡터로 변환
question = "우울증 선별에 사용할 수 있는 도구는 무엇인가?"

query_vector = embed_query(model=model, question=question)

# 6. 결과 확인
first_vector = chunk_vectors[0]

sum_of_squares = 0.0
for value in first_vector:
    sum_of_squares += value ** 2

vector_dimension = len(first_vector)
vector_norm = sum_of_squares ** 0.5

print()
print("생성된 청크 수:", len(chunks))
print("생성된 청크 벡터 수:", len(chunk_vectors))
print("청크 벡터 차원:", len(first_vector))
print("질문 벡터 차원:", len(query_vector))
print("첫 벡터 앞 10개 값:", first_vector[:10])
print("정규화된 벡터 길이:", vector_norm)