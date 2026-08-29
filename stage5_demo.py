from pathlib import Path

from src.chunking import chunk_page
from src.embeddings import (
    embed_passages,
    embed_query,
    load_embedding_model,
)
from src.pdf_parser import extract_page
from src.search import search_chunks

# 1. 실제 PDF에서 18페이지 추출
pdf_path = Path(
    "data/documents/우울증임상진료지침_대한의학회.pdf"
)

page = extract_page(
    pdf_path=pdf_path,
    document_id="depression_cpg_2022",
    page_number=18,
)


# 2. 페이지 청킹
chunks = chunk_page(
    page=page,
    chunk_size=300,
    overlap=50,
)

# 3. Chunk 객체에서 본문 문자열만 꺼내기
chunk_texts: list[str] = []

for chunk in chunks:
    chunk_texts.append(chunk.text)


# 4. 문서와 질문 임베딩
model = load_embedding_model()

chunk_vectors = embed_passages(
    model=model,
    texts=chunk_texts,
)

question = "우울증 선별에 사용할 수 있는 도구는 무엇인가?"

query_vector = embed_query(
    model=model,
    question=question,
)

# 5. 가장 관련 있는 청크 3개 검색
results = search_chunks(
    chunks=chunks,
    chunk_vectors=chunk_vectors,
    query_vector=query_vector,
    top_k=3,
)

# 6. 검색 결과 출력
print()
print("질문:", question)
print("=" * 70)

rank = 1

for result in results:
    chunk = result.chunk
    preview = " ".join(chunk.text.split())

    print("순위:", rank)
    print("점수:", round(result.score, 4))
    print("청크 ID:", chunk.chunk_id)
    print("페이지:", chunk.page_start)
    print("내용:", preview)
    print("-" * 70)

    rank += 1