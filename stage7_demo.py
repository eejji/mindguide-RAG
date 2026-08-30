from pathlib import Path

from src.chunking import chunk_pages
from src.embeddings import (
    embed_passages,
    embed_query,
    load_embedding_model,
)
from src.models import Document
from src.pdf_parser import extract_pages
from src.vector_store import (
    open_local_vector_store,
    recreate_collection,
    search_vector_store,
    store_chunks,
)


collection_name = "depression_guideline_chunks"

pdf_path = Path(
    "data/documents/우울증임상진료지침_대한의학회.pdf"
)

qdrant_path = Path(
    "data/qdrant_local"
)


# 1. 문서 출처 정보
document = Document(
    document_id="depression_cpg_2022",
    title="우울증임상진료지침_대한의학회",
    source_url=(
        "https://guideline.or.kr/chronic/"
        "view.php?number=98"
    ),
    language="ko",
)


# 2. PDF 전체 페이지 추출
pages = extract_pages(
    pdf_path=pdf_path,
    document_id=document.document_id,
)


# 3. 모든 페이지 청킹
chunks = chunk_pages(
    pages=pages,
    chunk_size=300,
    overlap=50,
)


# 4. 청크 본문만 별도 목록으로 준비
chunk_texts: list[str] = []

for chunk in chunks:
    chunk_texts.append(chunk.text)


print("PDF 전체 페이지 수: 152")
print("텍스트가 추출된 페이지 수:", len(pages))
print("생성된 전체 청크 수:", len(chunks))


# 5. 모든 청크 임베딩
model = load_embedding_model()

chunk_vectors = embed_passages(
    model=model,
    texts=chunk_texts,
)


# 6. Qdrant 열기
client = open_local_vector_store(
    storage_path=qdrant_path
)

try:
    vector_size = len(chunk_vectors[0])

    # 기존 5개짜리 컬렉션을 전체 문서 인덱스로 교체한다.
    recreate_collection(
        client=client,
        collection_name=collection_name,
        vector_size=vector_size,
    )

    store_chunks(
        client=client,
        collection_name=collection_name,
        document=document,
        chunks=chunks,
        chunk_vectors=chunk_vectors,
    )

    collection_info = client.get_collection(
        collection_name=collection_name
    )

    print("Qdrant에 저장된 Point 수:", collection_info.points_count)


   # 7. 전체 문서 대상 검색
    question = (
        "우울증 선별에 사용할 수 있는 "
        "도구는 무엇인가?"
    )

    query_vector = embed_query(
        model=model,
        question=question,
    )

    hits = search_vector_store(
        client=client,
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=5,
    )

    print()
    print("질문:", question)
    print("=" * 70)

    rank = 1

    for hit in hits:
        payload = hit.payload

        if payload is None:
            continue

        preview = " ".join(
            str(payload["text"]).split()
        )

        print("순위:", rank)
        print("점수:", round(hit.score, 4))
        print("청크 ID:", payload["chunk_id"])
        print("PDF 페이지:", payload["page_start"])
        print("내용:", preview)
        print("-" * 70)

        rank += 1

finally:
    client.close()