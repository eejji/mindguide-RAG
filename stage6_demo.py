from pathlib import Path

from src.chunking import chunk_page
from src.embeddings import (
    embed_passages,
    embed_query,
    load_embedding_model,
)
from src.models import Document
from src.pdf_parser import extract_page
from src.vector_store import (
    open_local_vector_store,
    recreate_collection,
    search_vector_store,
    store_chunks,
)


collection_name = "depression_guideline_chunks"

qdrant_path = Path(
    "data/qdrant_local"
)

pdf_path = Path(
    "data/documents/우울증임상진료지침_대한의학회.pdf"
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


# 2. 실제 PDF 페이지 추출
page = extract_page(
    pdf_path=pdf_path,
    document_id=document.document_id,
    page_number=18,
)


# 3. 청킹
chunks = chunk_page(
    page=page,
    chunk_size=300,
    overlap=50,
)


# 4. 청크 본문 문자열 준비
chunk_texts: list[str] = []

for chunk in chunks:
    chunk_texts.append(chunk.text)


# 5. 임베딩
model = load_embedding_model()

chunk_vectors = embed_passages(
    model=model,
    texts=chunk_texts,
)

# 6. 로컬 Qdrant
client = open_local_vector_store(storage_path=qdrant_path)

try:
    # 7. 컬렉션 생 (collection)
    vector_size = len(chunk_vectors[0])

    recreate_collection(
        client=client,
        collection_name=collection_name,
        vector_size=vector_size
    )

    # 8. 청크, 벡터, 출처 저장
    store_chunks(
        client=client,
        collection_name=collection_name,
        document=document,
        chunks=chunks,
        chunk_vectors=chunk_vectors
    )

    collection_info = client.get_collection(
        collection_name=collection_name
    )

    print()
    print("Qdrant 저장 위치:", qdrant_path)
    print("저장된 Point 수:", collection_info.points_count)

    # 9. 질문 임베딩
    question = (
        "우울증 선별에 사용할 수 있는 "
        "도구는 무엇인가?"
    )

    query_vector = embed_query(
        model=model,
        question=question,
    )
    # 10. Qdrant Top-3 검색
    hits = search_vector_store(
        client=client,
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=3,
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
        print("문서:", payload["title"])
        print("페이지:", payload["page_start"])
        print("출처:", payload["source_url"])
        print("내용:", preview)
        print("-" * 70)

        rank += 1

finally:
    # Windows에서 Qdrant 폴더 잠금이 남지 않도록 닫는다.
    client.close()
