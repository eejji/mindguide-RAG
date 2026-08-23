from src.models import Chunk, Document, Page

document = Document(
    document_id="depression_cpg_2022",
    title="우울증 진료지침",
    source_url="https://example.org/guideline",
    language="ko"
)

page = Page(
    document_id=document.document_id,
    page_number=12,
    text="우울증 선별에는 PHQ-9 등의 도구를 사용할 수 있다."
)


chunk = Chunk(
    chunk_id="depression_cpg_2022_p12_c001",
    document_id=page.document_id,
    page_start=page.page_number,
    page_end=page.page_number,
    text=page.text
)

# 이 조건이 반드시 참이어야 한다는 Python 문법
assert page.document_id == document.document_id
assert chunk.page_start <= chunk.page_end

print(f"문서 : {document.title}")
print(f"페이지 : {page.page_number}")
print(f"청크 : {chunk.text}")
print(f"출처 : {document.source_url}")