from pathlib import Path

from src.chunking import chunk_page
from src.models import Document
from src.pdf_parser import extract_page


document = Document(
    document_id = "depression_cpg_2022",
    title = "우울증임상진료지침_대한의학회",
    source_url = ("https://guideline.or.kr/chronic/"
                "view.php?number=98"),
    language="ko"
    )

pdf_path = Path("data/documents/우울증임상진료지침_대한의학회.pdf")

# pymupdf 라이브러리 통해 PDF 한 페이지의 텍스트 추출
page = extract_page(
    pdf_path = pdf_path,
    document_id = document.document_id,
    page_number = 18
)

# chunking.py의 chunk_page 함수를 활용해서 PDF의 한 페이지를 일정한 글자 수로 나눔
chunks = chunk_page(
    page = page,
    chunk_size = 300,
    overlap = 50
)

print("문서:", document.title)
print("추출 페이지:", page.page_number)
print("추출 글자 수:", len(page.text))
print("생성된 청크 수:", len(chunks))
print()

for chunk in chunks:
    preview = " ".join(chunk.text.split())

    print("청크 ID:", chunk.chunk_id)
    print("페이지:", chunk.page_start)
    print("내용:", preview)
    print("-" * 70)


phq_chunks = [
    chunk
    for chunk in chunks
    if "PHQ-9" in chunk.text
]

print("PHQ-9이 포함된 청크 수:", len(phq_chunks))


"""

청크 간 겹침 확인하는 코드

print()
print("청크 간 겹침 확인")
print("=" * 70)

chunk_size = 300
overlap = 50
step = chunk_size - overlap

for index in range(len(chunks) - 1):
    current_start = index * step
    next_start = (index + 1) * step

    current_raw = page.text[
        current_start:current_start + chunk_size
    ]

    next_raw = page.text[
        next_start:next_start + chunk_size
    ]

    current_overlap = current_raw[-overlap:]
    next_overlap = next_raw[:overlap]


    print(
        f"c{index + 1:03d} → "
        f"c{index + 2:03d}"
    )
    print("앞 청크 마지막 50자:", repr(current_overlap))
    print("뒤 청크 처음 50자:", repr(next_overlap))
    print("동일 여부:", current_overlap == next_overlap)
    print("-" * 70)"""