from src.models import Chunk, Page
"""
Page : 청킹할 원본 페이지 데이터
Chunk: 청킹이 끝난 검색용 텍스트 조각
"""

def chunk_page(page: Page, chunk_size: int, overlap: int = 0) -> list[Chunk]:
    """
    PDF의 한 페이지를 일정한 글자 수로 나눕니다.
    입력값:
        page:
            청킹할 페이지 객체입니다.
            document_id, page_number, text를 가지고 있습니다.
        chunk_size:
            하나의 청크에 포함할 최대 글자 수 입니다.
            예를 들어 300이면 텍스트를 최대 300자씩 나눕니다.

    반환값:
        list[Chunk]:
            분할된 Chunk 객체들을 리스트로 변환

    예외:
        chunk_size가 0 이하이면 ValueError를 발생시킵니다.
    """

    # chunk_size가 0이면 범위의 간격으로 사용할 수 없습니다.
    # 음수도 정상 청크 크기가 아니므로 실행 중단

    if chunk_size <= 0:
        raise ValueError("chunk_size는 1 이상이어야 합니다.")

    if overlap < 0:
        raise ValueError("overlap은 0 이상이어야 합니다.")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap은 chunk_size보다 작아야 합니다."
        )

    # 완성된 Chunk 객체들을 저장할 빈 리스트
    chunks: list[Chunk] = []

    # len(page.text): 페이지 전체 텍스트의 글자 수를 구함.
    # range(0, len(page.text), chunck_size)

    step = chunk_size - overlap # 다음 청크로 이동하 실제 글자 수
    # chunk_size=30, overlap=10인 경우 시작 위치는 0, 20, 40, 60으로 이동하면서 계속 중첩

    for chunk_number, start in enumerate(range(0, len(page.text), step), start=1):

        end = start + chunk_size # 각 청크에 1번부터 순서 번호 부여 / 현재 청크 끝날 위치 계산
        chunk_text = page.text[start:end].strip()
        # strip()을 통해 청크 앞 뒤의 불필요한 공백과 줄바꿈 제거

        if not chunk_text:
            continue

        # 하나의 Chunk 객체 생성 (잘라낸 텍스트, 출처 정보)
        chunk = Chunk(
            chunk_id=(f"{page.document_id}"
                    f"_p{page.page_number}"
                    f"_c{chunk_number:03d}"),
            document_id = page.document_id,
            page_start = page.page_number,
            page_end = page.page_number,
            text=chunk_text
        )

        chunks.append(chunk)

        # 현재 청크가 원문의 마지막 글자까지 포함되면 중복 청크 만들지 않고 종료
        if end >= len(page.text):
            break

    return chunks


def chunk_pages(
        pages: list[Page],
        chunk_size : int,
        overlap: int = 0
) -> list[Chunk]:
    """
    여러 Page 객체를 순서대로 청킹
    """

    all_chunks: list[Chunk] = []

    for page in pages:
        page_chunks = chunk_page(
            page = page,
            chunk_size = chunk_size,
            overlap = overlap
        )

        for chunk in page_chunks:
            all_chunks.append(chunk)

    return all_chunks