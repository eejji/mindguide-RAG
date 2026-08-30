import pytest

from src.models import Page
from src.chunking import chunk_page, chunk_pages

def test_short_page_creates_one_chunk():
    # 300자 보다 짧은 페이지 만들기
    page = Page(
        document_id="test_document",
        page_number=1,
        text="우울증 선별에는 PHQ-9을 사용할 수 있다."
    )

    chunks = chunk_page(
        page=page,
        chunk_size=300,
        overlap=50,
    )

    # 검증: 짧은 문장이므로 청크가 하나만 생성되어야 함.
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "test_document_p1_c001"
    assert chunks[0].document_id == "test_document"
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[0].text == page.text



def test_long_page_creates_overlapping_chunks():
    # 19글자의 위치를 쉽게 확인하기 위한 것
    page = Page( # Page class에 각 요소 및 항목 작성
        document_id="test_document",
        page_number=2,
        text="ABCDEFGHIJKLMNOPQRS"
    )

    # 청크 크기 10, 중복 크기 3
    # 다음 청크는 10 - 3 = 7글자씩 이동한다.
    chunks = chunk_page(
        page=page,
        chunk_size=10,
        overlap=3
    )

    # 실제 분할 결과 확인
    assert chunks[0].text == "ABCDEFGHIJ"
    assert chunks[1].text == "HIJKLMNOPQ"
    assert chunks[2].text == "OPQRS"

    # 앞 청크의 마지막 3글자, 다음 청크의 첫 3글자 겹침
    assert chunks[0].text[-3:] == chunks[1].text[:3]
    assert chunks[1].text[-3:] == chunks[2].text[:3]

    # 청크 번호가 순서대로 생성되는지 확인한다.
    assert chunks[0].chunk_id == "test_document_p2_c001"
    assert chunks[1].chunk_id == "test_document_p2_c002"
    assert chunks[2].chunk_id == "test_document_p2_c003"


def test_exact_chunk_size_does_not_create_duplicate_chunk():
    # 원문 길이와 chunk_size를 정확히 10자로 맞춘다.
    page = Page(
        document_id="test_document",
        page_number=3,
        text="ABCDEFGHIJ",
    )

    chunks = chunk_page(
        page=page,
        chunk_size=10,
        overlap=3,
    )

    # 확인용 출력
    print("\n원문 글자 수:", len(page.text))
    print("생성된 청크 수:", len(chunks))
    print("생성된 청크:", chunks[0])

    # 원문이 정확히 chunk_size와 같으면 청크는 하나여야 한다.
    assert len(page.text) == 10
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "test_document_p3_c001"
    assert chunks[0].text == "ABCDEFGHIJ"


def test_chunk_size_zero_raises_value_error():

    page = Page(
        document_id="test_document",
        page_number=4,
        text="테스트 문장입니다."
    )

    # ValueError가 발생해야 테스트 통과
    with pytest.raises(ValueError) as error:
        chunk_page(
            page=page,
            chunk_size=0,
            overlap=0
        )

    assert str(error.value) == "chunk_size는 1 이상이어야 합니다."


def test_negative_overlap_raises_value_error():
    # 준비: 테스트에 사용할 Page 객체를 만든다.
    page = Page(
        document_id="test_document",
        page_number=5,
        text="테스트 문장입니다.",
    )

    # 실행: 음수 overlap을 입력한다.
    # ValueError가 발생해야 테스트가 통과한다.
    with pytest.raises(ValueError) as error:
        chunk_page(
            page=page,
            chunk_size=10,
            overlap=-1,
        )

    # 검증: 우리가 예상한 오류 메시지가 맞는지 확인한다.
    assert str(error.value) == "overlap은 0 이상이어야 합니다."


def test_overlap_equal_to_chunk_size_raises_value_error():
    # 준비
    page = Page(
        document_id="test_document",
        page_number=6,
        text="테스트 문장입니다.",
    )

    # 실행: chunk_size와 같은 overlap을 입력한다.
    with pytest.raises(ValueError) as error:
        chunk_page(
            page=page,
            chunk_size=10,
            overlap=10,
        )

    # 검증
    assert str(error.value) == "overlap은 chunk_size보다 작아야 합니다."


def test_empty_page_creates_no_chunks():

    page = Page(
        document_id="test_document",
        page_number=7,
        text=""
    )

    chunks = chunk_page(
        page=page,
        chunk_size=10,
        overlap=3
    )

    assert chunks == []


def test_chunk_pages_preserves_each_page_number():
    pages = [
        Page("test_document", 1, "ABCDE"),
        Page("test_document", 2, "123456"),
    ]

    chunks = chunk_pages(
        pages=pages,
        chunk_size=5,
        overlap=0,
    )

    assert len(chunks) == 3
    assert chunks[0].chunk_id == "test_document_p1_c001"
    assert chunks[1].chunk_id == "test_document_p2_c001"
    assert chunks[2].chunk_id == "test_document_p2_c002"
    assert chunks[0].page_start == 1
    assert chunks[1].page_start == 2
