from pathlib import Path

import pytest

from src.models import Page
from src.pdf_parser import extract_page, extract_pages

# 현재 테스트 파일 위치를 기준으로 프로젝트 최상위 폴더 찾기
project_root = Path(__file__).resolve().parents[1]

# 실제 실습용 PDF의 절대 경로 생성
# Path 라이브러리 모듈로해서 합치는 것
guideline_pdf_path = (
    project_root
    / "data"
    / "documents"
    / "우울증임상진료지침_대한의학회.pdf"
)

@pytest.mark.skipif(
    not guideline_pdf_path.is_file(),
    reason="실습용 우울증 임상진료지침 PDF가 없습니다."
)
def test_extract_real_guideline_page():
    # 실행되면 실제 진료지침의 18페이지를 추출
    """
    MT) 18페이지를 추출해야하나?
    원래 같으면 다 검사해야되는 것 아닌가?
    """
    page = extract_page(
        pdf_path = guideline_pdf_path,
        document_id="depression_cpg_2022",
        page_number=18
    )

    # 실제 추출 결과를 눈으로 확인한다.
    print("\n문서 ID:", page.document_id)
    print("페이지:", page.page_number)
    print("추출 글자 수:", len(page.text))
    print("-" * 70)
    print(page.text)
    print("-" * 70)

    # 검증
    assert isinstance(page, Page)
    assert page.document_id == "depression_cpg_2022"
    assert page.page_number == 18
    assert len(page.text) > 100
    assert "우울증" in page.text
    assert "PHQ-9" in page.text


@pytest.mark.skipif(
    not guideline_pdf_path.is_file(),
    reason="실습용 우울증 임상진료지침 PDF가 없습니다."
)
def test_extract_real_guideline_page_range():
    pages = extract_pages(
        pdf_path=guideline_pdf_path,
        document_id="depression_cpg_2022",
        start_page=18,
        end_page=19,
    )

    assert len(pages) == 2
    assert pages[0].page_number == 18
    assert pages[1].page_number == 19
    assert "PHQ-9" in pages[0].text
