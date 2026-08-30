from pathlib import Path
import pymupdf
from src.models import Page

def extract_page(
        pdf_path: Path,
        document_id: str,
        page_number: int,
) -> Page:

    """
    PDF에서 지정한 한 페이지의 텍스트를 추출
    입력:
        pdf_path --> 읽을 PDF 파일의 경로
        documnet_id: PDF를 구분할 내부 문서 ID
        page_number: 사람이 사용하는 1부터 시작하는 페이지 번호

    반환:
        추출된 텍스트가 들어 있는 Page 객체
    """

    if page_number < 1:
        raise ValueError("page_number는 1 이상이어야 합니다.")

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF를 찾을 수 없습니다. {pdf_path}")

    # with 블록이 끝나면 PDF 파일이 자동으로 닫힙니다.
    with pymupdf.open(pdf_path) as pdf:
        if page_number > pdf.page_count:
            raise ValueError(
                f"페이지 범위를 벗어났습니다."
                f"전체 페이지: {pdf.page_count}"
            )

        # 첫 페이지 0번
        pdf_page = pdf.load_page(page_number - 1)

        text = pdf_page.get_text("text").strip()

    if not text:
        raise ValueError(
            f"{page_number} 페이지에서 텍스트를 추출하지 못했습니다."
        )

    return Page(
        document_id = document_id,
        page_number = page_number,
        text = text
    )

def extract_pages(
        pdf_path: Path,
        document_id: str,
        start_page: int = 1,
        end_page: int | None = None
) -> list[Page]:

    """
    PDF의 지정 범위에서 텍스트가 있는 페이지를 모두 추출
    """

    if start_page < 1:
        raise ValueError("start_page는 1 이상이어야 한다.")

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF를 찾을 수 없습니다. {pdf_path}"
        )

    pages: list[Page] = []

    with pymupdf.open(pdf_path) as pdf:
        last_page = end_page

        if last_page is None:
            last_page = pdf.page_count

        if start_page > pdf.page_count:
            raise ValueError("start_page가 PDF 전체 페이지 수를 초과했습니다.")

        if last_page > pdf.page_count:
            raise ValueError(
                "end_page가 PDF 전체 페이지 수를 초과했습니다."
            )

        if start_page > last_page:
            raise ValueError(
                "start_page는 end_page보다 작거나 같아야 합니다."
            )

        for page_number in range(start_page, last_page + 1):
            pdf_page = pdf.load_page(page_number - 1)

            text = pdf_page.get_text("text").strip()

            if not text:
                continue

            page = Page(
                document_id = document_id,
                page_number = page_number,
                text = text
            )

            pages.append(page)

    if not pages:
        raise ValueError(
            "지정한 범위에서 텍스트를 추출하지 못했습니다."
        )

    return pages