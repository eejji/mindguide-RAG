from dataclasses import dataclass

@dataclass(frozen=True) # 생성자가 같은 반복코드를 Python이 자동 생성
class Document: # 문서 자체의 출처 정보
    document_id: str
    title: str
    source_url: str
    language: str

@dataclass(frozen=True)
class Page: # 한 페이지의 텍스트
    document_id: str
    page_number: int
    text: str

@dataclass(frozen=True)
class Chunk: # 검색에 사용할 작은 텍스트 조각
    chunk_id: str
    document_id: str
    page_start: int
    page_end: int
    text:str