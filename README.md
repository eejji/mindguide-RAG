# MindGuideOps Rebuild

공식 우울증 임상진료지침을 활용해 의료 RAG의 데이터 처리 과정을 처음부터 다시 구현하는 학습 프로젝트입니다.

기존 포트폴리오 완성본을 복사하지 않고, Codex의 단계별 설명을 참고해 코드를 직접 작성하고 실행 결과를 확인하고 있습니다. 이 저장소에는 현재까지 직접 실습한 범위만 기록합니다.

## 현재 구현 범위

현재 단계는 **실제 PDF 한 페이지 추출 및 페이지 단위 문자 청킹**입니다.

- `Document`, `Page`, `Chunk` 데이터 모델
- PyMuPDF를 이용한 지정 페이지 텍스트 추출
- `chunk_size`와 `overlap`을 적용한 문자 단위 청킹
- 문서 ID, 페이지 번호, 청크 순번을 포함한 추적 가능한 청크 ID
- 실제 지침 18페이지를 이용한 실행 예제
- 청킹·실제 PDF 추출·벡터 검색·Qdrant 자동 테스트 11건
- `intfloat/multilingual-e5-small`을 이용한 384차원 문서·질문 임베딩
- 코사인 유사도 기반 인메모리 Top-K 검색
- Qdrant local에 벡터와 출처 payload를 저장하고 Top-K 검색

아직 전체 PDF 적재, 검색 평가, 의료 안전 Agent, FastAPI는 구현하지 않았습니다.

## 현재 데이터 흐름

```text
PDF 파일
  -> 18페이지 텍스트 추출
  -> Page 객체
  -> 300자 단위 분할 (50자 overlap)
  -> 페이지와 문서 ID가 포함된 Chunk 객체 5개
  -> E5-small 384차원 Embedding
  -> 질문과 코사인 유사도 비교
  -> 관련 Chunk Top-3
  -> Qdrant에 벡터 + 본문 + 페이지 + 출처 저장
```

## 프로젝트 구조

```text
mindguide-ops-rebuild/
|-- data/
|   `-- documents/
|       `-- README.md
|-- src/
|   |-- __init__.py
|   |-- models.py
|   |-- pdf_parser.py
|   |-- chunking.py
|   |-- embeddings.py
|   |-- search.py
|   `-- vector_store.py
|-- tests/
|   |-- test_chunking.py
|   |-- test_pdf_parser.py
|   |-- test_search.py
|   `-- test_vector_store.py
|-- stage1_demo.py
|-- stage2_demo.py
|-- stage4_demo.py
|-- stage5_demo.py
|-- stage6_demo.py
|-- LEARNING_LOG.md
`-- requirements.txt
```

진료지침 PDF는 파일 크기와 재배포 조건을 고려해 Git 저장소에 포함하지 않습니다.

## 실행 환경

- Python 3.12
- PyMuPDF 1.28.2
- Sentence Transformers 3.0 이상
- Qdrant Client 1.9 이상
- pytest 9.1.1

Windows PowerShell 기준:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
$env:PYTHONUTF8 = '1'
```

`PYTHONUTF8` 설정은 Windows 터미널에서 PDF 본문의 특수 기호가 깨지는 문제를 방지합니다.

## 실습용 PDF 준비

대한의학회·질병관리청의 `일차 의료용 우울증 임상진료지침`을 사용했습니다.

- 공식 출처: <https://guideline.or.kr/chronic/view.php?number=98>
- 저장 위치: `data/documents/우울증임상진료지침_대한의학회.pdf`

자세한 준비 방법은 [`data/documents/README.md`](data/documents/README.md)를 참고합니다.

## 실행 방법

데이터 모델 확인:

```powershell
python .\stage1_demo.py
```

실제 PDF 18페이지 추출 및 청킹:

```powershell
python .\stage2_demo.py
```

실제 청크와 질문의 E5 임베딩 확인:

```powershell
python .\stage4_demo.py
```

코사인 유사도 Top-3 검색:

```powershell
python .\stage5_demo.py
```

Qdrant local 벡터 저장 및 Top-3 검색:

```powershell
python .\stage6_demo.py
```

전체 자동 테스트:

```powershell
python -m pytest -v
```

실습 PDF가 로컬에 있으면 실제 18페이지 추출 테스트까지 실행하며, PDF가 없는 환경에서는 해당 테스트만 건너뜁니다. 실제 PDF 테스트는 대표 페이지의 한글·표·특수문자 추출을 확인하는 통합 테스트입니다. 모든 페이지의 데이터 품질 검사는 이후 별도의 문서 적재·품질 점검 단계에서 수행합니다.

현재 확인한 결과:

```text
추출 페이지: 18
추출 글자 수: 1153
생성된 청크 수: 5
chunk_size: 300
overlap: 50
```

텍스트 길이와 청크 수는 PDF 파일과 PyMuPDF 버전에 따라 달라질 수 있습니다.

## 이번 단계에서 이해한 내용

- `document_id`는 검색 결과를 원문 출처와 다시 연결하는 식별자입니다.
- 페이지 번호는 검색 근거의 위치를 인용하기 위해 유지합니다.
- 긴 페이지를 작은 청크로 나누면 질문과 직접 관련된 범위를 검색하기 쉬워집니다.
- overlap은 청크 경계에서 문맥이 완전히 끊기는 문제를 줄이지만 중복 저장을 증가시킵니다.
- 단순 문자 청킹은 단어나 문장을 중간에서 자를 수 있습니다.
- PDF의 표와 머리말·꼬리말은 텍스트 추출 과정에서 본문과 섞일 수 있습니다.
- 실패 테스트로 마지막 중복 청크 문제를 재현하고 수정한 뒤 회귀 테스트로 보호했습니다.
- E5에서는 문서에 `passage:`, 질문에 `query:` 접두어를 사용해야 합니다.
- 임베딩 유사도는 정답 확률이 아니라 검색 관련도입니다.

## 현재 한계

- 한 문서의 한 페이지만 실습했습니다.
- 문자 수 기반 청킹이라 문장 경계를 보존하지 않습니다.
- PDF 머리말, 페이지 번호, 표 구조를 별도로 정제하지 않았습니다.
- 실제 PDF 자동 테스트는 대표 18페이지 한 건이며 전체 페이지 품질 검사는 아직 없습니다.
- 검색 품질 평가는 아직 없습니다.
- 인메모리 기준 검색과 Qdrant 검색을 모두 구현했지만 검색 대상은 아직 한 페이지뿐입니다.
- Qdrant에는 아직 대표 18페이지의 청크 5개만 저장합니다.
- 이 단계에는 생성형 답변이나 의료 안전 Agent 동작이 없습니다.

## 다음 단계

- PDF 전처리 품질 점검
- 전체 PDF 페이지 적재와 Qdrant 인덱스 생성
- 검색 평가, 의료 안전 라우팅, API
