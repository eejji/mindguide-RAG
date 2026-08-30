import json
from pathlib import Path

from src.embeddings import MODEL_NAME, embed_query, load_embedding_model

from src.vector_store import open_local_vector_store, search_vector_store

eval_question_path = Path("data/eval_questions.json")
qdrant_path = Path("data/qdrant_local")
eval_results_dir = Path("data/eval_results")

collection_name = "depression_guideline_chunks"
top_k = 5

def load_eval_data(path: Path) -> dict:
    """
    평가 질문과 정답 페이지 정보를 불러온다.
    """

    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    return data


def contains_any_keyword(
        text: str,
        keywords: list[str],
) -> bool:

    """
    검색 결과에 진단용 키워드가 1개라도 있는지 확인
    """

    normalized_text = text.casefold()

    for keyword in keywords:
        # 여러 언어를 대소문자 구분 없이 비교하기 좋은 형태로 변환
        normalized_keyword = keyword.casefold()

        if normalized_keyword in normalized_text:
            return True

    return False


def status_mark(is_success: bool) -> str:
    """
    성공 여부를 출력용 O 또는 X로 변환
    """
    if is_success:
        return "O"
    return "X"


def main() -> None:
    eval_data = load_eval_data(path = eval_question_path)

    # json 파일에 저장되어있는 검색 검증용 데이터를 불러오는 작업
    questions = eval_data["retrieval_eval"]

    print("E5 Model loading..")
    model = load_embedding_model()
    client = open_local_vector_store(storage_path=qdrant_path)

    page_hit_at_1_count = 0
    page_hit_at_3_count = 0
    page_hit_at_5_count = 0
    evidence_hit_at_1_count = 0
    evidence_hit_at_3_count = 0
    evidence_hit_at_5_count = 0
    keyword_hit_at_5_count = 0
    page_reciprocal_rank_sum = 0.0
    evidence_reciprocal_rank_sum = 0.0

    question_results: list[dict] = []


    try:
        print()
        print("검색 평가 시작:", len(questions), "개 질문")
        print("=" * 90)

        for item in questions:
            qid = item["qid"]
            question = item["question"]
            relevant_pages = item["relevant_pages"]
            relevant_chunk_ids = item["relevant_chunk_ids"]
            keywords = item["keywords"]

            query_vector = embed_query(model=model, question=question)

            hits = search_vector_store(
                client = client,
                collection_name = collection_name,
                query_vector = query_vector,
                top_k = top_k
            )

            first_relevant_page_rank = None
            first_relevant_chunk_rank = None
            first_keyword_rank = None
            hit_summaries: list[dict] = []

            rank = 1

            for hit in hits:
                payload = hit.payload

                if payload is None:
                    rank += 1
                    continue

                page_number = int(payload["page_start"])
                chunk_id = str(payload["chunk_id"])
                text = str(payload["text"])

                if first_relevant_page_rank is None:
                    if page_number in relevant_pages:
                        first_relevant_page_rank = rank

                if first_relevant_chunk_rank is None:
                    if chunk_id in relevant_chunk_ids:
                        first_relevant_chunk_rank = rank

                if first_keyword_rank is None:
                    keyword_found = contains_any_keyword(text = text, keywords = keywords)

                    if keyword_found:
                        first_keyword_rank = rank

                hit_summary = {
                    "rank": rank,
                    "score": float(hit.score),
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "text": text
                }

                hit_summaries.append(hit_summary)
                rank += 1

            page_hit_at_1 = (first_relevant_page_rank is not None and first_relevant_page_rank <= 1)
            page_hit_at_3 = (first_relevant_page_rank is not None and first_relevant_page_rank <= 3)
            page_hit_at_5 = (first_relevant_page_rank is not None and first_relevant_page_rank <= 5)
            evidence_hit_at_1 = (first_relevant_chunk_rank is not None and first_relevant_chunk_rank <= 1)
            evidence_hit_at_3 = (first_relevant_chunk_rank is not None and first_relevant_chunk_rank <= 3)
            evidence_hit_at_5 = (first_relevant_chunk_rank is not None and first_relevant_chunk_rank <= 5)
            keyword_hit_at_5 = (first_keyword_rank is not None and first_keyword_rank <= 5)


            if page_hit_at_1:
                page_hit_at_1_count += 1

            if page_hit_at_3:
                page_hit_at_3_count += 1

            if page_hit_at_5:
                page_hit_at_5_count += 1

            if evidence_hit_at_1:
                evidence_hit_at_1_count += 1

            if evidence_hit_at_3:
                evidence_hit_at_3_count += 1

            if evidence_hit_at_5:
                evidence_hit_at_5_count += 1

            if keyword_hit_at_5:
                keyword_hit_at_5_count += 1


            page_reciprocal_rank = 0.0
            evidence_reciprocal_rank = 0.0

            if first_relevant_page_rank is not None:
                page_reciprocal_rank = (1.0 / first_relevant_page_rank)
                page_reciprocal_rank_sum += page_reciprocal_rank

            if first_relevant_chunk_rank is not None:
                evidence_reciprocal_rank = (1.0 / first_relevant_chunk_rank)
                evidence_reciprocal_rank_sum += evidence_reciprocal_rank

            question_result = {
                "qid": qid,
                "question": question,
                "relevant_pages": relevant_pages,
                "first_relevant_page_rank": (
                    first_relevant_page_rank
                ),
                "relevant_chunk_ids": relevant_chunk_ids,
                "first_relevant_chunk_rank": (
                    first_relevant_chunk_rank
                ),
                "first_keyword_rank": first_keyword_rank,
                "page_reciprocal_rank": page_reciprocal_rank,
                "evidence_reciprocal_rank": evidence_reciprocal_rank,
                "hits": hit_summaries,
            }

            question_results.append(question_result)

            print(
                qid,
                "Page@1:",
                status_mark(page_hit_at_1),
                "Evidence@1:",
                status_mark(evidence_hit_at_1),
                "Evidence@3:",
                status_mark(evidence_hit_at_3),
                "Evidence@5:",
                status_mark(evidence_hit_at_5),
                "첫 Evidence 순위:",
                first_relevant_chunk_rank,
                "Page@3:",
                status_mark(page_hit_at_3),
                "Page@5:",
                status_mark(page_hit_at_5),
            )
            print("질문:", question)
            print("기대 페이지:", relevant_pages)

            if hits:
                first_payload = hits[0].payload

                if first_payload is not None:
                    print(
                        "검색 1위:",
                        "p.",
                        first_payload["page_start"],
                        "|",
                        first_payload["chunk_id"],
                    )

            print("-" * 90)

    finally:
        client.close()

    question_count = len(questions)

    page_hit_at_1 = (
        page_hit_at_1_count / question_count
    )
    page_hit_at_3 = (
        page_hit_at_3_count / question_count
    )
    page_hit_at_5 = (
        page_hit_at_5_count / question_count
    )
    page_mrr_at_5 = (
        page_reciprocal_rank_sum / question_count
    )
    evidence_hit_at_1 = (
        evidence_hit_at_1_count / question_count
    )
    evidence_hit_at_3 = (
        evidence_hit_at_3_count / question_count
    )
    evidence_hit_at_5 = (
        evidence_hit_at_5_count / question_count
    )
    evidence_mrr_at_5 = (
        evidence_reciprocal_rank_sum / question_count
    )
    keyword_hit_at_5 = (
        keyword_hit_at_5_count / question_count
    )

    metrics = {
        "question_count": question_count,
        "top_k": top_k,
        "page_hit_at_1": page_hit_at_1,
        "page_hit_at_3": page_hit_at_3,
        "page_hit_at_5": page_hit_at_5,
        "page_mrr_at_5": page_mrr_at_5,
        "evidence_hit_at_1": evidence_hit_at_1,
        "evidence_hit_at_3": evidence_hit_at_3,
        "evidence_hit_at_5": evidence_hit_at_5,
        "evidence_mrr_at_5": evidence_mrr_at_5,
        "keyword_hit_at_5": keyword_hit_at_5,
    }

    print()
    print("전체 평가 결과")
    print("=" * 50)
    print("질문 수:", question_count)
    print("Page Hit@1:", round(page_hit_at_1, 3))
    print("Page Hit@3:", round(page_hit_at_3, 3))
    print("Page Hit@5:", round(page_hit_at_5, 3))
    print("Page MRR@5:", round(page_mrr_at_5, 3))
    print("Evidence Hit@1:", round(evidence_hit_at_1, 3))
    print("Evidence Hit@3:", round(evidence_hit_at_3, 3))
    print("Evidence Hit@5:", round(evidence_hit_at_5, 3))
    print("Evidence MRR@5:", round(evidence_mrr_at_5, 3))
    print(
        "Keyword Hit@5:",
        round(keyword_hit_at_5, 3),
        "(진단용)",
    )

    eval_results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        eval_results_dir
        / "latest_retrieval_eval.json"
    )

    output = {
        "schema_version": 1,
        "index_version": eval_data["index_version"],
        "embedding_model": MODEL_NAME,
        "collection_name": collection_name,
        "metrics": metrics,
        "question_results": question_results,
    }

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("결과 저장:", result_path)


if __name__ == "__main__":
    main()
