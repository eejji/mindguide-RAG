import json
from pathlib import Path

from src.embeddings import embed_query, load_embedding_model
from src.vector_store import (
    open_local_vector_store,
    search_vector_store
)

from src.retrieval_policy import calculate_threshold_metrics


positive_questions_path = Path("data/eval_questions.json")
negative_questions_path = Path("data/rejection_questions.json")
qdrant_path = Path("data/qdrant_local")

collection_name = "depression_guideline_chunks"

def load_json(path: Path) -> dict:
    """
    JSON 파일을 읽어 Python 딕셔너리로 변환
    """
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    return data

def evaluate_question(
        model,
        client,
        qid: str,
        question: str,
        expected_supported: bool,
        difficulty: str,
) -> dict:
    """
    질문을 임베딩하고 Qdrant에서 가장 가까운 청크 1개 찾기
    """

    query_vector = embed_query(
        model=model, question=question
    )

    hits = search_vector_store(
        client=client,
        collection_name=collection_name,
        query_vector = query_vector,
        top_k = 1
    )

    if len(hits) == 0:
        raise RuntimeError("검색 결과가 없습니다.")

    first_hit = hits[0]

    if first_hit.payload is None:
        raise RuntimeError("검색 결과의 payload가 없습니다.")

    payload = first_hit.payload

    result = {
        "qid": qid,
        "question": question,
        "expected_supported": expected_supported,
        "difficulty": difficulty,
        "top_1_score": float(first_hit.score),
        "top_1_chunk_id": str(payload["chunk_id"]),
        "top_1_page": int(payload["page_start"]),
    }

    return result


def print_result(result: dict) -> None:
    """
    질문 하나의 검색 결과를 읽기 좋게 출력한다.
    """

    if result["expected_supported"]:
        expected_label = "SUPPORTED"
    else:
        expected_label = "NO_EVIDENCE"

    print()
    print(
        result["qid"],
        "| 기대:",
        expected_label,
        "| 난이도:",
        result["difficulty"],
    )
    print("질문:", result["question"])
    print("Top-1 점수:", round(result["top_1_score"], 4))
    print("Top-1 청크:", result["top_1_chunk_id"])
    print("Top-1 페이지:", result["top_1_page"])
    print("-" * 80)


def main() -> None:
    positive_data = load_json(positive_questions_path)
    negative_data = load_json(negative_questions_path)

    positive_questions = positive_data['retrieval_eval']
    negative_questions = negative_data['rejection_eval']

    print("임베딩 모델 불러오는 중..")
    model = load_embedding_model()

    client = open_local_vector_store(storage_path=qdrant_path)

    results: list[dict] = []

    try:
        print()
        print("양성 질문 검색 시작")

        for item in positive_questions:
            result = evaluate_question(
                model=model,
                client=client,
                qid=item["qid"],
                question=item["question"],
                expected_supported=True,
                difficulty="positive",
            )

            results.append(result)
            print_result(result)

        print()
        print("무근거 질문 검색 시작")

        for item in negative_questions:
            result = evaluate_question(
                model=model,
                client=client,
                qid=item["qid"],
                question=item["question"],
                expected_supported=False,
                difficulty=item["difficulty"],
            )

            results.append(result)
            print_result(result)

    finally:
        client.close()

    positive_min_score = None
    negative_max_score = None

    for result in results:
        score = result["top_1_score"]

        if result["expected_supported"]:
            if positive_min_score is None:
                positive_min_score = score
            elif score < positive_min_score:
                positive_min_score = score
        else:
            if negative_max_score is None:
                negative_max_score = score
            elif score > negative_max_score:
                negative_max_score = score

    print()
    print("점수 분포 요약")
    print("=" * 50)
    print("양성 질문 최저 점수:", round(positive_min_score, 4))
    print("무근거 질문 최고 점수:", round(negative_max_score, 4))

    score_gap = positive_min_score - negative_max_score

    print("점수 간격:", round(score_gap, 4))

    if score_gap > 0:
        candidate_threshold = (
            positive_min_score + negative_max_score
        ) / 2

        print("두 그룹이 겹치지 않습니다.")
        print(
            "임시 임계값 후보:",
            round(candidate_threshold, 4),
        )
    else:
        print("양성 질문과 무근거 질문의 점수가 겹칩니다.")
        print("모든 질문을 완벽히 나누는 임계값은 없습니다.")

    # 검색 결과 16개를 비교 함수에 전달해서 실행한다.
    compare_threshold(records=results)

def compare_threshold(records: list[dict]) -> None:
    """
    입력:
        질문 16개의 기대 판정과 점수가 담긴 목록
    역할:
        임계값을 바꿔가며 TP, FP, TN, FN과 지표 계산
    반환:
        비교 결과를 화면에 출력
    """

    print()
    print("임계값별 평가 결과")
    print("=" * 100)

    # 정수로 증가시킨 뒤 1000으로 나누어 임계값을 만든다.
    threshold_number = 750

    while threshold_number <= 980:
        threshold = threshold_number / 1000

        # 같은 검색 결과에 새로운 임계값만 적용한다.
        metrics = calculate_threshold_metrics(
            records=records,
            threshold=threshold,
        )

        print(
            f"{threshold:.3f}",
            "| TP:", metrics["true_positive"],
            "FP:", metrics["false_positive"],
            "TN:", metrics["true_negative"],
            "FN:", metrics["false_negative"],
            "| Precision:",
            round(metrics["supported_precision"], 3),
            "Coverage:",
            round(metrics["supported_coverage"], 3),
            "FAR:",
            round(metrics["false_accept_rate"], 3),
        )

        # 다음 임계값으로 이동한다.
        threshold_number += 5


if __name__ == "__main__":
    main()