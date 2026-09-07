def should_accept(
        score: float,
        threshold: float
) -> bool:

    """
    입력:
        score: 검색 결과의 Top-1 유사도 점수
        threshold: 근거를 허용할 최소 점수

    역할:
        검색 점수가 임계값 이상인지 확인
    
    반환:
        허용하면 True, 거절하면 False
    """

    if score >= threshold:
        return True

    return False

def calculate_threshold_metrics(
        records: list[dict],
        threshold: float
) -> dict:
    """
    입력:
        records: 질문별 기대 판정과 Top-1 점수가 들어 있는 목록
        threshold: 평가할 임계값

    역할:
        각 질문을 허용하거나 거절하고
        TP, FP, TN, FN을 계산

    반환:
        건수와 평가 지표가 들어있는 딕셔너리
    """

    if len(records) == 0:
        raise ValueError("평가할 질문이 없습니다.")

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for record in records:
        expected_supported = record["expected_supported"]
        score = record["top_1_score"]

        predicted_supported = should_accept(
            score = score,
            threshold = threshold
        )

        if expected_supported:
            if predicted_supported:
                true_positive += 1
            else:
                false_negative += 1
        else:
            if predicted_supported:
                false_positive += 1
            else:
                true_negative += 1
                

    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive
    accepted_count = true_positive + false_positive

    if positive_count == 0:
        raise ValueError("양성 질문이 없습니다.")

    if negative_count == 0:
        raise ValueError("무근거 질문이 없습니다.")

    supported_precision = 0.0

    if accepted_count > 0:
        supported_precision = (true_positive / accepted_count)

    supported_coverage = (true_positive / positive_count)

    false_accept_rate = (false_positive / negative_count)

    metrics = {
        "threshold": threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "supported_precision": supported_precision,
        "supported_coverage": supported_coverage,
        "false_accept_rate": false_accept_rate,
    }

    return metrics