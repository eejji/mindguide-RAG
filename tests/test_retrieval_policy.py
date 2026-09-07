from src.retrieval_policy import calculate_threshold_metrics, should_accept

def test_score_equal_to_threshold_is_accepted():

    result = should_accept(score=0.88, threshold=0.88)

    assert result is True

def test_all_questions_rejected_does_not_crash():
    # 양성 질문과 무근거 질문을 하나씩 준비
    records  = [
        {
            "expected_supported": True,
            "top_1_score": 0.8
        },
        {
            "expected_supported": False,
            "top_1_score": 0.7
        }
    ]

    # 두 질문 모두 0.95 미만이므로 거절된다.
    metrics = calculate_threshold_metrics(
        records=records,
        threshold=0.95,
    )

    # 양성 질문은 잘못 거절됐으므로 FN이다.
    # 무근거 질문은 올바르게 거절됐으므로 TN이다.
    assert metrics["true_positive"] == 0
    assert metrics["false_positive"] == 0
    assert metrics["true_negative"] == 1
    assert metrics["false_negative"] == 1

    # 허용 건수가 0이어도 오류 없이 지표를 반환해야 한다.
    assert metrics["supported_precision"] == 0.0
    assert metrics["supported_coverage"] == 0.0
    assert metrics["false_accept_rate"] == 0.0


