import re
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

class WorkflowState(TypedDict):
    """
    각 단계가 공유하는 질문, 처리 결과, 답변, 출처.
    """
    question: str # 질문
    route: str # 처리경로 or 최종 처리 상태
    answer: str # 사용자에게 보여줄 답변
    sources: list[str] # 검색한 자료의 출처 목록

def check_input(state: WorkflowState) -> dict:
    """
    입력: 질문이 답긴 상태 / 반환: 다음 처리를 결정할 route
    """
    print("1. 입력 검사")
    question = state["question"].strip()
    compact = "".join(question.split()).casefold()

    if not compact:
        return {"route": "EMPTY_INPUT"}


    negative_phrases = [
        "죽고싶지않",
        "자살하고싶지않",
        "스스로해치고싶지않"
    ]

    risk_text = compact
    needs_safety_check = False

    for phrase in negative_phrases:
        if phrase in risk_text:
            risk_text = risk_text.replace(phrase, "")
            needs_safety_check = True
            

    # 위기 표현 확인
    emergency_phrases = [
        "죽고싶", "자살하고싶", "스스로해치", "목숨을끊"
    ]

    for phrase in emergency_phrases:
        if phrase in risk_text:
            return {"route": "EMERGENCY_GUIDANCE"}
        
    if needs_safety_check:
        return {"route": "SAFETY_CHECK"}
    
    # 개인정보 감지 범위: 휴대폰, 이메일, 주민번호
    pii_patterns = [
        r"(?<!\d)01[016789][ -]?\d{3,4}[ -]?\d{4}(?!\d)",
        r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        r"(?<!\d)\d{6}[ -]?[1-4]\d{6}(?!\d)",
    ]

    for pattern in pii_patterns:
        if re.search(pattern, question):
            return {"route": "PII_BLOCK"}

    clinical_phrases = [
        "진단해줘", "진단해주세요",
        "내가우울증", "나우울증", "저우울증", "제가우울증",
        "약을몇알", "약몇알",
        "용량을늘", "용량늘", "용량을줄", "용량줄",
        "약을끊어도", "약끊어도",
    ]

    for phrase in clinical_phrases:
        if phrase in compact:
            return {"route": "CLINICIAN_REVIEW"}
        
    return {"route": "RAG"}


def choose_next(state: WorkflowState) -> str:
    """
    검사 결과에 따라 실행할 노드의 이름 반환
    """
    if state["route"] == "RAG":
        return "rag"

    return "guide"

def guide_node(state: WorkflowState) -> dict:
    """
    모델 호출 없이, 처리 결과에 맞는 안내문 반환
    """
    print("2. 고정 안내")

    messages = {
        "EMPTY_INPUT": "질문을 입력해 주세요.",

        "EMERGENCY_GUIDANCE": (
            "지금 안전이 걱정된다면 혼자 있지 말고 "
            "가까운 사람에게 도움을 요청하세요. "
            "한국에서 즉각적인 위험은 119, "
            "자살예방 상담은 109로 연락할 수 있습니다."
        ),

        "PII_BLOCK": (
            "개인정보로 보이는 내용이 있어 처리를 중단했습니다. "
            "전화번호·이메일·주민번호 등을 제거하고 다시 질문해 주세요."
        ),

        "CLINICIAN_REVIEW": (
            "개인별 진단이나 약물 변경은 담당 의료진과 상의해 주세요. "
            "이 프로그램은 실제 의료진에게 요청을 전달하지 않습니다."
        ),

        "SAFETY_CHECK": (
            "현재 자신을 해칠 생각이나 계획이 있나요? "
            "현재 안전을 확인할 필요가 있어 일반 답변 생성을 보류했습니다. "
            "즉각적인 위험이 있다면 한국에서는 119, "
            "자살예방 상담은 109로 도움을 요청할 수 있습니다."
        ),
    }

    return {
        "answer": messages[state["route"]],
        "sources": [],
    }


def rag_node(state: WorkflowState) -> dict:
    """
    일반 질문에서만 기존 검색, 답변 생성 함수 실행
    """
    print("2. 검색 및 답변 생성")

    # 이 경로에 들어왔을 때만 검색, LLM 코드 불러오기

    from stage10_demo import build_context, generate_answer
    from src.embeddings import embed_query, load_embedding_model
    from src.vector_store import (
        open_local_vector_store,
        search_vector_store
    )

    model = load_embedding_model()

    query_vector = embed_query(
        model=model,
        question=state["question"]
    )

    client = open_local_vector_store(Path("data/qdrant_local"))

    try:
        hits = search_vector_store(
            client = client,
            collection_name = "depression_guideline_chunks",
            query_vector = query_vector,
            top_k = 3
        )
    finally:
        client.close()

    context, sources = build_context(hits)

    # 본문이 없는 검색 결과로는 Claude를 호출하지 않음
    if not context.strip():
        return {
            "route" : "NO_EVIDENCE",
            "answer" : "검색된 본문이 없어 답변을 생성하지 않습니다.",
            "sources" : []
        }

    answer = generate_answer(state["question"], context)

    return {
        "route" : "ANSWER_DRAFT",
        "answer" : answer,
        "sources" : sources
    }

def build_workflow():
    """처리 함수들을 등록하고, 실행 순서 연결"""

    builder = StateGraph(WorkflowState)

    builder.add_node("check_input", check_input)
    builder.add_node("guide", guide_node)
    builder.add_node("rag", rag_node)

    builder.add_edge(START, "check_input")

    # 검사 후 choose_next가 선택한 한 경로만 실행
    builder.add_conditional_edges(
        "check_input",
        choose_next,
        {"guide": "guide", "rag":"rag"}
    )

    builder.add_edge("guide", END)
    builder.add_edge("rag", END)

    return builder.compile()


def main() -> None:
    question = input("질문: ").strip()
    workflow = build_workflow()

    # 매 실행마다 새로운 상태 만들어서 시작
    result = workflow.invoke({
        "question": question,
        "route": "",
        "answer": "",
        "sources": []
    })

    print("\n처리 결과: ", result["route"])
    print("답변: ", result["answer"])

    if result["sources"]:
        print("\n모델에 제공한 출처: ")

        for source in result["sources"]:
            print(source)


if __name__ == "__main__":
    main()

