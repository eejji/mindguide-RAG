import os

import streamlit as st

from stage11_demo import build_workflow, check_input


def main() -> None:
    # 브라우저 탭 이름과 화면 너비를 설정합니다.
    st.set_page_config(
        page_title="MindGuide 학습용 데모",
        layout="centered",
    )

    st.title("MindGuide")
    st.caption("의료 문서 검색 · 답변 초안 · 처리 경로 확인")

    st.warning(
        "학습용 프로토타입입니다. "
        "진단·처방이나 실제 의료진 연결을 제공하지 않습니다. "
        "개인정보나 실제 환자 정보는 입력하지 마세요."
    )

    st.caption("한 질문씩 처리하며 이전 대화를 기억하지 않습니다.")

    # 질문 작성과 제출 버튼을 하나로 묶습니다.
    with st.form("question_form"):
        question = st.text_area(
            "질문",
            placeholder=(
                "예: 일차 의료에서 우울증 선별에 "
                "사용할 수 있는 도구는 무엇인가?"
            ),
            height=120,
        )

        st.caption(
            "일반 질문은 질문과 검색된 참고 자료를 Claude API에 전송하며 "
            "이용요금이 발생할 수 있습니다."
        )

        submitted = st.form_submit_button("실행")

    # 제출하지 않았다면 화면만 보여주고 끝냅니다.
    if not submitted:
        return

    # 기존 LangGraph에 전달할 새로운 상태를 만듭니다.
    state = {
        "question": question.strip(),
        "route": "",
        "answer": "",
        "sources": [],
    }

    # API 키가 없으면 일반 검색·답변 생성만 중단합니다.
    # 고정 안내는 API 키 없이도 실행할 수 있습니다.
    # 기존 check_input 함수를 재사용합니다.
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        checked = check_input(state)

        if checked["route"] == "RAG":
            st.error(
                "Claude API 키가 설정되지 않았습니다. "
                "실행 터미널의 ANTHROPIC_API_KEY를 설정한 뒤 "
                "앱을 다시 시작해 주세요. "
                "키를 코드나 채팅에 붙여넣지 마세요."
            )
            return

    # 실제 질문 처리는 11단계 워크플로가 담당합니다.
    try:
        with st.spinner(
            "질문을 처리하고 있습니다. 첫 검색은 시간이 걸릴 수 있습니다."
        ):
            workflow = build_workflow()
            result = workflow.invoke(state)

    except Exception as error:
        # 오류 원문에 민감한 정보가 있을 수 있어 종류만 보여줍니다.
        st.error(
            "처리하지 못했습니다. "
            "API 설정·네트워크·로컬 Qdrant 상태를 확인해 주세요."
        )
        st.caption(f"오류 종류: {type(error).__name__}")
        return

    # 반환된 결과에서 처리 경로를 꺼내 보여줍니다.
    st.subheader("처리 결과")
    st.code(result["route"], language=None)

    # 생성된 답변과 고정 안내를 구분합니다.
    if result["route"] == "ANSWER_DRAFT":
        st.subheader("답변 초안")
        st.markdown(result["answer"])
        st.caption(
            "답변 내용과 인용의 정확성을 자동으로 검증한 결과는 아닙니다."
        )

    else:
        st.subheader("고정 안내")
        st.info(result["answer"])

    # 출처가 있을 때만 표시합니다.
    if result["sources"]:
        st.subheader("모델에 제공한 출처")

        for source in result["sources"]:
            # 출처 문자열의 첫 번째 줄을 접기/펼치기 제목으로 사용합니다.
            source_title = source.split("\n")[0]

            with st.expander(source_title):
                st.text(source)


if __name__ == "__main__":
    main()