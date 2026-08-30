import json
from pathlib import Path


def test_eval_questions_have_unique_ids_and_qrels():
    project_root = Path(__file__).resolve().parents[1]
    eval_path = project_root / "data" / "eval_questions.json"

    with open(eval_path, encoding="utf-8") as file:
        eval_data = json.load(file)

    questions = eval_data["retrieval_eval"]
    seen_qids: set[str] = set()

    assert questions

    for item in questions:
        qid = item["qid"]

        assert qid not in seen_qids
        assert item["question"].strip()
        assert item["relevant_pages"]
        assert item["relevant_chunk_ids"]

        seen_qids.add(qid)
