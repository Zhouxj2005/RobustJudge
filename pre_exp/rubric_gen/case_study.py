from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "rubric_with_simplified_prompt.json"
MATRIX_PATH = BASE_DIR / "rubric_matrix.json"
OUTPUT_PATH = BASE_DIR / "rubric_case_study_first3_v2.md"
NUM_QUESTIONS = 3
TOP_K = 3
SCORE_MARGIN = 0.08


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_rubric_response(rubric_response: str) -> list[str]:
    try:
        data = json.loads(rubric_response.strip())
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    return [
        item["criterion"].strip()
        for item in data
        if isinstance(item, dict)
        and set(item) == {"criterion"}
        and isinstance(item["criterion"], str)
        and item["criterion"].strip()
    ]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"name_\d+", "name", text)
    text = re.sub(r"\b\d+\b", "num", text)
    text = re.sub(r"[^a-z0-9+\-\s]", " ", text)
    return " ".join(text.split())


def extract_oxidation_states(text: str) -> set[str]:
    return set(re.findall(r"([+-]?\d+)\s+(?:oxidation|redox)\s+state", text.lower()))


def text_score(source: str, target: str) -> float:
    source_norm = normalize_text(source)
    target_norm = normalize_text(target)

    if source_norm == target_norm:
        return 1.0

    source_tokens = set(source_norm.split())
    target_tokens = set(target_norm.split())
    overlap = len(source_tokens & target_tokens)
    union = len(source_tokens | target_tokens) or 1
    jaccard = overlap / union
    seq_ratio = SequenceMatcher(None, source_norm, target_norm).ratio()
    score = 0.6 * seq_ratio + 0.4 * jaccard

    source_states = extract_oxidation_states(source)
    target_states = extract_oxidation_states(target)
    if source_states and target_states:
        if source_states == target_states:
            score += 0.25
        else:
            score -= 0.25

    return score


def get_active_unique_ids(row: list[int]) -> list[int]:
    return [idx + 1 for idx, value in enumerate(row) if value == 1]


def match_unique_ids(
    rubric: str,
    active_unique_ids: list[int],
    unique_rubrics: list[dict],
) -> list[int]:
    if not active_unique_ids:
        return []

    scored = []
    for unique_id in active_unique_ids:
        unique_text = unique_rubrics[unique_id - 1]["criterion"]
        scored.append((text_score(rubric, unique_text), unique_id))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0]
    cutoff = max(best_score - SCORE_MARGIN, 0.4)

    return [
        unique_id
        for score, unique_id in scored[:TOP_K]
        if score >= cutoff
    ]


def render_question_section(item: dict, matrix_item: dict) -> str:
    lines = [
        f"## Question {item['question_index']}",
        "",
        "### Prompt",
        item["question"],
        "",
        "### Unique Rubrics",
    ]

    unique_rubrics = matrix_item["unique_rubrics"]
    for unique in unique_rubrics:
        lines.append(f"{unique['rubric_index']}. {unique['criterion']}")

    lines.append("")
    lines.append("### Rubric Lists")

    rubric_lists = [parse_rubric_response(r) for r in item.get("rubric_responses", [])]
    for sample_idx, rubrics in enumerate(rubric_lists, start=1):
        lines.append("")
        lines.append(f"#### Sample {sample_idx}")
        active_unique_ids = get_active_unique_ids(matrix_item["matrix"][sample_idx - 1])
        lines.append(f"Active unique rubric ids from matrix: {active_unique_ids}")

        for rubric_idx, rubric in enumerate(rubrics, start=1):
            matched_ids = match_unique_ids(rubric, active_unique_ids, unique_rubrics)
            matched_text = ", ".join(map(str, matched_ids)) if matched_ids else "[]"
            lines.append(f"{rubric_idx}. {rubric}")
            lines.append(f"   - Matched unique rubric ids: {matched_text}")

    return "\n".join(lines)


def main() -> None:
    input_path = Path(os.environ.get("CASE_STUDY_INPUT_PATH", str(INPUT_PATH)))
    matrix_path = Path(os.environ.get("CASE_STUDY_MATRIX_PATH", str(MATRIX_PATH)))
    output_path = Path(os.environ.get("CASE_STUDY_OUTPUT_PATH", str(OUTPUT_PATH)))
    num_questions = int(os.environ.get("CASE_STUDY_NUM_QUESTIONS", str(NUM_QUESTIONS)))

    input_data = load_json(input_path)
    matrix_data = load_json(matrix_path)
    matrix_by_question = {
        item["question_index"]: item
        for item in matrix_data
        if "question_index" in item
    }

    sections = ["# Rubric Dedup Case Study", ""]
    for item in input_data[:num_questions]:
        question_index = item["question_index"]
        matrix_item = matrix_by_question.get(question_index)
        if not matrix_item:
            continue
        sections.append(render_question_section(item, matrix_item))
        sections.append("")

    output_path.write_text("\n".join(sections).strip() + "\n", encoding="utf-8")
    print(f"Saved case study markdown to {output_path}")


if __name__ == "__main__":
    main()
