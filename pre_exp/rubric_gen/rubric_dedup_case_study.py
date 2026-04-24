from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MATRIX_PATH = BASE_DIR / "rubric_matrix_list_match.json"
RUBRIC_PATH = BASE_DIR / "rubric_with_dedup_oriented_prompt.json"
OUTPUT_PATH = BASE_DIR / "rubric_list_dedup_case_study.md"


def parse_rubric_response(text: str) -> list[str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [
        item["criterion"].strip()
        for item in data
        if isinstance(item, dict)
        and isinstance(item.get("criterion"), str)
        and item["criterion"].strip()
    ]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    matrix_data = load_json(MATRIX_PATH)
    rubric_data = {
        item["question_index"]: item for item in load_json(RUBRIC_PATH)
    }

    lines: list[str] = ["# Rubric Dedup Case Study", ""]

    for matrix_item in matrix_data:
        question_index = matrix_item["question_index"]
        rubric_item = rubric_data.get(question_index, {})
        rubric_samples = [
            parse_rubric_response(text)
            for text in rubric_item.get("rubric_responses", [])
        ]

        lines.append(f"## Question {question_index}")
        lines.append("")
        lines.append("### Prompt")
        lines.append(matrix_item["question"])
        lines.append("")

        lines.append("### Unique Rubrics")
        for rubric in matrix_item.get("unique_rubrics", []):
            lines.append(f"{rubric['rubric_index']}. {rubric['criterion']}")
        lines.append("")

        lines.append("### Rubric Lists")
        for i, sample in enumerate(rubric_samples, 1):
            active = [
                j + 1
                for j, value in enumerate(matrix_item.get("matrix", [])[i - 1])
                if value
            ]
            lines.append(f"#### Sample {i}")
            lines.append(f"Active unique rubric ids from matrix: {active}")
            for j, rubric in enumerate(sample, 1):
                lines.append(f"{j}. {rubric}")
            lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved case study markdown to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
