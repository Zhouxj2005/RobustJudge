from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, Semaphore

try:
    from .api_kimi import call_kimi
except ImportError:
    from api_kimi import call_kimi


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "rubric_with_dedup_oriented_prompt.json"
OUTPUT_PATH = BASE_DIR / "rubric_matrix_list_match.json"
PROMPT_PATH = BASE_DIR / "rubric_dedup_prompt.json"

NUM_QUESTIONS = 100
QUESTION_WORKERS = 100
MAX_API_CONCURRENCY = 500

SHOW_PROMPT_ONCE = False
PROMPT_LOCK = Lock()
API_SEMAPHORE = Semaphore(MAX_API_CONCURRENCY)

with PROMPT_PATH.open("r", encoding="utf-8") as f:
    prompt_config = json.load(f)

SYSTEM_PROMPT = prompt_config.get(
    "system_prompt_list_match_with_question_requirement_v3",
    prompt_config.get(
        "system_prompt_list_match_with_question_cot_v2",
    prompt_config.get(
        "system_prompt_with_question_oneshot_v1",
        prompt_config["system_prompt"],
    ),
    ),
)
LIST_MATCH_PROMPT = prompt_config.get(
    "list_match_prompt_with_question_requirement_v3",
    prompt_config.get(
        "list_match_prompt_with_question_cot_v2",
    prompt_config.get(
        "batch_match_prompt_with_question_oneshot_v1",
        prompt_config["batch_match_prompt"],
    ),
    ),
)


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

def format_numbered_rubrics(rubrics: list[str]) -> str:
    return "\n".join(f"{i}. {rubric}" for i, rubric in enumerate(rubrics, 1))


def build_list_match_prompt(
    question: str, sample_rubrics: list[str], unique_rubrics: list[str]
) -> str:
    prompt = (
        LIST_MATCH_PROMPT
        .replace("[[QUESTION]]", question)
        .replace("[[SAMPLE_RUBRICS]]", format_numbered_rubrics(sample_rubrics))
        .replace("[[UNIQUE_RUBRICS]]", format_numbered_rubrics(unique_rubrics))
    )
    return prompt


def parse_list_match_response(
    response: str | None, sample_count: int, candidate_count: int
) -> tuple[list[int | None], str | None]:
    if not isinstance(response, str):
        return [None] * sample_count, "response is not a string"

    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [None] * sample_count, "response is not valid JSON array text"

    if not isinstance(data, list) or len(data) != sample_count:
        return [None] * sample_count, (
            f"response list length mismatch: expected {sample_count}, "
            f"got {len(data) if isinstance(data, list) else type(data).__name__}"
        )

    parsed: list[int | None] = []
    invalid_details: list[str] = []
    for idx, item in enumerate(data, 1):
        if item == 0:
            parsed.append(0)
            continue
        if isinstance(item, int) and 1 <= item <= candidate_count:
            parsed.append(item)
            continue
        parsed.append(None)
        invalid_details.append(
            f"position {idx}: got {item!r}, expected 0 or integer in [1, {candidate_count}]"
        )
    if invalid_details:
        return parsed, "; ".join(invalid_details)
    return parsed, None


def show_prompt_once(prompt: str) -> None:
    global SHOW_PROMPT_ONCE
    if SHOW_PROMPT_ONCE:
        return

    with PROMPT_LOCK:
        if SHOW_PROMPT_ONCE:
            return
        print("Prompt for list matching:")
        print(prompt)
        print("-" * 50)
        SHOW_PROMPT_ONCE = True


def run_list_match(
    question: str, sample_rubrics: list[str], unique_rubrics: list[str]
) -> list[int | None]:
    prompt = build_list_match_prompt(question, sample_rubrics, unique_rubrics)
    show_prompt_once(prompt)
    with API_SEMAPHORE:
        response = call_kimi(prompt, system_prompt=SYSTEM_PROMPT)
    batch_matches, parse_error = parse_list_match_response(
        response, len(sample_rubrics), len(unique_rubrics)
    )
    print(f"List match response: {response}")
    if any(index is None for index in batch_matches):
        raise ValueError(
            "Invalid list-match response: "
            f"{parse_error}. candidate_count={len(unique_rubrics)}, "
            f"sample_count={len(sample_rubrics)}, response={response}"
        )
    return [None if index == 0 else index - 1 for index in batch_matches]


def find_match_indices_for_sample(
    question: str, sample_rubrics: list[str], unique_rubrics: list[str]
) -> list[int | None]:
    if not unique_rubrics:
        return [None] * len(sample_rubrics)
    return run_list_match(question, sample_rubrics, unique_rubrics)


def build_matrix_for_question(item: dict) -> dict:
    question = item["question"]
    sample_rubrics = [parse_rubric_response(r) for r in item.get("rubric_responses", [])]

    unique_rubrics: list[str] = []
    sample_to_unique_indices: list[list[int]] = []
    sample_match_indices: list[list[int]] = []
    total_rubrics = sum(len(rubrics) for rubrics in sample_rubrics)
    processed = 0

    for rubrics in sample_rubrics:
        indices = []
        match_indices = find_match_indices_for_sample(question, rubrics, unique_rubrics)
        for rubric, match_index in zip(rubrics, match_indices):
            processed += 1
            print(f"question_index={item['question_index']}: {processed}/{total_rubrics}")
            if match_index is None:
                unique_rubrics.append(rubric)
                match_index = len(unique_rubrics) - 1
            indices.append(match_index)
        # Keep the per-rubric match order aligned with the original rubric list.
        # Indices are 0-based so they can be used directly for array lookup.
        sample_match_indices.append(indices.copy())
        sample_to_unique_indices.append(sorted(set(indices)))

    matrix = []
    for indices in sample_to_unique_indices:
        row = [0] * len(unique_rubrics)
        for index in indices:
            row[index] = 1
        matrix.append(row)

    return {
        "question_index": item["question_index"],
        "question": item["question"],
        "num_samples": len(sample_rubrics),
        "num_unique_rubrics": len(unique_rubrics),
        "unique_rubrics": [
            {"rubric_index": i + 1, "criterion": rubric}
            for i, rubric in enumerate(unique_rubrics)
        ],
        "sample_match_indices": sample_match_indices,
        "matrix": matrix,
    }


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_question(item: dict) -> tuple[int, dict]:
    question_index = item["question_index"]
    print(f"Processing question_index={question_index}")
    return question_index, build_matrix_for_question(item)


def main() -> None:
    input_path = Path(os.environ.get("RUBRIC_MATRIX_INPUT_PATH", str(INPUT_PATH)))
    output_path = Path(os.environ.get("RUBRIC_MATRIX_OUTPUT_PATH", str(OUTPUT_PATH)))
    num_questions = int(os.environ.get("RUBRIC_MATRIX_NUM_QUESTIONS", str(NUM_QUESTIONS)))

    data = load_json(input_path, [])
    results = load_json(output_path, [])
    completed_question_indices = {
        item["question_index"] for item in results if "question_index" in item
    }
    pending_items = []

    for i, item in enumerate(data):
        if i >= num_questions:
            break
        if item["question_index"] in completed_question_indices:
            print(f"Skipping question_index={item['question_index']}")
            continue
        pending_items.append(item)

    if not pending_items:
        save_json(output_path, results)
        print(f"Saved matrix results to {output_path}")
        return

    with ThreadPoolExecutor(max_workers=min(QUESTION_WORKERS, len(pending_items))) as executor:
        futures = [executor.submit(process_question, item) for item in pending_items]
        for future in as_completed(futures):
            try:
                question_index, result = future.result()
            except Exception as exc:
                print(f"Question failed and was skipped: {exc}")
                continue
            results.append(result)
            results.sort(key=lambda item: item.get("question_index", float("inf")))
            save_json(output_path, results)
            print(f"Finished question_index={question_index}")

    print(f"Saved matrix results to {output_path}")


if __name__ == "__main__":
    main()
