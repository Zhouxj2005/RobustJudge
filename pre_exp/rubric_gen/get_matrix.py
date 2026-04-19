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
INPUT_PATH = BASE_DIR / "rubric_with_simplified_prompt.json"
OUTPUT_PATH = BASE_DIR / "rubric_matrix_100.json"
PROMPT_PATH = BASE_DIR / "rubric_dedup_prompt.json"

NUM_QUESTIONS = 100
TARGET_INPUT_BUDGET = 12000
MAX_CANDIDATES_PER_BATCH = 50
QUESTION_WORKERS = 100
BATCH_WORKERS = 8
MAX_API_CONCURRENCY = 24
SKIP_FAILED_BATCH = True

SHOW_PROMPT_ONCE = False
PROMPT_LOCK = Lock()
API_SEMAPHORE = Semaphore(MAX_API_CONCURRENCY)

with PROMPT_PATH.open("r", encoding="utf-8") as f:
    prompt_config = json.load(f)

SYSTEM_PROMPT = prompt_config.get(
    "system_prompt_with_question_oneshot_v1",
    prompt_config["system_prompt"],
)
BATCH_MATCH_PROMPT = prompt_config.get(
    "batch_match_prompt_with_question_oneshot_v1",
    prompt_config["batch_match_prompt"],
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


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def build_prompt(question: str, query_rubric: str, candidates: list[str]) -> str:
    candidate_text = "\n".join(f"{i}. {rubric}" for i, rubric in enumerate(candidates, 1))
    prompt = (
        BATCH_MATCH_PROMPT
        .replace("[[QUERY_RUBRIC]]", query_rubric)
        .replace("[[CANDIDATE_RUBRICS]]", candidate_text)
    )
    return prompt.replace("[[QUESTION]]", question)


def parse_batch_match_response(response: str | None, candidate_count: int) -> list[int]:
    if not isinstance(response, str):
        return []

    try:
        data = json.loads(response.strip())
    except json.JSONDecodeError:
        return []

    matched_indices = data.get("matched_indices")
    if not isinstance(matched_indices, list):
        return []

    return sorted(
        {
            index
            for index in matched_indices
            if isinstance(index, int) and 1 <= index <= candidate_count
        }
    )


def split_batches(unique_rubrics: list[str]) -> list[tuple[int, list[str]]]:
    batches = []
    start = 0

    while start < len(unique_rubrics):
        batch = []
        tokens = 0

        while start + len(batch) < len(unique_rubrics):
            rubric = unique_rubrics[start + len(batch)]
            cost = estimate_tokens(f"{len(batch) + 1}. {rubric}\n")

            if batch and (
                len(batch) >= MAX_CANDIDATES_PER_BATCH
                or tokens + cost > TARGET_INPUT_BUDGET
            ):
                break

            batch.append(rubric)
            tokens += cost

            if len(batch) >= MAX_CANDIDATES_PER_BATCH or tokens >= TARGET_INPUT_BUDGET:
                break

        batches.append((start, batch))
        start += len(batch)

    return batches


def show_prompt_once(prompt: str) -> None:
    global SHOW_PROMPT_ONCE
    if SHOW_PROMPT_ONCE:
        return

    with PROMPT_LOCK:
        if SHOW_PROMPT_ONCE:
            return
        print("Prompt for batch matching:")
        print(prompt)
        print("-" * 50)
        SHOW_PROMPT_ONCE = True


def run_batch_match(question: str, query_rubric: str, start: int, batch: list[str]) -> list[int]:
    prompt = build_prompt(question, query_rubric, batch)
    show_prompt_once(prompt)
    with API_SEMAPHORE:
        response = call_kimi(prompt, system_prompt=SYSTEM_PROMPT)
    batch_matches = parse_batch_match_response(response, len(batch))
    return [start + index - 1 for index in batch_matches]


def find_match_index(question: str, query_rubric: str, unique_rubrics: list[str]) -> int | None:
    batches = split_batches(unique_rubrics)
    if not batches:
        return None

    matched = []
    failed_batches = []
    with ThreadPoolExecutor(max_workers=min(BATCH_WORKERS, len(batches))) as executor:
        future_to_batch = {
            executor.submit(run_batch_match, question, query_rubric, start, batch): (start, batch)
            for start, batch in batches
        }
        for future in as_completed(future_to_batch):
            start, batch = future_to_batch[future]
            try:
                matched.extend(future.result())
            except Exception as exc:
                print(
                    f"Batch match failed once for query='{query_rubric[:60]}', "
                    f"start={start}, size={len(batch)}: {exc}"
                )
                failed_batches.append((start, batch))

    for start, batch in failed_batches:
        print(f"Retrying batch sequentially: start={start}, size={len(batch)}")
        try:
            matched.extend(run_batch_match(question, query_rubric, start, batch))
        except Exception as exc:
            print(
                f"Sequential retry failed for query='{query_rubric[:60]}', "
                f"start={start}, size={len(batch)}: {exc}"
            )
            if not SKIP_FAILED_BATCH:
                raise

    return min(matched) if matched else None


def build_matrix_for_question(item: dict) -> dict:
    question = item["question"]
    sample_rubrics = [parse_rubric_response(r) for r in item.get("rubric_responses", [])]

    unique_rubrics: list[str] = []
    sample_to_unique_indices: list[list[int]] = []
    total_rubrics = sum(len(rubrics) for rubrics in sample_rubrics)
    processed = 0

    for rubrics in sample_rubrics:
        indices = []
        for rubric in rubrics:
            processed += 1
            print(f"question_index={item['question_index']}: {processed}/{total_rubrics}")
            match_index = find_match_index(question, rubric, unique_rubrics)
            if match_index is None:
                unique_rubrics.append(rubric)
                match_index = len(unique_rubrics) - 1
            indices.append(match_index)
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
