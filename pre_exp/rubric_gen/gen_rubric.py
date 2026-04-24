from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from itertools import islice
from pathlib import Path

from datasets import load_dataset

try:
    from .api_qwen32b import call_qwen32b
except ImportError:
    from api_qwen32b import call_qwen32b


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DATASET_DIR = ROOT_DIR / "RubricHub_v1"
PROMPT_PATH = BASE_DIR / "rubric_syn_prompt.json"
# OUTPUT_PATH = BASE_DIR / "rubric_with_simplified_prompt.json"
OUTPUT_PATH = BASE_DIR / "rubric_with_dedup_oriented_prompt.json"
NUM_QUESTIONS = 100
NUM_SAMPLES = 6
MAX_RETRIES = 3
SAMPLE_WORKERS = 6
QUESTION_TIMEOUT_SECONDS = 360


def is_valid_rubric_list(candidate: object) -> bool:
    return (
        isinstance(candidate, list)
        and bool(candidate)
        and all(
            isinstance(cur, dict)
            and set(cur.keys()) == {"criterion"}
            and isinstance(cur["criterion"], str)
            and cur["criterion"].strip()
            for cur in candidate
        )
    )


def split_bundled_criterion(
    question: str,
    criterion: str,
    split_system_prompt: str,
    split_prompt_template: str,
) -> list[str]:
    filled_prompt = (
        split_prompt_template
        .replace("{prompt}", question)
        .replace("{criterion}", criterion)
    )

    try:
        split_response = call_qwen32b(
            filled_prompt,
            system_prompt=split_system_prompt,
            temperature=0.0,
        )
        split_candidate = json.loads(split_response.strip())
        if is_valid_rubric_list(split_candidate):
            return [item["criterion"].strip() for item in split_candidate]
    except Exception:
        pass

    return [criterion.strip()]


def postprocess_rubric_items(
    question: str,
    candidate: list[dict],
    split_system_prompt: str,
    split_prompt_template: str,
) -> list[dict]:
    processed: list[dict] = []
    seen: set[str] = set()

    with ThreadPoolExecutor(max_workers=min(len(candidate), SAMPLE_WORKERS)) as executor:
        future_to_idx = {
            executor.submit(
                split_bundled_criterion,
                question,
                item["criterion"].strip(),
                split_system_prompt,
                split_prompt_template,
            ): idx
            for idx, item in enumerate(candidate)
        }

        split_results: list[list[str] | None] = [None] * len(candidate)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                split_results[idx] = future.result()
            except Exception:
                split_results[idx] = [candidate[idx]["criterion"].strip()]

    for split_items in split_results:
        for split_item in split_items or []:
            if split_item and split_item not in seen:
                processed.append({"criterion": split_item})
                seen.add(split_item)

    return processed


def try_generate_one_rubric(
    question: str,
    filled_prompt: str,
    split_system_prompt: str,
    split_prompt_template: str,
) -> tuple[str | None, bool]:
    rubric_response = None
    is_valid = False

    for _ in range(MAX_RETRIES + 1):
        try:
            rubric_response = call_qwen32b(filled_prompt,temperature=1.0)
        except Exception:
            continue

        try:
            candidate = json.loads(rubric_response.strip())
            if isinstance(candidate, list):
                candidate = postprocess_rubric_items(
                    question,
                    candidate,
                    split_system_prompt,
                    split_prompt_template,
                )
            if is_valid_rubric_list(candidate):
                rubric_response = json.dumps(candidate, ensure_ascii=False)
                is_valid = True
                break
        except json.JSONDecodeError:
            pass

    if not is_valid:
        return "[]", False

    return rubric_response, True


def generate_rubrics(n: int = NUM_QUESTIONS) -> None:
    with PROMPT_PATH.open("r", encoding="utf-8") as f:
        prompt_config = json.load(f)
    prompt_key = os.environ.get("RUBRIC_PROMPT_KEY", "dedup_oriented_prompt_v4")
    if prompt_key not in prompt_config:
        raise KeyError(f"Prompt key not found: {prompt_key}")
    split_system_prompt_key = os.environ.get(
        "RUBRIC_SPLIT_SYSTEM_PROMPT_KEY",
        "rubric_split_system_prompt_v2",
    )
    split_prompt_key = os.environ.get(
        "RUBRIC_SPLIT_PROMPT_KEY",
        "rubric_split_prompt_v2",
    )
    if split_system_prompt_key not in prompt_config:
        raise KeyError(f"Split system prompt key not found: {split_system_prompt_key}")
    if split_prompt_key not in prompt_config:
        raise KeyError(f"Split prompt key not found: {split_prompt_key}")
    output_path = Path(os.environ.get("RUBRIC_OUTPUT_PATH", str(OUTPUT_PATH)))
    prompt_template = prompt_config[prompt_key]
    split_system_prompt = prompt_config[split_system_prompt_key]
    split_prompt_template = prompt_config[split_prompt_key]
    sample_workers = int(os.environ.get("RUBRIC_SAMPLE_WORKERS", str(SAMPLE_WORKERS)))

    print(f"Using prompt template: {prompt_key}")
    print(f"Using split prompt template: {split_prompt_key}")

    dataset = load_dataset(
        "parquet",
        data_files={"train": str(DATASET_DIR / "**/*.parquet")},
        streaming=True,
    )
    items = list(islice(dataset["train"], n))

    results = []
    # 断点续写
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            results = json.load(f)

    done = {item["question_index"] for item in results}

    for idx, item in enumerate(items):
        if idx in done:
            continue

        question = item["prompt"][0]["content"] if isinstance(item["prompt"], list) else item["prompt"]
        filled_prompt = prompt_template.replace("{prompt}", question)
        rubric_responses: list[str | None] = [None] * NUM_SAMPLES
        parse_failed = False

        with ThreadPoolExecutor(max_workers=min(sample_workers, NUM_SAMPLES)) as executor:
            future_to_sample_idx = {
                executor.submit(
                    try_generate_one_rubric,
                    question,
                    filled_prompt,
                    split_system_prompt,
                    split_prompt_template,
                ): sample_idx
                for sample_idx in range(NUM_SAMPLES)
            }
            try:
                for future in as_completed(future_to_sample_idx, timeout=QUESTION_TIMEOUT_SECONDS):
                    sample_idx = future_to_sample_idx[future]
                    try:
                        rubric_response, is_valid = future.result()
                    except Exception:
                        rubric_response, is_valid = "[]", False
                    rubric_responses[sample_idx] = rubric_response
                    if not is_valid:
                        parse_failed = True
            except TimeoutError:
                parse_failed = True
                print(f"Question {idx} timed out after {QUESTION_TIMEOUT_SECONDS}s")
                for future, sample_idx in future_to_sample_idx.items():
                    if rubric_responses[sample_idx] is None:
                        future.cancel()

        rubric_responses = [
            response if response is not None else "[]"
            for response in rubric_responses
        ]

        if parse_failed:
            print(f"Failed to parse rubric for question {idx}: {question}")

        results.append(
            {
                "question_index": idx,
                "question": question,
                "filled_prompt": filled_prompt,
                "rubric_responses": rubric_responses,
            }
        )

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Processed question {idx + 1}/{n}")


if __name__ == "__main__":
    num_questions = int(os.environ.get("RUBRIC_NUM_QUESTIONS", str(NUM_QUESTIONS)))
    generate_rubrics(num_questions)
