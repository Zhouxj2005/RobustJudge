from __future__ import annotations

import json
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
OUTPUT_PATH = BASE_DIR / "rubric_with_simplified_prompt.json"
NUM_QUESTIONS = 100
NUM_SAMPLES = 6
MAX_RETRIES = 3


def generate_rubrics(n: int = NUM_QUESTIONS) -> None:
    with PROMPT_PATH.open("r", encoding="utf-8") as f:
        simplified_prompt = json.load(f)["simplified_prompt"]

    # print(simplified_prompt)
    # return

    dataset = load_dataset(
        "parquet",
        data_files={"train": str(DATASET_DIR / "**/*.parquet")},
        streaming=True,
    )
    items = list(islice(dataset["train"], n))

    results = []
    # 断点续写
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            results = json.load(f)

    done = {item["question_index"] for item in results}

    for idx, item in enumerate(items):
        if idx in done:
            continue

        question = item["prompt"][0]["content"] if isinstance(item["prompt"], list) else item["prompt"]
        filled_prompt = simplified_prompt.replace("{prompt}", question)
        rubric_responses = []
        parse_failed = False

        for _ in range(NUM_SAMPLES):
            rubric_response = None
            is_valid = False

            for _ in range(MAX_RETRIES + 1):
                rubric_response = call_qwen32b(filled_prompt)
                try:
                    candidate = json.loads(rubric_response.strip())
                    if (
                        isinstance(candidate, list)
                        and candidate
                        and all(
                            isinstance(cur, dict)
                            and set(cur.keys()) == {"criterion"}
                            and isinstance(cur["criterion"], str)
                            and cur["criterion"].strip()
                            for cur in candidate
                        )
                    ):
                        is_valid = True
                        break
                except json.JSONDecodeError:
                    pass

            if not is_valid:
                parse_failed = True

            rubric_responses.append(rubric_response)

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

        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Processed question {idx + 1}/{n}")


if __name__ == "__main__":
    generate_rubrics()
