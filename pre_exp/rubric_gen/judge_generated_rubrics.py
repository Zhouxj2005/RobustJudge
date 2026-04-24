from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from .api_qwen32b import call_qwen32b
except ImportError:
    from api_qwen32b import call_qwen32b


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent

RUBRIC_PATH = BASE_DIR / "rubric_with_dedup_oriented_prompt.json"
RESPONSES_PATH = ROOT_DIR / "model_res.json"
PROMPT_PATH = ROOT_DIR / "prompt.json"
RESULT_PATH = BASE_DIR / "generated_rubric_judge_result.json"

GEN_MODELS = ["qwen2.5-72b", "gpt-oss-120b", "qwen3-235b"]
N_SAMPLES = 8
JUDGE_MODEL = "qwen3-32b"
SYSTEM_PROMPT = "You are a rigorous LLM judge. Return only a valid JSON object."
MAX_SAMPLE_WORKERS = 8
MAX_TASK_WORKERS = len(GEN_MODELS) * 6
MAX_RETRIES_PER_SAMPLE = 3
REQUEST_TIMEOUT = 300

FLAG = False


def load_json(path: Path):
    return json.load(open(path, "r", encoding="utf-8"))


def save_json(data, path: Path) -> None:
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def parse_generated_rubric(rubric_response: str):
    text = strip_json_fence(rubric_response)
    if not text:
        raise ValueError("Empty generated rubric response")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        preview = text[:200].replace("\n", "\\n")
        raise ValueError(f"Failed to parse generated rubric JSON: {preview}") from exc
    return [{"description": item["criterion"].strip(), "points": 1} for item in data]


def parse_trial(text: str):
    text = strip_json_fence(text)
    return json.loads(text)["scoring_details"]


def get_score(
    query: str,
    response: str,
    rubric_list: list[dict],
    prompt_template: str,
    n: int,
    q_idx: int,
    gen_model: str,
    sample_key: str,
    max_sample_workers: int,
    max_retries_per_sample: int,
    request_timeout: int,
):
    global FLAG
    rubric_str = json.dumps(rubric_list, ensure_ascii=False)
    prompt = (
        prompt_template
        .replace("{{QUERY}}", query)
        .replace("{{RESPONSE}}", response)
        .replace("{{RUBRIC}}", rubric_str)
    )

    # if FLAG == False:
    #     print(f"Judging with prompt:\n{prompt}\n")
    #     FLAG = True

    print(
        f"start question={q_idx} model={gen_model} sample={sample_key} "
        f"prompt_len={len(prompt)} repeats={n}",
        flush=True,
    )

    def judge_once(repeat_idx: int):
        last_error = None
        for attempt_idx in range(max_retries_per_sample):
            # print(f"Attempting to judge sample...")
            try:
                return parse_trial(
                    call_qwen32b(
                        prompt,
                        system_prompt=SYSTEM_PROMPT,
                        model=JUDGE_MODEL,
                        temperature=1.0,
                        timeout=request_timeout,
                    )
                )
            except Exception as exc:
                last_error = exc
                print(
                    f"retry question={q_idx} model={gen_model} sample={sample_key} "
                    f"repeat={repeat_idx} attempt={attempt_idx + 1}/{max_retries_per_sample} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
        print(
            f"failed question={q_idx} model={gen_model} sample={sample_key} "
            f"repeat={repeat_idx} error={type(last_error).__name__}: {last_error}",
            flush=True,
        )
        return []

    with ThreadPoolExecutor(max_workers=min(max_sample_workers, n)) as executor:
        futures = [executor.submit(judge_once, repeat_idx) for repeat_idx in range(1, n + 1)]
        scores = [future.result() for future in futures]

    completed = sum(1 for score in scores if score)
    print(
        f"done question={q_idx} model={gen_model} sample={sample_key} "
        f"successful_repeats={completed}/{n}",
        flush=True,
    )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rubric-path", type=Path, default=RUBRIC_PATH)
    parser.add_argument("--responses-path", type=Path, default=RESPONSES_PATH)
    parser.add_argument("--prompt-path", type=Path, default=PROMPT_PATH)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-sample-workers", type=int, default=2)
    parser.add_argument("--max-task-workers", type=int, default=len(GEN_MODELS))
    parser.add_argument("--max-retries-per-sample", type=int, default=1)
    parser.add_argument("--request-timeout", type=int, default=180)
    args = parser.parse_args()

    rubric_data = load_json(args.rubric_path)
    responses = load_json(args.responses_path)
    prompt_template = load_json(args.prompt_path)["list-grader-template"]

    result = load_json(args.result_path) if args.result_path.exists() else {}
    selected = rubric_data[:args.max_questions] if args.max_questions else rubric_data

    for item in selected:
        q_idx = item["question_index"]
        q_key = str(q_idx)
        if q_key not in result:
            result[q_key] = {}

        pending = []
        for gen_model in GEN_MODELS:
            if gen_model not in result[q_key]:
                result[q_key][gen_model] = {}

            response = responses[q_idx][gen_model]

            for sample_idx, rubric_response in enumerate(item["rubric_responses"], start=1):
                sample_key = str(sample_idx)
                if sample_key in result[q_key][gen_model]:
                    continue
                pending.append(
                    (
                        gen_model,
                        sample_key,
                        response,
                        rubric_response,
                    )
                )

        if not pending:
            continue

        print(f"question={q_idx} pending_tasks={len(pending)}", flush=True)

        with ThreadPoolExecutor(max_workers=min(args.max_task_workers, len(pending))) as executor:
            futures = {
                executor.submit(
                    get_score,
                    item["question"],
                    response,
                    parse_generated_rubric(rubric_response),
                    prompt_template,
                    args.n_samples,
                    q_idx,
                    gen_model,
                    sample_key,
                    args.max_sample_workers,
                    args.max_retries_per_sample,
                    args.request_timeout,
                ): (gen_model, sample_key)
                for gen_model, sample_key, response, rubric_response in pending
            }

            for future in as_completed(futures):
                gen_model, sample_key = futures[future]
                result[q_key][gen_model][sample_key] = future.result()
                save_json(result, args.result_path)

if __name__ == "__main__":
    main()
