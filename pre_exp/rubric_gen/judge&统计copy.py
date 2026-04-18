from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pre_exp.eval.client import Get
from pre_exp.rubric_gen.api_kimi import call_kimi
from pre_exp.rubric_gen.api_qwen32b import call_qwen32b


RUBRIC_PATH = BASE_DIR / "rubric_with_simplified_prompt.json"
RESPONSES_PATH = ROOT_DIR / "model_res.json"
RESULT_PATH = BASE_DIR / "generated_rubric_judge_result.json"
METRIC_PATH = BASE_DIR / "generated_rubric_r_sem.json"

DEFAULT_GEN_MODELS = ["qwen2.5-72b", "gpt-oss-120b", "qwen3-235b"]
DEFAULT_JUDGE_MODELS = ["Kimi-K2.5"]
DEFAULT_N_SAMPLES = 8
DEFAULT_BOOTSTRAP_B = 200
DEFAULT_SEED = 42
QWEN32B_SYSTEM_PROMPT = "You are a rigorous LLM judge. Return only a valid JSON object."
KIMI_SYSTEM_PROMPT = "You are a rigorous LLM judge. Return only a valid JSON object."
MAX_SAMPLE_WORKERS = 8
MAX_PARSE_RETRIES_PER_SAMPLE = 5
COMPACT_JUDGE_TEMPLATE = """You are a strict binary judge.

Evaluate the AI response against every rubric item.

Rules:
- Each rubric item has weight 1.
- Return score 1 only if the response fully satisfies that rubric item.
- Return score 0 if the item is missing, partially satisfied, ambiguous, or incorrect.
- Keep the reason very short.

[User Query]
{{QUERY}}

[AI Response]
{{RESPONSE}}

[Rubric List]
{{RUBRIC}}

Return only valid JSON in this format:
{
  "scoring_details": [
    {
      "rubric_index": 1,
      "is_met": true,
      "score": 1,
      "weight": 1,
      "reason": "short reason"
    }
  ]
}
"""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def _extract_score(rubric: dict[str, Any]) -> float:
    invalid_values = [None, np.nan]
    if "score" in rubric and rubric["score"] not in invalid_values:
        return float(rubric["score"])
    if (
        "is_met" in rubric
        and "weight" in rubric
        and rubric["is_met"] not in invalid_values
        and rubric["weight"] not in invalid_values
    ):
        return float(rubric["weight"]) if str(rubric["is_met"]).lower() == "true" else 0.0
    return np.nan


def parse_generated_rubric(rubric_response: str) -> list[dict[str, Any]]:
    try:
        candidate = json.loads(rubric_response.strip())
    except json.JSONDecodeError:
        return []

    if not isinstance(candidate, list):
        return []

    rubric_list: list[dict[str, Any]] = []
    for item in candidate:
        if (
            isinstance(item, dict)
            and set(item.keys()) == {"criterion"}
            and isinstance(item["criterion"], str)
            and item["criterion"].strip()
        ):
            rubric_list.append(
                {
                    "criterion": item["criterion"].strip(),
                    "points": 1,
                }
            )
    return rubric_list


def score_with_rubric_list(
    *,
    query: str,
    response: str,
    rubric_list: list[dict[str, Any]],
    judge_model: str,
    n_samples: int,
    model_client: Get,
    prompt_template: str,
) -> list[list[dict[str, Any]]]:
    rubric_str = json.dumps(rubric_list, ensure_ascii=False)
    prompt = (
        prompt_template
        .replace("{{QUERY}}", query)
        .replace("{{RESPONSE}}", response)
        .replace("{{RUBRIC}}", rubric_str)
    )

    def parse_trial(raw: Any) -> list[dict[str, Any]] | None:
        if raw is None:
            return None
        raw = str(raw).strip()
        if raw.startswith("```json"):
            raw = raw[len("```json"):].strip()
        elif raw.startswith("```"):
            raw = raw[len("```"):].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        scoring_details = parsed.get("scoring_details", [])
        if not isinstance(scoring_details, list):
            return None
        if len(scoring_details) != len(rubric_list):
            return None
        return scoring_details

    def fetch_one_kimi_trial() -> list[dict[str, Any]]:
        last_trial = []
        for _ in range(MAX_PARSE_RETRIES_PER_SAMPLE):
            raw = call_kimi(
                prompt,
                system_prompt=KIMI_SYSTEM_PROMPT,
                model="Kimi-K2.5",
            )
            parsed = parse_trial(raw)
            if parsed is not None:
                return parsed
            last_trial = []
        return last_trial

    if judge_model == "qwen3-32b":
        results = call_qwen32b(
            prompt,
            system_prompt=QWEN32B_SYSTEM_PROMPT,
            model=judge_model,
            n=n_samples,
            temperature=0.1,
        )
    elif judge_model in {"Kimi-K2.5", "kimi"}:
        with ThreadPoolExecutor(max_workers=min(MAX_SAMPLE_WORKERS, n_samples)) as executor:
            futures = [executor.submit(fetch_one_kimi_trial) for _ in range(n_samples)]
            return [future.result() for future in futures]
    else:
        results, _ = model_client.calc(prompt, model=judge_model, n=n_samples)

    parsed_results: list[list[dict[str, Any]]] = []
    for raw in results:
        parsed = parse_trial(raw)
        parsed_results.append(parsed if parsed is not None else [])
    return parsed_results


def run_judging(
    *,
    rubric_data: list[dict[str, Any]],
    responses: list[dict[str, str]],
    gen_models: list[str],
    judge_models: list[str],
    n_samples: int,
    output_path: Path,
    max_questions: int | None = None,
    question_indices: set[int] | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        result = load_json(output_path)
    else:
        result = {}

    client = Get()
    prompt_template = COMPACT_JUDGE_TEMPLATE

    selected = rubric_data
    if question_indices is not None:
        selected = [item for item in selected if item["question_index"] in question_indices]
    if max_questions is not None:
        selected = selected[:max_questions]

    for item in selected:
        q_idx = item["question_index"]
        q_key = str(q_idx)
        if q_key not in result:
            result[q_key] = {
                "question": item["question"],
                "rubric_samples": {},
            }
        for rubric_sample_idx, rubric_response in enumerate(item.get("rubric_responses", []), start=1):
            rubric_list = parse_generated_rubric(rubric_response)
            sample_key = str(rubric_sample_idx)
            if sample_key not in result[q_key]["rubric_samples"]:
                result[q_key]["rubric_samples"][sample_key] = {
                    "rubric_list": rubric_list,
                    "gen_models": {},
                }
            if not rubric_list:
                continue

            for gen_model in gen_models:
                response = responses[q_idx][gen_model]
                if gen_model not in result[q_key]["rubric_samples"][sample_key]["gen_models"]:
                    result[q_key]["rubric_samples"][sample_key]["gen_models"][gen_model] = {}
                for judge_model in judge_models:
                    if judge_model in result[q_key]["rubric_samples"][sample_key]["gen_models"][gen_model]:
                        continue
                    result[q_key]["rubric_samples"][sample_key]["gen_models"][gen_model][judge_model] = score_with_rubric_list(
                        query=item["question"],
                        response=response,
                        rubric_list=rubric_list,
                        judge_model=judge_model,
                        n_samples=n_samples,
                        model_client=client,
                        prompt_template=prompt_template,
                    )
            save_json(result, output_path)
    return result


def _valid_trial_scores(trials: list[list[dict[str, Any]]], num_rubrics: int) -> tuple[np.ndarray, np.ndarray]:
    rubric_scores: list[list[float]] = [[] for _ in range(num_rubrics)]
    query_scores: list[float] = []

    for trial in trials:
        if len(trial) < num_rubrics:
            continue
        cur_scores = []
        valid = True
        for rubric_idx in range(num_rubrics):
            score = _extract_score(trial[rubric_idx])
            if np.isnan(score):
                valid = False
                break
            cur_scores.append(float(score))
        if not valid:
            continue
        for rubric_idx, score in enumerate(cur_scores):
            rubric_scores[rubric_idx].append(score)
        query_scores.append(float(sum(cur_scores)))

    rubric_arr = np.array(rubric_scores, dtype=float) if rubric_scores else np.empty((0, 0))
    query_arr = np.array(query_scores, dtype=float)
    return rubric_arr, query_arr


def _bootstrap_sem(samples: np.ndarray, n_samples: int, B: int, rng: np.random.Generator) -> float:
    if len(samples) < n_samples or n_samples <= 0:
        return float("nan")
    bootstrap_means = np.zeros(B, dtype=float)
    for b in range(B):
        indices = rng.choice(len(samples), size=n_samples, replace=True)
        bootstrap_means[b] = float(np.mean(samples[indices]))
    return float(np.std(bootstrap_means, ddof=1))


def compute_r_sem(
    *,
    judge_result: dict[str, Any],
    judge_models: list[str],
    gen_models: list[str],
    n_samples: int,
    B: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    per_item: list[dict[str, Any]] = []
    query_level_by_judge: dict[str, list[float]] = defaultdict(list)
    rubric_level_by_judge: dict[str, list[float]] = defaultdict(list)

    for question_key, question_data in judge_result.items():
        for rubric_sample_key, rubric_sample_data in question_data["rubric_samples"].items():
            rubric_list = rubric_sample_data["rubric_list"]
            num_rubrics = len(rubric_list)
            if num_rubrics == 0:
                continue
            total_score = float(sum(item.get("points", 1) for item in rubric_list))

            for gen_model in gen_models:
                model_data = rubric_sample_data["gen_models"].get(gen_model, {})
                for judge_model in judge_models:
                    trials = model_data.get(judge_model, [])
                    rubric_scores, query_scores = _valid_trial_scores(trials, num_rubrics)
                    if len(query_scores) < n_samples or rubric_scores.size == 0:
                        continue

                    query_sem = _bootstrap_sem(query_scores, n_samples, B, rng)
                    query_r_sem = query_sem / total_score if total_score else float("nan")

                    rubric_r_sems = []
                    for rubric_idx in range(num_rubrics):
                        rubric_sem = _bootstrap_sem(rubric_scores[rubric_idx], n_samples, B, rng)
                        rubric_r_sems.append(rubric_sem)
                    avg_rubric_r_sem = float(np.nanmean(rubric_r_sems))

                    per_item.append(
                        {
                            "question_index": int(question_key),
                            "rubric_sample_index": int(rubric_sample_key),
                            "gen_model": gen_model,
                            "judge_model": judge_model,
                            "num_rubrics": num_rubrics,
                            "query_r_sem": query_r_sem,
                            "avg_rubric_r_sem": avg_rubric_r_sem,
                        }
                    )
                    if not np.isnan(query_r_sem):
                        query_level_by_judge[judge_model].append(query_r_sem)
                    if not np.isnan(avg_rubric_r_sem):
                        rubric_level_by_judge[judge_model].append(avg_rubric_r_sem)

    summary = {}
    for judge_model in judge_models:
        summary[judge_model] = {
            "mean_query_r_sem": float(np.nanmean(query_level_by_judge[judge_model])) if query_level_by_judge[judge_model] else float("nan"),
            "mean_avg_rubric_r_sem": float(np.nanmean(rubric_level_by_judge[judge_model])) if rubric_level_by_judge[judge_model] else float("nan"),
            "num_evaluated_pairs": len([item for item in per_item if item["judge_model"] == judge_model]),
        }

    all_query = [item["query_r_sem"] for item in per_item if not np.isnan(item["query_r_sem"])]
    all_rubric = [item["avg_rubric_r_sem"] for item in per_item if not np.isnan(item["avg_rubric_r_sem"])]
    return {
        "config": {
            "n_samples": n_samples,
            "bootstrap_B": B,
            "seed": seed,
            "gen_models": gen_models,
            "judge_models": judge_models,
        },
        "summary_by_judge": summary,
        "overall": {
            "mean_query_r_sem": float(np.nanmean(all_query)) if all_query else float("nan"),
            "mean_avg_rubric_r_sem": float(np.nanmean(all_rubric)) if all_rubric else float("nan"),
            "num_evaluated_pairs": len(per_item),
        },
        "per_item": per_item,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge generated rubric lists and compute query/rubric-level R-SEM.")
    parser.add_argument("--rubric-path", type=Path, default=RUBRIC_PATH)
    parser.add_argument("--responses-path", type=Path, default=RESPONSES_PATH)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--metric-path", type=Path, default=METRIC_PATH)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--bootstrap-b", type=int, default=DEFAULT_BOOTSTRAP_B)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gen-models", nargs="*", default=DEFAULT_GEN_MODELS)
    parser.add_argument("--judge-models", nargs="*", default=DEFAULT_JUDGE_MODELS)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--question-indices", nargs="*", type=int, default=None)
    args = parser.parse_args()

    rubric_data = load_json(args.rubric_path)
    responses = load_json(args.responses_path)
    judge_result = run_judging(
        rubric_data=rubric_data,
        responses=responses,
        gen_models=args.gen_models,
        judge_models=args.judge_models,
        n_samples=args.n_samples,
        output_path=args.result_path,
        max_questions=args.max_questions,
        question_indices=set(args.question_indices) if args.question_indices is not None else None,
    )
    metrics = compute_r_sem(
        judge_result=judge_result,
        judge_models=args.judge_models,
        gen_models=args.gen_models,
        n_samples=args.n_samples,
        B=args.bootstrap_b,
        seed=args.seed,
    )
    save_json(metrics, args.metric_path)


if __name__ == "__main__":
    main()
