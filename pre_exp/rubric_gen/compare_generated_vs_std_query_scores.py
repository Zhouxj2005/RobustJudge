from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent

DEFAULT_STD_RESULT_PATH = ROOT_DIR / "result.json"
DEFAULT_GROUND_TRUTH_PATH = ROOT_DIR / "ground_truth.json"
DEFAULT_GENERATED_RESULT_PATH = BASE_DIR / "generated_rubric_judge_result.json"
DEFAULT_GENERATED_RUBRIC_PATH = BASE_DIR / "rubric_with_dedup_oriented_prompt.json"
DEFAULT_OUTPUT_JSON = BASE_DIR / "generated_vs_std_query_score_comparison.json"

DEFAULT_STD_JUDGE_MODEL = "qwen3-32b"
DEFAULT_MIN_STD_VALID_SAMPLES = 12
DEFAULT_MIN_GEN_VALID_SAMPLES = 1


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json") :].strip()
    elif text.startswith("```"):
        text = text[len("```") :].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def parse_generated_rubric_points(rubric_response: str) -> list[float]:
    try:
        data = json.loads(strip_json_fence(rubric_response))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    points = []
    for item in data:
        if not isinstance(item, dict):
            return []
        criterion = item.get("criterion")
        if not isinstance(criterion, str) or not criterion.strip():
            return []
        # In judge_generated_rubrics.py, every generated rubric is judged with points=1.
        points.append(1.0)
    return points


def extract_score(rubric: dict[str, Any]) -> float:
    weight = rubric.get("weight")
    try:
        numeric_weight = float(weight) if weight is not None else None
    except (TypeError, ValueError):
        numeric_weight = None

    score = rubric.get("score")
    if score is not None:
        try:
            score = float(score)
            if (
                not np.isnan(score)
                and score >= 0
                and (numeric_weight is None or (numeric_weight > 0 and score <= numeric_weight))
            ):
                return score
        except (TypeError, ValueError):
            pass

    is_met = rubric.get("is_met")
    if is_met is not None and numeric_weight is not None:
        try:
            if numeric_weight <= 0:
                return np.nan
            return numeric_weight if str(is_met).lower() == "true" else 0.0
        except (TypeError, ValueError):
            pass
    return np.nan


def extract_generated_binary_score(rubric: dict[str, Any]) -> float:
    is_met = rubric.get("is_met")
    if is_met is not None:
        text = str(is_met).strip().lower()
        if text == "true":
            return 1.0
        if text == "false":
            return 0.0

    score = rubric.get("score")
    if score is not None:
        try:
            score = float(score)
            if not np.isnan(score) and 0.0 <= score <= 1.0:
                return score
        except (TypeError, ValueError):
            pass
    return np.nan


def extract_positive_weight(rubric: dict[str, Any]) -> float:
    weight = rubric.get("weight")
    if weight is None:
        return np.nan
    try:
        weight = float(weight)
    except (TypeError, ValueError):
        return np.nan
    return weight if weight > 0 else np.nan


def build_std_total_lookup(ground_truth: dict[str, Any]) -> dict[tuple[int, str], dict[str, float]]:
    lookup: dict[tuple[int, str], dict[str, float]] = {}
    for q_key, question_data in ground_truth.items():
        q_idx = int(q_key)
        for gen_model, rubric_scores in question_data.items():
            total = 0.0
            valid = True
            for item in rubric_scores:
                try:
                    weight = float(item.get("weight", 1))
                except (TypeError, ValueError):
                    valid = False
                    break
                total += weight
            if valid and total > 0:
                lookup[(q_idx, gen_model)] = {
                    "num_rubrics": float(len(rubric_scores)),
                }
    return lookup


def build_generated_total_lookup(rubric_data: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, float]]:
    lookup: dict[tuple[int, int], dict[str, float]] = {}
    for item in rubric_data:
        q_idx = int(item["question_index"])
        for sample_idx, rubric_response in enumerate(item.get("rubric_responses", []), start=1):
            points = parse_generated_rubric_points(rubric_response)
            total = float(sum(points))
            if total > 0:
                lookup[(q_idx, sample_idx)] = {
                    "num_rubrics": float(len(points)),
                }
    return lookup


def collect_std_query_scores(
    std_result: dict[str, Any],
    std_total_lookup: dict[tuple[int, str], dict[str, float]],
    judge_model: str,
    min_valid_samples: int,
) -> dict[tuple[int, str], dict[str, Any]]:
    collected: dict[tuple[int, str], dict[str, Any]] = {}
    for q_key, question_data in std_result.items():
        q_idx = int(q_key)
        for gen_model, judge_results in question_data.items():
            trials = judge_results.get(judge_model)
            if not isinstance(trials, list):
                continue
            rubric_meta = std_total_lookup.get((q_idx, gen_model))
            if rubric_meta is None:
                continue
            expected_num_rubrics = int(rubric_meta["num_rubrics"])
            if expected_num_rubrics <= 0:
                continue

            relative_scores = []
            for trial in trials:
                if not isinstance(trial, list):
                    continue
                if len(trial) < expected_num_rubrics:
                    continue
                trial_items = trial[:expected_num_rubrics]
                scores = np.array([extract_score(rubric) for rubric in trial_items], dtype=float)
                weights = np.array([extract_positive_weight(rubric) for rubric in trial_items], dtype=float)
                if scores.size == 0 or np.any(np.isnan(scores)) or np.any(np.isnan(weights)):
                    continue
                total_points = float(weights.sum())
                if total_points <= 0:
                    continue
                relative_scores.append(float(scores.sum() / total_points))

            if len(relative_scores) < min_valid_samples:
                continue

            collected[(q_idx, gen_model)] = {
                "relative_scores": relative_scores,
                "mean_score": float(np.mean(relative_scores)),
                "num_valid_samples": len(relative_scores),
            }
    return collected


def collect_generated_query_scores(
    generated_result: dict[str, Any],
    generated_total_lookup: dict[tuple[int, int], dict[str, float]],
    min_valid_samples: int,
) -> dict[tuple[int, str], dict[str, Any]]:
    collected: dict[tuple[int, str], dict[str, Any]] = {}
    for q_key, question_data in generated_result.items():
        q_idx = int(q_key)
        for gen_model, sample_results in question_data.items():
            pair_key = (q_idx, gen_model)
            sample_details = []
            fused_scores = []

            for sample_key, trials in sample_results.items():
                sample_idx = int(sample_key)
                rubric_meta = generated_total_lookup.get((q_idx, sample_idx))
                if rubric_meta is None:
                    continue
                expected_num_rubrics = int(rubric_meta["num_rubrics"])
                if expected_num_rubrics <= 0:
                    continue

                relative_scores = []
                for trial in trials:
                    if not isinstance(trial, list):
                        continue
                    if len(trial) < expected_num_rubrics:
                        continue
                    scores = np.array(
                        [extract_generated_binary_score(rubric) for rubric in trial[:expected_num_rubrics]],
                        dtype=float,
                    )
                    if scores.size == 0 or np.any(np.isnan(scores)):
                        continue
                    relative_scores.append(float(scores.sum() / expected_num_rubrics))

                if len(relative_scores) < min_valid_samples:
                    continue

                fused_scores.extend(relative_scores)
                sample_details.append(
                    {
                        "sample_idx": sample_idx,
                        "relative_scores": relative_scores,
                        "mean_score": float(np.mean(relative_scores)),
                        "num_valid_samples": len(relative_scores),
                    }
                )

            if not sample_details or not fused_scores:
                continue

            sample_mean_scores = [item["mean_score"] for item in sample_details]
            collected[pair_key] = {
                "fused_relative_scores": fused_scores,
                "fused_mean_score": float(np.mean(fused_scores)),
                "num_fused_valid_samples": len(fused_scores),
                "sample_details": sample_details,
                "avg_over_sample_means": float(np.mean(sample_mean_scores)),
                "max_sample_mean": float(np.max(sample_mean_scores)),
                "min_sample_mean": float(np.min(sample_mean_scores)),
                "num_valid_samples": len(sample_details),
            }
    return collected


def summarize_comparison(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    diffs = [record[field] - record["std_mean_score"] for record in records]
    higher = sum(diff > 0 for diff in diffs)
    lower = sum(diff < 0 for diff in diffs)
    equal = len(diffs) - higher - lower
    return {
        "comparison_field": field,
        "num_pairs": len(records),
        "num_generated_higher": higher,
        "num_generated_lower": lower,
        "num_equal": equal,
        "mean_difference": float(np.mean(diffs)) if diffs else float("nan"),
        "median_difference": float(np.median(diffs)) if diffs else float("nan"),
        "win_rate": float(higher / len(diffs)) if diffs else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare query-level scores judged with generated rubrics vs standard rubrics (qwen3-32b judge only)."
    )
    parser.add_argument("--std-result-path", type=Path, default=DEFAULT_STD_RESULT_PATH)
    parser.add_argument("--ground-truth-path", type=Path, default=DEFAULT_GROUND_TRUTH_PATH)
    parser.add_argument("--generated-result-path", type=Path, default=DEFAULT_GENERATED_RESULT_PATH)
    parser.add_argument("--generated-rubric-path", type=Path, default=DEFAULT_GENERATED_RUBRIC_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--std-judge-model", type=str, default=DEFAULT_STD_JUDGE_MODEL)
    parser.add_argument("--min-std-valid-samples", type=int, default=DEFAULT_MIN_STD_VALID_SAMPLES)
    parser.add_argument("--min-gen-valid-samples", type=int, default=DEFAULT_MIN_GEN_VALID_SAMPLES)
    args = parser.parse_args()

    std_total_lookup = build_std_total_lookup(load_json(args.ground_truth_path))
    generated_total_lookup = build_generated_total_lookup(load_json(args.generated_rubric_path))

    std_scores = collect_std_query_scores(
        load_json(args.std_result_path),
        std_total_lookup,
        judge_model=args.std_judge_model,
        min_valid_samples=args.min_std_valid_samples,
    )
    generated_scores = collect_generated_query_scores(
        load_json(args.generated_result_path),
        generated_total_lookup,
        min_valid_samples=args.min_gen_valid_samples,
    )

    common_keys = sorted(set(std_scores.keys()) & set(generated_scores.keys()))
    records = []
    for q_idx, gen_model in common_keys:
        std_item = std_scores[(q_idx, gen_model)]
        gen_item = generated_scores[(q_idx, gen_model)]
        records.append(
            {
                "question_index": q_idx,
                "gen_model": gen_model,
                "std_mean_score": std_item["mean_score"],
                "std_num_valid_samples": std_item["num_valid_samples"],
                "generated_fused_mean_score": gen_item["fused_mean_score"],
                "generated_avg_over_sample_means": gen_item["avg_over_sample_means"],
                "generated_max_sample_mean": gen_item["max_sample_mean"],
                "generated_min_sample_mean": gen_item["min_sample_mean"],
                "generated_num_valid_samples": gen_item["num_valid_samples"],
                "generated_num_fused_valid_samples": gen_item["num_fused_valid_samples"],
                "generated_minus_std_fused": gen_item["fused_mean_score"] - std_item["mean_score"],
                "generated_minus_std_avg_over_samples": gen_item["avg_over_sample_means"] - std_item["mean_score"],
                "generated_sample_details": gen_item["sample_details"],
            }
        )

    records.sort(key=lambda item: item["generated_minus_std_fused"], reverse=True)

    output = {
        "config": {
            "std_result_path": str(args.std_result_path),
            "ground_truth_path": str(args.ground_truth_path),
            "generated_result_path": str(args.generated_result_path),
            "generated_rubric_path": str(args.generated_rubric_path),
            "std_judge_model": args.std_judge_model,
            "min_std_valid_samples": args.min_std_valid_samples,
            "min_gen_valid_samples": args.min_gen_valid_samples,
            "generated_primary_comparison": "generated_fused_mean_score vs std_mean_score",
            "score_definition": "query-level relative score = total_trial_score / total_rubric_points",
        },
        "dataset_summary": {
            "num_std_pairs": len(std_scores),
            "num_generated_pairs": len(generated_scores),
            "num_common_pairs": len(common_keys),
        },
        "summary_generated_fused_vs_std": summarize_comparison(records, "generated_fused_mean_score"),
        "summary_generated_avg_over_samples_vs_std": summarize_comparison(records, "generated_avg_over_sample_means"),
        "records": records,
    }

    save_json(output, args.output_json)


if __name__ == "__main__":
    main()
