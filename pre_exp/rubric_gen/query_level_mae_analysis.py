from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DEFAULT_RUBRIC_PATH = BASE_DIR / "rubric_with_simplified_prompt.json"
DEFAULT_RESULT_PATH = BASE_DIR / "generated_rubric_judge_result.json"
DEFAULT_GROUND_TRUTH_PATH = ROOT_DIR / "ground_truth.json"
DEFAULT_OUTPUT_JSON = BASE_DIR / "query_level_mae_metrics.json"
DEFAULT_OUTPUT_PNG = ROOT_DIR / "figures" / "generated_rubric_query_level_mae_vs_n.png"
DEFAULT_BOOTSTRAP_B = 50
DEFAULT_SEED = 42


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_rubric_points(rubric_response: str) -> list[float]:
    try:
        rubric_list = json.loads(rubric_response.strip())
    except json.JSONDecodeError:
        return []
    if not isinstance(rubric_list, list):
        return []

    points = []
    for item in rubric_list:
        if not isinstance(item, dict):
            return []
        criterion = item.get("criterion")
        if not isinstance(criterion, str) or not criterion.strip():
            return []
        point = item.get("points", 1)
        try:
            point = float(point)
        except (TypeError, ValueError):
            point = 1.0
        points.append(point if point > 0 else 1.0)
    return points


def extract_score(rubric: dict[str, Any]) -> float:
    score = rubric.get("score")
    if score is not None:
        try:
            score = float(score)
            if not np.isnan(score):
                return score
        except (TypeError, ValueError):
            pass

    is_met = rubric.get("is_met")
    weight = rubric.get("weight")
    if is_met is not None and weight is not None:
        try:
            return float(weight) if str(is_met).lower() == "true" else 0.0
        except (TypeError, ValueError):
            pass
    return np.nan


def build_rubric_lookup(rubric_data: list[dict[str, Any]]) -> dict[tuple[int, int], list[float]]:
    rubric_lookup = {}
    for item in rubric_data:
        q_idx = int(item["question_index"])
        for sample_idx, rubric_response in enumerate(item.get("rubric_responses", []), start=1):
            rubric_lookup[(q_idx, sample_idx)] = parse_rubric_points(rubric_response)
    return rubric_lookup


def build_gt_lookup(ground_truth: dict[str, Any]) -> dict[tuple[int, str], float]:
    gt_lookup = {}
    for question_key, question_data in ground_truth.items():
        q_idx = int(question_key)
        for gen_model, rubric_scores in question_data.items():
            scores = np.array([extract_score(item) for item in rubric_scores], dtype=float)
            points = np.array([float(item.get("weight", 1)) for item in rubric_scores], dtype=float)
            if scores.size == 0 or np.any(np.isnan(scores)) or points.sum() <= 0:
                continue
            gt_lookup[(q_idx, gen_model)] = float(scores.sum() / points.sum())
    return gt_lookup


def collect_buckets(
    judge_result: dict[str, Any],
    rubric_lookup: dict[tuple[int, int], list[float]],
    gt_lookup: dict[tuple[int, str], float],
) -> tuple[dict[tuple[int, str, int], dict[str, Any]], dict[str, Any]]:
    query_buckets: dict[tuple[int, str, int], dict[str, Any]] = {}

    metadata = {
        "num_questions_in_result": len(judge_result),
        "num_gt_pairs": len(gt_lookup),
    }

    for question_key, question_data in judge_result.items():
        q_idx = int(question_key)
        for gen_model, rubric_samples in question_data.items():
            gt_relative_score = gt_lookup.get((q_idx, gen_model))
            if gt_relative_score is None or np.isnan(gt_relative_score):
                continue

            for sample_key, trials in rubric_samples.items():
                sample_idx = int(sample_key)
                rubric_points = rubric_lookup.get((q_idx, sample_idx), [])
                if not rubric_points:
                    continue

                rubric_points = np.array(rubric_points, dtype=float)
                total_points = float(rubric_points.sum())
                if total_points <= 0:
                    continue

                for trial in trials:
                    if len(trial) < len(rubric_points):
                        continue
                    scores = np.array([extract_score(trial[i]) for i in range(len(rubric_points))], dtype=float)
                    if np.any(np.isnan(scores)):
                        continue

                    relative_score = float(scores.sum() / total_points)
                    query_key = (q_idx, gen_model, sample_idx)
                    query_buckets.setdefault(
                        query_key,
                        {"gt_relative_score": float(gt_relative_score), "relative_scores": []},
                    )["relative_scores"].append(relative_score)

    return query_buckets, metadata


def bootstrap_mae(scores: list[float], gt_score: float, n: int, B: int, rng: np.random.Generator) -> float:
    scores_array = np.array(scores, dtype=float)
    if scores_array.size < n or n <= 0:
        return float("nan")

    abs_errors = np.zeros(B, dtype=float)
    for b in range(B):
        indices = rng.choice(scores_array.size, size=n, replace=True)
        sampled_mean = float(np.mean(scores_array[indices]))
        abs_errors[b] = abs(sampled_mean - gt_score)
    return float(np.mean(abs_errors))


def average_independent_mae(
    query_buckets: dict[tuple[int, str, int], dict[str, Any]],
    n_values: list[int],
    B: int,
    seed: int,
    reducer_name: str,
    reducer: Callable[[np.ndarray], float],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    query_groups: dict[tuple[int, str], list[tuple[tuple[int, str, int], dict[str, Any]]]] = {}
    for bucket_key, bucket in query_buckets.items():
        query_groups.setdefault((bucket_key[0], bucket_key[1]), []).append((bucket_key, bucket))

    mean_mae = []
    group_counts_by_n = []
    per_n_group_details = []

    for n in n_values:
        group_maes = []
        group_detail = []
        for group_key, group_items in query_groups.items():
            bucket_maes = []
            bucket_keys = []
            for bucket_key, bucket in group_items:
                bucket_mae = bootstrap_mae(
                    bucket["relative_scores"],
                    bucket["gt_relative_score"],
                    n=n,
                    B=B,
                    rng=rng,
                )
                if np.isnan(bucket_mae):
                    continue
                bucket_maes.append(bucket_mae)
                bucket_keys.append(list(bucket_key))

            if not bucket_maes:
                continue

            bucket_maes_array = np.array(bucket_maes, dtype=float)
            reduced_mae = float(reducer(bucket_maes_array))
            group_maes.append(reduced_mae)
            group_detail.append(
                {
                    "group_key": list(group_key),
                    "bucket_keys": bucket_keys,
                    "bucket_maes": bucket_maes,
                    "aggregated_mae": reduced_mae,
                    "aggregation": reducer_name,
                }
            )

        mean_mae.append(float(np.mean(group_maes)) if group_maes else float("nan"))
        group_counts_by_n.append(len(group_maes))
        per_n_group_details.append({"n": n, "groups": group_detail})

    return {
        "aggregation": reducer_name,
        "n_values": n_values,
        "mean_mae": mean_mae,
        "group_counts_by_n": group_counts_by_n,
        "num_groups": len(query_groups),
        "per_group_summary": [
            {
                "group_key": list(group_key),
                "num_rubric_samples": len(group_items),
                "bucket_keys": [list(bucket_key) for bucket_key, _ in group_items],
                "num_available_samples_per_bucket": [len(bucket["relative_scores"]) for _, bucket in group_items],
                "gt_relative_score": float(group_items[0][1]["gt_relative_score"]),
            }
            for group_key, group_items in query_groups.items()
        ],
        "per_n_group_details": per_n_group_details,
    }


def plot_curves(
    curves: list[dict[str, Any]],
    output_png: Path,
    title: str,
    ylabel: str,
) -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "PingFang SC", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8.5, 5.5))
    for curve in curves:
        plt.plot(
            curve["x"],
            curve["y"],
            marker=curve.get("marker", "o"),
            linewidth=2,
            label=curve.get("label"),
        )

    plt.title(title)
    plt.xlabel("Number of Sampling Iterations (n)")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    if any(curve.get("label") for curve in curves):
        plt.legend()
    for curve in curves:
        for xi, yi in zip(curve["x"], curve["y"]):
            plt.annotate(
                f"{yi:.4f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, curve.get("y_offset", 8)),
                ha="center",
                fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute query-level independent-bucket MAE for generated rubrics.")
    parser.add_argument("--rubric-path", type=Path, default=DEFAULT_RUBRIC_PATH)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--ground-truth-path", type=Path, default=DEFAULT_GROUND_TRUTH_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--bootstrap-b", type=int, default=DEFAULT_BOOTSTRAP_B)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    rubric_lookup = build_rubric_lookup(load_json(args.rubric_path))
    gt_lookup = build_gt_lookup(load_json(args.ground_truth_path))
    query_buckets, metadata = collect_buckets(
        load_json(args.result_path),
        rubric_lookup,
        gt_lookup,
    )

    query_max_n = max((len(bucket["relative_scores"]) for bucket in query_buckets.values()), default=0)
    n_values = list(range(1, query_max_n + 1))

    avg_result = average_independent_mae(
        query_buckets,
        n_values,
        B=args.bootstrap_b,
        seed=args.seed,
        reducer_name="avg",
        reducer=np.mean,
    )
    min_result = average_independent_mae(
        query_buckets,
        n_values,
        B=args.bootstrap_b,
        seed=args.seed + 1,
        reducer_name="min",
        reducer=np.min,
    )
    max_result = average_independent_mae(
        query_buckets,
        n_values,
        B=args.bootstrap_b,
        seed=args.seed + 2,
        reducer_name="max",
        reducer=np.max,
    )

    metrics = {
        "config": {
            "rubric_path": str(args.rubric_path),
            "result_path": str(args.result_path),
            "ground_truth_path": str(args.ground_truth_path),
            "bootstrap_B": args.bootstrap_b,
            "seed": args.seed,
            "query_independent_n_range": [1, query_max_n],
        },
        "dataset_summary": {
            **metadata,
            "num_query_buckets": len(query_buckets),
        },
        "query_level_rubric_sample_independent_mae_avg": avg_result,
        "query_level_rubric_sample_independent_mae_min": min_result,
        "query_level_rubric_sample_independent_mae_max": max_result,
    }

    save_json(metrics, args.output_json)
    plot_curves(
        [
            {
                "x": avg_result["n_values"],
                "y": avg_result["mean_mae"],
                "label": "AVG",
                "marker": "o",
                "y_offset": 8,
            },
            {
                "x": min_result["n_values"],
                "y": min_result["mean_mae"],
                "label": "MIN",
                "marker": "s",
                "y_offset": -14,
            },
            {
                "x": max_result["n_values"],
                "y": max_result["mean_mae"],
                "label": "MAX",
                "marker": "^",
                "y_offset": 10,
            },
        ],
        args.output_png,
        title="Query-level Independent Bucket MAE vs n",
        ylabel="Average Query-level MAE",
    )


if __name__ == "__main__":
    main()
