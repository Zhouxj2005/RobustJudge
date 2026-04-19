from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DEFAULT_RUBRIC_PATH = BASE_DIR / "rubric_with_simplified_prompt.json"
DEFAULT_RESULT_PATH = BASE_DIR / "generated_rubric_judge_result.json"
DEFAULT_GROUND_TRUTH_PATH = ROOT_DIR / "ground_truth.json"
DEFAULT_OUTPUT_JSON = BASE_DIR / "query_level_spearman_metrics.json"
DEFAULT_OUTPUT_PNG = ROOT_DIR / "figures" / "generated_rubric_query_level_spearman_vs_n.png"
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
) -> tuple[dict[tuple[int, str, int], dict[str, Any]], dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    query_buckets: dict[tuple[int, str, int], dict[str, Any]] = {}
    fused_query_buckets: dict[tuple[int, str], dict[str, Any]] = {}

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
                    fused_key = (q_idx, gen_model)

                    query_buckets.setdefault(
                        query_key,
                        {"gt_relative_score": float(gt_relative_score), "relative_scores": []},
                    )["relative_scores"].append(relative_score)

                    fused_query_buckets.setdefault(
                        fused_key,
                        {"gt_relative_score": float(gt_relative_score), "relative_scores": []},
                    )["relative_scores"].append(relative_score)

    return query_buckets, fused_query_buckets, metadata


def bootstrap_mean(scores: list[float], n: int, rng: np.random.Generator) -> float:
    scores = np.array(scores, dtype=float)
    indices = rng.choice(scores.size, size=n, replace=True)
    return float(np.mean(scores[indices]))


def average_bucket_spearman(
    buckets: dict[tuple[Any, ...], dict[str, Any]],
    n_values: list[int],
    B: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    mean_spearman = []
    bucket_counts_by_n = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in n_values:
            valid_bucket_items = [
                (bucket_key, bucket)
                for bucket_key, bucket in buckets.items()
                if len(bucket["relative_scores"]) >= n
            ]
            bucket_counts_by_n.append(len(valid_bucket_items))
            if len(valid_bucket_items) < 2:
                mean_spearman.append(float("nan"))
                continue

            gt_values = np.array([bucket["gt_relative_score"] for _, bucket in valid_bucket_items], dtype=float)
            bootstrap_rhos = []
            for _ in range(B):
                sampled_scores = np.array(
                    [bootstrap_mean(bucket["relative_scores"], n, rng) for _, bucket in valid_bucket_items],
                    dtype=float,
                )
                rho = spearmanr(sampled_scores, gt_values).statistic
                if not np.isnan(rho):
                    bootstrap_rhos.append(float(rho))
            mean_spearman.append(float(np.nanmean(bootstrap_rhos)) if bootstrap_rhos else float("nan"))

    return {
        "n_values": n_values,
        "mean_spearman": mean_spearman,
        "bucket_counts_by_n": bucket_counts_by_n,
        "num_buckets": len(buckets),
        "per_bucket_summary": [
            {
                "bucket_key": list(bucket_key),
                "num_available_samples": len(bucket["relative_scores"]),
                "gt_relative_score": float(bucket["gt_relative_score"]),
            }
            for bucket_key, bucket in buckets.items()
        ],
    }


def average_independent_spearman(
    query_buckets: dict[tuple[int, str, int], dict[str, Any]],
    n_values: list[int],
    B: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    query_groups: dict[tuple[int, str], list[tuple[tuple[int, str, int], dict[str, Any]]]] = {}
    for bucket_key, bucket in query_buckets.items():
        query_groups.setdefault((bucket_key[0], bucket_key[1]), []).append((bucket_key, bucket))

    mean_spearman = []
    group_counts_by_n = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in n_values:
            valid_groups = []
            for group_key, group_items in query_groups.items():
                valid_items = [(bucket_key, bucket) for bucket_key, bucket in group_items if len(bucket["relative_scores"]) >= n]
                if valid_items:
                    valid_groups.append((group_key, valid_items))
            group_counts_by_n.append(len(valid_groups))
            if len(valid_groups) < 2:
                mean_spearman.append(float("nan"))
                continue

            gt_values = np.array([group_items[0][1]["gt_relative_score"] for _, group_items in valid_groups], dtype=float)
            bootstrap_rhos = []
            for _ in range(B):
                sampled_scores = []
                for _, group_items in valid_groups:
                    _, bucket = group_items[rng.integers(len(group_items))]
                    sampled_scores.append(bootstrap_mean(bucket["relative_scores"], n, rng))
                rho = spearmanr(np.array(sampled_scores, dtype=float), gt_values).statistic
                if not np.isnan(rho):
                    bootstrap_rhos.append(float(rho))
            mean_spearman.append(float(np.nanmean(bootstrap_rhos)) if bootstrap_rhos else float("nan"))

    return {
        "n_values": n_values,
        "mean_spearman": mean_spearman,
        "group_counts_by_n": group_counts_by_n,
        "num_groups": len(query_groups),
        "per_group_summary": [
            {
                "group_key": list(group_key),
                "num_rubric_samples": len(group_items),
                "num_available_samples_per_rubric": [len(bucket["relative_scores"]) for _, bucket in group_items],
                "gt_relative_score": float(group_items[0][1]["gt_relative_score"]),
            }
            for group_key, group_items in query_groups.items()
        ],
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
    parser = argparse.ArgumentParser(description="Compute query-level Spearman for generated rubrics.")
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
    query_buckets, fused_query_buckets, metadata = collect_buckets(
        load_json(args.result_path),
        rubric_lookup,
        gt_lookup,
    )

    query_max_n = max((len(bucket["relative_scores"]) for bucket in query_buckets.values()), default=0)
    fused_query_max_n = max((len(bucket["relative_scores"]) for bucket in fused_query_buckets.values()), default=0)
    query_common_max_n = min(query_max_n, fused_query_max_n)

    query_independent_result = average_independent_spearman(
        query_buckets,
        list(range(1, query_common_max_n + 1)),
        B=args.bootstrap_b,
        seed=args.seed,
    )
    query_fused_result = average_bucket_spearman(
        fused_query_buckets,
        list(range(1, query_common_max_n + 1)),
        B=args.bootstrap_b,
        seed=args.seed + 1,
    )

    metrics = {
        "config": {
            "rubric_path": str(args.rubric_path),
            "result_path": str(args.result_path),
            "ground_truth_path": str(args.ground_truth_path),
            "bootstrap_B": args.bootstrap_b,
            "seed": args.seed,
            "query_common_n_range": [1, query_common_max_n],
            "query_fused_full_n_range": [1, fused_query_max_n],
        },
        "dataset_summary": {
            **metadata,
            "num_query_buckets": len(query_buckets),
            "num_fused_query_buckets": len(fused_query_buckets),
        },
        "query_level_rubric_sample_independent": query_independent_result,
        "query_level_rubric_sample_fused_common_n": query_fused_result,
    }

    save_json(metrics, args.output_json)
    plot_curves(
        [
            {
                "x": query_independent_result["n_values"],
                "y": query_independent_result["mean_spearman"],
                "label": "rubric-sample 独立",
                "marker": "o",
                "y_offset": 8,
            },
            {
                "x": query_fused_result["n_values"],
                "y": query_fused_result["mean_spearman"],
                "label": "rubric-sample 融合",
                "marker": "s",
                "y_offset": -14,
            },
        ],
        args.output_png,
        title="Query-level Spearman 随 n 变化",
        ylabel="Average Query-level Spearman",
    )


if __name__ == "__main__":
    main()
