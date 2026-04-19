from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DEFAULT_RUBRIC_PATH = BASE_DIR / "rubric_with_simplified_prompt.json"
DEFAULT_RESULT_PATH = BASE_DIR / "generated_rubric_judge_result.json"
DEFAULT_OUTPUT_JSON = BASE_DIR / "query_level_r_sem_metrics.json"
DEFAULT_OUTPUT_PNG = ROOT_DIR / "figures" / "generated_rubric_query_level_r_sem_vs_n.png"
DEFAULT_CRITERIA_OUTPUT_PNG = ROOT_DIR / "figures" / "generated_rubric_criteria_level_r_sem_vs_n.png"
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


def collect_buckets(
    judge_result: dict[str, Any],
    rubric_lookup: dict[tuple[int, int], list[float]],
) -> tuple[dict[tuple[int, str, int], list[float]], dict[tuple[int, str], list[float]], dict[tuple[int, str, int, int], list[float]], dict[str, Any]]:
    query_buckets: dict[tuple[int, str, int], list[float]] = {}
    fused_query_buckets: dict[tuple[int, str], list[float]] = {}
    rubric_buckets: dict[tuple[int, str, int, int], list[float]] = {}

    metadata = {
        "num_questions": len(judge_result),
        "gen_models": sorted({gen_model for question in judge_result.values() for gen_model in question.keys()}),
    }

    for question_key, question_data in judge_result.items():
        q_idx = int(question_key)
        for gen_model, rubric_samples in question_data.items():
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

                    normalized_query_score = float(scores.sum() / total_points)
                    query_buckets.setdefault((q_idx, gen_model, sample_idx), []).append(normalized_query_score)
                    fused_query_buckets.setdefault((q_idx, gen_model), []).append(normalized_query_score)

                    normalized_rubric_scores = scores / rubric_points
                    for rubric_idx, rubric_score in enumerate(normalized_rubric_scores, start=1):
                        rubric_buckets.setdefault((q_idx, gen_model, sample_idx, rubric_idx), []).append(float(rubric_score))

    return query_buckets, fused_query_buckets, rubric_buckets, metadata


def bootstrap_sem(scores: list[float], n: int, B: int, rng: np.random.Generator) -> float:
    scores = np.array(scores, dtype=float)
    if scores.size < n or n <= 0:
        return float("nan")
    bootstrap_means = np.zeros(B, dtype=float)
    for b in range(B):
        indices = rng.choice(scores.size, size=n, replace=True)
        bootstrap_means[b] = float(np.mean(scores[indices]))
    return float(np.std(bootstrap_means, ddof=1))


def average_bucket_sem(
    buckets: dict[tuple[Any, ...], list[float]],
    n_values: list[int],
    B: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    mean_r_sem = []
    bucket_counts_by_n = []

    for n in n_values:
        sems = []
        for scores in buckets.values():
            sem = bootstrap_sem(scores, n, B, rng)
            if not np.isnan(sem):
                sems.append(sem)
        mean_r_sem.append(float(np.mean(sems)) if sems else float("nan"))
        bucket_counts_by_n.append(len(sems))

    return {
        "n_values": n_values,
        "mean_r_sem": mean_r_sem,
        "bucket_counts_by_n": bucket_counts_by_n,
        "num_buckets": len(buckets),
        "per_bucket_summary": [
            {"bucket_key": list(bucket_key), "num_available_samples": len(scores)}
            for bucket_key, scores in buckets.items()
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
    parser = argparse.ArgumentParser(description="Compute query-level and rubric-level R-SEM for generated rubrics.")
    parser.add_argument("--rubric-path", type=Path, default=DEFAULT_RUBRIC_PATH)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--criteria-output-png", type=Path, default=DEFAULT_CRITERIA_OUTPUT_PNG)
    parser.add_argument("--bootstrap-b", type=int, default=DEFAULT_BOOTSTRAP_B)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    rubric_lookup = build_rubric_lookup(load_json(args.rubric_path))
    query_buckets, fused_query_buckets, rubric_buckets, metadata = collect_buckets(
        load_json(args.result_path),
        rubric_lookup,
    )

    query_max_n = max((len(scores) for scores in query_buckets.values()), default=0)
    fused_query_max_n = max((len(scores) for scores in fused_query_buckets.values()), default=0)
    rubric_max_n = max((len(scores) for scores in rubric_buckets.values()), default=0)
    query_common_max_n = min(query_max_n, fused_query_max_n)

    query_independent_result = average_bucket_sem(
        query_buckets,
        list(range(1, query_common_max_n + 1)),
        B=args.bootstrap_b,
        seed=args.seed,
    )
    query_fused_result = average_bucket_sem(
        fused_query_buckets,
        list(range(1, query_common_max_n + 1)),
        B=args.bootstrap_b,
        seed=args.seed + 1,
    )
    rubric_result = average_bucket_sem(
        rubric_buckets,
        list(range(1, rubric_max_n + 1)),
        B=args.bootstrap_b,
        seed=args.seed + 2,
    )

    metrics = {
        "config": {
            "rubric_path": str(args.rubric_path),
            "result_path": str(args.result_path),
            "bootstrap_B": args.bootstrap_b,
            "seed": args.seed,
            "query_common_n_range": [1, query_common_max_n],
            "rubric_level_n_range": [1, rubric_max_n],
        },
        "dataset_summary": {
            **metadata,
            "num_query_buckets": len(query_buckets),
            "num_fused_query_buckets": len(fused_query_buckets),
            "num_rubric_buckets": len(rubric_buckets),
        },
        "query_level_rubric_sample_independent": query_independent_result,
        "query_level_rubric_sample_fused": query_fused_result,
        "rubric_level_independent": rubric_result,
    }

    save_json(metrics, args.output_json)
    plot_curves(
        [
            {
                "x": query_independent_result["n_values"],
                "y": query_independent_result["mean_r_sem"],
                "label": "rubric-sample 独立",
                "marker": "o",
                "y_offset": 8,
            },
            {
                "x": query_fused_result["n_values"],
                "y": query_fused_result["mean_r_sem"],
                "label": "rubric-sample 融合",
                "marker": "s",
                "y_offset": -14,
            },
        ],
        args.output_png,
        title="Query-level R-SEM 随 n 变化",
        ylabel="Average Query-level R-SEM",
    )
    plot_curves(
        [
            {
                "x": rubric_result["n_values"],
                "y": rubric_result["mean_r_sem"],
                "marker": "o",
                "y_offset": 8,
            }
        ],
        args.criteria_output_png,
        title="Rubric-level R-SEM 随 n 变化",
        ylabel="Average Rubric-level R-SEM",
    )


if __name__ == "__main__":
    main()
