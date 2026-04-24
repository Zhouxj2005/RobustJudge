from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DEFAULT_MATCH_PATH = BASE_DIR / "rubric_matrix_list_match.json"
DEFAULT_RESULT_PATH = BASE_DIR / "generated_rubric_judge_result.json"
DEFAULT_OUTPUT_JSON = BASE_DIR / "rubric_context_consistency_metrics.json"
DEFAULT_OUTPUT_PNG = ROOT_DIR / "figures" / "generated_rubric_context_consistency_t_obs.png"
DEFAULT_PERMUTATIONS = 200
DEFAULT_SEED = 42
DEFAULT_ALPHA = 0.05
DEFAULT_TOP_K = 30
DEFAULT_MIN_CONTEXTS = 2
DEFAULT_REQUIRED_SAMPLES_PER_CONTEXT = 8


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_score(rubric: dict[str, Any]) -> float:
    score = rubric.get("score")
    if score is not None:
        try:
            score = float(score)
            if not np.isnan(score) and score >= 0:
                return score
        except (TypeError, ValueError):
            pass

    is_met = rubric.get("is_met")
    weight = rubric.get("weight")
    if is_met is not None and weight is not None:
        try:
            weight = float(weight)
            if weight < 0:
                return np.nan
            return weight if str(is_met).lower() == "true" else 0.0
        except (TypeError, ValueError):
            pass
    return np.nan


def normalize_rubric_score(rubric: dict[str, Any]) -> float:
    raw_score = extract_score(rubric)
    if np.isnan(raw_score):
        return np.nan

    weight = rubric.get("weight", 1)
    try:
        weight = float(weight)
    except (TypeError, ValueError):
        weight = 1.0
    if weight <= 0:
        return np.nan
    return float(raw_score / weight)


def average_pairwise_wasserstein(score_lists: list[np.ndarray]) -> float:
    if len(score_lists) < 2:
        return float("nan")

    distances = [
        wasserstein_distance(scores_a, scores_b)
        for scores_a, scores_b in combinations(score_lists, 2)
    ]
    return float(np.mean(distances)) if distances else float("nan")


def permutation_test_t_obs(
    score_lists: list[np.ndarray],
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    t_obs = average_pairwise_wasserstein(score_lists)
    if np.isnan(t_obs):
        return float("nan"), float("nan")

    pooled = np.concatenate(score_lists)
    sizes = [len(scores) for scores in score_lists]
    count = 0

    for _ in range(permutations):
        shuffled = rng.permutation(pooled)
        permuted_lists = []
        start = 0
        for size in sizes:
            permuted_lists.append(shuffled[start : start + size])
            start += size
        t_perm = average_pairwise_wasserstein(permuted_lists)
        if t_perm >= t_obs:
            count += 1

    p_value = float((count + 1) / (permutations + 1))
    return t_obs, p_value


def collect_context_scores(
    match_data: list[dict[str, Any]],
    judge_result: dict[str, Any],
) -> tuple[dict[tuple[int, str, int], dict[int, list[float]]], dict[tuple[int, str, int], str], dict[str, Any]]:
    grouped_scores: dict[tuple[int, str, int], dict[int, list[float]]] = {}
    rubric_text_lookup: dict[tuple[int, str, int], str] = {}
    stats = {
        "num_questions_in_match": len(match_data),
        "num_questions_in_result": len(judge_result),
        "num_trials_seen": 0,
        "num_trial_unique_context_scores": 0,
        "num_missing_sample_mappings": 0,
    }

    for item in match_data:
        q_idx = int(item["question_index"])
        sample_match_indices = item.get("sample_match_indices")
        unique_rubrics = item.get("unique_rubrics", [])
        if not isinstance(sample_match_indices, list):
            raise ValueError(
                "match file is missing 'sample_match_indices'. "
                "Please regenerate rubric_matrix_list_match.json with the updated get_matrix.py."
            )

        question_result = judge_result.get(str(q_idx), {})
        if not isinstance(question_result, dict):
            continue

        for gen_model, rubric_samples in question_result.items():
            if not isinstance(rubric_samples, dict):
                continue

            for sample_key, trials in rubric_samples.items():
                sample_idx = int(sample_key)
                if sample_idx < 1 or sample_idx > len(sample_match_indices):
                    stats["num_missing_sample_mappings"] += 1
                    continue

                local_to_unique = sample_match_indices[sample_idx - 1]
                for trial in trials:
                    stats["num_trials_seen"] += 1
                    unique_scores_in_trial: dict[int, list[float]] = {}

                    for local_rubric_idx, unique_rubric_idx in enumerate(local_to_unique):
                        if local_rubric_idx >= len(trial):
                            continue
                        rubric = trial[local_rubric_idx]
                        normalized_score = normalize_rubric_score(rubric)
                        if np.isnan(normalized_score):
                            continue
                        unique_scores_in_trial.setdefault(int(unique_rubric_idx), []).append(normalized_score)

                    for unique_rubric_idx, scores in unique_scores_in_trial.items():
                        stats["num_trial_unique_context_scores"] += 1
                        grouped_scores.setdefault((q_idx, gen_model, unique_rubric_idx), {}).setdefault(sample_idx, []).append(
                            float(np.mean(scores))
                        )
                        if unique_rubric_idx < len(unique_rubrics):
                            rubric_text_lookup[(q_idx, gen_model, unique_rubric_idx)] = str(
                                unique_rubrics[unique_rubric_idx].get("criterion", "")
                            )

    return grouped_scores, rubric_text_lookup, stats


def analyze_groups(
    grouped_scores: dict[tuple[int, str, int], dict[int, list[float]]],
    rubric_text_lookup: dict[tuple[int, str, int], str],
    permutations: int,
    seed: int,
    alpha: float,
    min_contexts: int,
    required_samples_per_context: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []

    for group_key, context_scores in grouped_scores.items():
        filtered_contexts = {
            sample_idx: np.array(scores, dtype=float)
            for sample_idx, scores in context_scores.items()
            if len(scores) == required_samples_per_context
        }
        if len(filtered_contexts) < min_contexts:
            continue

        ordered_context_items = sorted(filtered_contexts.items())
        score_lists = [scores for _, scores in ordered_context_items]
        t_obs, p_value = permutation_test_t_obs(score_lists, permutations=permutations, rng=rng)
        if np.isnan(t_obs):
            continue

        q_idx, gen_model, unique_rubric_idx = group_key
        context_details = []
        context_means = []
        for sample_idx, scores in ordered_context_items:
            mean_score = float(np.mean(scores))
            context_means.append(mean_score)
            context_details.append(
                {
                    "sample_idx": sample_idx,
                    "num_scores": int(scores.size),
                    "mean_score": mean_score,
                    "var_score": float(np.var(scores, ddof=1)) if scores.size > 1 else 0.0,
                }
            )

        records.append(
            {
                "question_index": q_idx,
                "gen_model": gen_model,
                "unique_rubric_index": unique_rubric_idx,
                "unique_rubric_criterion": rubric_text_lookup.get(group_key, ""),
                "num_contexts": len(ordered_context_items),
                "total_scores": int(sum(scores.size for _, scores in ordered_context_items)),
                "t_obs": float(t_obs),
                "p_value": float(p_value),
                "reject_h0": bool(p_value < alpha),
                "context_mean_range": float(max(context_means) - min(context_means)),
                "context_details": context_details,
            }
        )

    records.sort(key=lambda item: (-item["t_obs"], item["question_index"], item["gen_model"], item["unique_rubric_index"]))
    return records


def plot_t_obs(records: list[dict[str, Any]], output_png: Path, top_k: int) -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "PingFang SC", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No valid (q, g, u) groups", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close()
        return

    t_obs_values = np.array([record["t_obs"] for record in records], dtype=float)
    sorted_indices = np.argsort(t_obs_values)
    sorted_t_obs = t_obs_values[sorted_indices]
    sorted_reject = np.array([records[idx]["reject_h0"] for idx in sorted_indices], dtype=bool)

    top_records = records[:top_k]
    top_labels = [
        f"q{record['question_index']}-{record['gen_model']}-u{record['unique_rubric_index']}"
        for record in top_records
    ]
    top_values = [record["t_obs"] for record in top_records]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = np.where(sorted_reject, "#d55e00", "#0072b2")
    axes[0].scatter(np.arange(len(sorted_t_obs)), sorted_t_obs, c=colors, s=18, alpha=0.85)
    axes[0].set_title("Sorted T_obs Scatter")
    axes[0].set_xlabel("Sorted (q, g, u) Index")
    axes[0].set_ylabel("T_obs")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    bar_colors = ["#d55e00" if record["reject_h0"] else "#0072b2" for record in top_records]
    axes[1].bar(np.arange(len(top_records)), top_values, color=bar_colors, alpha=0.9)
    axes[1].set_title(f"Top {len(top_records)} T_obs")
    axes[1].set_xlabel("(q, g, u)")
    axes[1].set_ylabel("T_obs")
    axes[1].set_xticks(np.arange(len(top_records)))
    axes[1].set_xticklabels(top_labels, rotation=75, ha="right", fontsize=8)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether the judge-score distribution of the same unique rubric is consistent across rubric-list contexts."
    )
    parser.add_argument("--match-path", type=Path, default=DEFAULT_MATCH_PATH)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-contexts", type=int, default=DEFAULT_MIN_CONTEXTS)
    parser.add_argument("--required-samples-per-context", type=int, default=DEFAULT_REQUIRED_SAMPLES_PER_CONTEXT)
    args = parser.parse_args()

    grouped_scores, rubric_text_lookup, collection_stats = collect_context_scores(
        load_json(args.match_path),
        load_json(args.result_path),
    )
    records = analyze_groups(
        grouped_scores=grouped_scores,
        rubric_text_lookup=rubric_text_lookup,
        permutations=args.permutations,
        seed=args.seed,
        alpha=args.alpha,
        min_contexts=args.min_contexts,
        required_samples_per_context=args.required_samples_per_context,
    )

    metrics = {
        "config": {
            "match_path": str(args.match_path),
            "result_path": str(args.result_path),
            "permutations": args.permutations,
            "seed": args.seed,
            "alpha": args.alpha,
            "top_k": args.top_k,
            "min_contexts": args.min_contexts,
            "required_samples_per_context": args.required_samples_per_context,
            "t_obs_definition": "average pairwise Wasserstein distance across contexts",
        },
        "dataset_summary": {
            **collection_stats,
            "num_grouped_keys_before_filtering": len(grouped_scores),
            "num_valid_records": len(records),
            "num_rejected_records": int(sum(record["reject_h0"] for record in records)),
        },
        "records": records,
    }

    save_json(metrics, args.output_json)
    plot_t_obs(records, args.output_png, args.top_k)


if __name__ == "__main__":
    main()
