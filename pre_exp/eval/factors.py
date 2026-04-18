from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .config import FIGURE_DIR
from .data import get_query_text
from .scoring import _extract_score


TASK_FLAG_PATTERNS = {
    "format_constraint": [r"\bjson\b", r"\bformat\b", r"\btable\b", r"\bbullet\b", r"\bword[s]?\b", r"\bcharacter[s]?\b", r"\bsentence[s]?\b", r"\blist\b", r"\byaml\b"],
    "translation_or_rewrite": [r"translat", r"rewrite", r"paraphrase", r"summar", r"edit", r"polish"],
    "code_or_tool": [r"\bcode\b", r"\bpython\b", r"\bsql\b", r"\bfunction\b", r"\bdebug\b", r"\bprogram\b", r"\bscript\b"],
    "reasoning_math": [r"\bmath\b", r"calculate", r"compute", r"solve", r"equation", r"proof", r"reason", r"analysis", r"analyze", r"explain why"],
    "factual_grounding": [r"\bfact\b", r"accur", r"citation", r"cite", r"source", r"evidence", r"grounded", r"truthful"],
    "safety_refusal": [r"safe", r"safety", r"harmful", r"dangerous", r"illegal", r"refuse", r"decline", r"policy"],
}
PRIMARY_TASK_PRIORITY = ["translation_or_rewrite", "code_or_tool", "reasoning_math", "factual_grounding", "safety_refusal", "format_constraint"]


def _count_words(text):
    return len(re.findall(r"\w+", text)) if isinstance(text, str) else 0


def _safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return spearmanr(x, y).statistic


def _bootstrap_sem_per_entry(scores_matrix, n=1, B=50):
    scores_matrix = np.asarray(scores_matrix, dtype=float)
    if scores_matrix.ndim == 1:
        scores_matrix = scores_matrix.reshape(1, -1)

    sample_size = scores_matrix.shape[1]
    bootstrap_means = np.zeros((scores_matrix.shape[0], B))
    for b in range(B):
        indices = np.random.choice(sample_size, n, replace=True)
        bootstrap_means[:, b] = np.nanmean(scores_matrix[:, indices], axis=1)
    return np.nanstd(bootstrap_means, axis=1, ddof=1)


def infer_task_flags(prompt, rubrics):
    text = f"{prompt}\n{' '.join(item.get('description', '') for item in rubrics)}".lower()
    flags = {name: any(re.search(pattern, text) for pattern in patterns) for name, patterns in TASK_FLAG_PATTERNS.items()}
    primary_task = next((task for task in PRIMARY_TASK_PRIORITY if flags[task]), "other_general")
    return flags, primary_task


def build_factor_dataframe(data, ground_truth, responses, first_n, gen_models, judge_models, min_valid_samples=12, sem_n=1, B=50):
    records = []
    for dialog_id in data.keys():
        dialog = first_n[int(dialog_id)]
        prompt = get_query_text(dialog)
        rubrics = dialog["Rubrics"]
        rubric_points = np.array([item["points"] for item in rubrics], dtype=float)
        max_score = rubric_points.sum()
        flags, primary_task = infer_task_flags(prompt, rubrics)
        for gen_model in gen_models:
            response_text = responses[int(dialog_id)][gen_model]
            gt_scores = np.array([_extract_score(item) for item in ground_truth[dialog_id][gen_model]], dtype=float)
            gt_norm_score = gt_scores.sum() / max_score if max_score > 0 else np.nan
            gt_midness = 1 - min(abs(gt_norm_score - 0.5) / 0.5, 1.0) if not np.isnan(gt_norm_score) else np.nan
            for judge_model in judge_models:
                trials = data[dialog_id][gen_model][judge_model]
                valid_trials = []
                for trial in trials:
                    if len(trial) < len(rubrics):
                        continue
                    trial_scores = np.array([_extract_score(trial[k]) for k in range(len(rubrics))], dtype=float)
                    if not np.any(np.isnan(trial_scores)):
                        valid_trials.append(trial_scores)
                if len(valid_trials) < min_valid_samples:
                    continue
                valid_trials = np.array(valid_trials[:min_valid_samples], dtype=float)
                item_sem = _bootstrap_sem_per_entry(valid_trials.T, n=sem_n, B=B)
                item_rel_sem = np.divide(item_sem, rubric_points, out=np.full_like(item_sem, np.nan), where=rubric_points > 0)
                conv_scores = valid_trials.sum(axis=1)
                conv_sem = _bootstrap_sem_per_entry(conv_scores, n=sem_n, B=B)[0]
                mean_scores = valid_trials.mean(axis=0)
                judge_norm_score = mean_scores.sum() / max_score if max_score > 0 else np.nan
                record = {
                    "dialog_id": int(dialog_id),
                    "gen_model": gen_model,
                    "judge_model": judge_model,
                    "prompt_words": _count_words(prompt),
                    "prompt_chars": len(prompt),
                    "response_words": _count_words(response_text),
                    "response_chars": len(response_text),
                    "rubric_list_len": len(rubrics),
                    "total_points": max_score,
                    "gt_norm_score": gt_norm_score,
                    "gt_midness": gt_midness,
                    "mean_item_rel_sem": np.nanmean(item_rel_sem),
                    "conv_rel_sem": conv_sem / max_score if max_score > 0 else np.nan,
                    "rubric_alignment": _safe_spearman(mean_scores, gt_scores),
                    "abs_score_bias": abs(judge_norm_score - gt_norm_score),
                    "primary_task": primary_task,
                }
                record.update(flags)
                records.append(record)
    return pd.DataFrame(records)


def summarize_factor_correlations(df, features, targets):
    rows = []
    for judge_name, df_judge in [("ALL", df)] + [(judge, df[df["judge_model"] == judge]) for judge in sorted(df["judge_model"].unique())]:
        for feature in features:
            for target in targets:
                tmp = df_judge[[feature, target]].dropna()
                if len(tmp) >= 8:
                    rho = _safe_spearman(tmp[feature].values, tmp[target].values)
                    rows.append({"judge_model": judge_name, "feature": feature, "target": target, "rho": rho, "abs_rho": abs(rho), "n": len(tmp)})
    return pd.DataFrame(rows)


def summarize_by_quantile(df, feature, targets, q=4):
    tmp = df[[feature] + targets].dropna().copy()
    q = min(q, tmp[feature].nunique())
    if q < 2:
        return None
    tmp["bin"] = pd.qcut(tmp[feature], q=q, duplicates="drop")
    summary = tmp.groupby("bin")[targets].mean()
    summary["count"] = tmp.groupby("bin").size()
    return summary


def summarize_binary_flags(df, flag_columns):
    rows = []
    for flag in flag_columns:
        subset = df[df[flag]].copy()
        if len(subset):
            rows.append(
                {
                    "flag": flag,
                    "count": len(subset),
                    "sample_ratio": len(subset) / len(df),
                    "mean_item_rel_sem": subset["mean_item_rel_sem"].mean(),
                    "conv_rel_sem": subset["conv_rel_sem"].mean(),
                    "rubric_alignment": subset["rubric_alignment"].mean(),
                    "abs_score_bias": subset["abs_score_bias"].mean(),
                }
            )
    return pd.DataFrame(rows).sort_values("mean_item_rel_sem", ascending=False)


def plot_factor_analysis(df_factors):
    outcome_cols = ["mean_item_rel_sem", "conv_rel_sem", "rubric_alignment", "abs_score_bias"]
    numeric_features = ["rubric_list_len", "total_points", "prompt_words", "response_words", "gt_norm_score", "gt_midness"]
    corr_df = summarize_factor_correlations(df_factors, numeric_features, outcome_cols)
    print("整体数据上的 Spearman 相关性（按绝对值排序）:")
    print(corr_df[corr_df["judge_model"] == "ALL"].sort_values(["target", "abs_rho"], ascending=[True, False]))

    plot_features = [("rubric_list_len", "Rubric-list length"), ("response_words", "Response length (words)"), ("gt_midness", "GT midness (closer to middle score => larger)")]
    fig, axes = plt.subplots(len(plot_features), 2, figsize=(14, 4 * len(plot_features)))
    for row_idx, (feature, title) in enumerate(plot_features):
        summary = summarize_by_quantile(df_factors, feature, outcome_cols, q=4)
        if summary is None:
            continue
        labels = [str(idx) for idx in summary.index]
        axes[row_idx, 0].bar(labels, summary["mean_item_rel_sem"], color="#E15759", alpha=0.9, label="mean_item_rel_sem")
        axes[row_idx, 0].bar(labels, summary["conv_rel_sem"], color="#F28E2B", alpha=0.7, label="conv_rel_sem")
        axes[row_idx, 0].set_title(f"{title} vs Robustness")
        axes[row_idx, 0].legend()
        axes[row_idx, 1].bar(labels, summary["rubric_alignment"], color="#4E79A7", alpha=0.9, label="rubric_alignment")
        axes[row_idx, 1].bar(labels, summary["abs_score_bias"], color="#76B7B2", alpha=0.7, label="abs_score_bias")
        axes[row_idx, 1].set_title(f"{title} vs GT consistency")
        axes[row_idx, 1].legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "factor_numeric_effects.png", dpi=300, bbox_inches="tight")
    plt.show()

    flag_summary = summarize_binary_flags(df_factors, list(TASK_FLAG_PATTERNS.keys()))
    primary_task_summary = (
        df_factors.groupby("primary_task")
        .agg(sample_count=("dialog_id", "size"), mean_item_rel_sem=("mean_item_rel_sem", "mean"), conv_rel_sem=("conv_rel_sem", "mean"), rubric_alignment=("rubric_alignment", "mean"), abs_score_bias=("abs_score_bias", "mean"))
        .reset_index()
    )
    primary_task_summary = primary_task_summary[primary_task_summary["sample_count"] >= 5].sort_values("mean_item_rel_sem", ascending=False)
    print("不同任务特征标签下的平均表现:")
    print(flag_summary)
    print("主任务类型上的聚合结果（仅保留样本数 >= 5 的类别）:")
    print(primary_task_summary)

    gen_summary = df_factors.groupby(["judge_model", "gen_model"]).agg(
        {"mean_item_rel_sem": "mean", "conv_rel_sem": "mean", "rubric_alignment": "mean", "abs_score_bias": "mean"}
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.heatmap(gen_summary["mean_item_rel_sem"].unstack(), annot=True, fmt=".4f", cmap="YlOrRd", ax=axes[0, 0])
    sns.heatmap(gen_summary["rubric_alignment"].unstack(), annot=True, fmt=".4f", cmap="YlGnBu", ax=axes[0, 1])
    axes[0, 0].set_title("Gen Model impact on Item-level Instability")
    axes[0, 1].set_title("Gen Model impact on Rubric-level GT Alignment")
    axes[1, 0].barh(primary_task_summary["primary_task"], primary_task_summary["mean_item_rel_sem"], color="#E15759")
    axes[1, 1].barh(primary_task_summary["primary_task"], primary_task_summary["rubric_alignment"], color="#4E79A7")
    axes[1, 0].set_title("Primary task type vs Item-level instability")
    axes[1, 1].set_title("Primary task type vs GT alignment")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "factor_task_and_gen_model_effects.png", dpi=300, bbox_inches="tight")
    plt.show()

