from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import ROOT_DIR
from .data import get_query_text
from .scoring import _extract_score


def extract_case_study_data(data, first_n, gen_models, judge_models):
    records = []
    for gen_model in gen_models:
        for judge_model in judge_models:
            for i in data.keys():
                trials = data[i][gen_model][judge_model]
                rubrics = first_n[int(i)]["Rubrics"]
                num_rubrics = len(rubrics)
                for k in range(num_rubrics):
                    scores = []
                    for trial in trials:
                        if len(trial) < num_rubrics:
                            continue
                        scores.append(_extract_score(trial[k]))
                    valid_scores = [score for score in scores if not np.isnan(score)]
                    if len(valid_scores) >= 12:
                        final_scores = valid_scores[:12]
                        points = rubrics[k]["points"]
                        std = np.std(final_scores, ddof=1)
                        records.append(
                            {
                                "judge_model": judge_model,
                                "gen_model": gen_model,
                                "dialog_id": i,
                                "rubric_idx": k,
                                "points": points,
                                "rubric_list_len": num_rubrics,
                                "std": std,
                                "rel_std": std / points if points > 0 else 0,
                                "mean_score": np.mean(final_scores),
                            }
                        )
    return pd.DataFrame(records)


def analyze_top_unstable_items(df, top_k=20):
    for judge in df["judge_model"].unique():
        print("\n" + "=" * 50)
        print(f">>> Judge Model: {judge}")
        df_judge = df[df["judge_model"] == judge]
        top_unstable = df_judge.sort_values(by="rel_std", ascending=False).head(top_k)
        print(f"Top {top_k} 不稳定 Item 的平均 rubric-list 长度: {top_unstable['rubric_list_len'].mean():.2f}")
        print(f"该 Judge 所有 Item 的平均 rubric-list 长度: {df_judge['rubric_list_len'].mean():.2f}")
        for _, row in top_unstable.head(5).iterrows():
            print(f"Dialog: {row['dialog_id']}, Gen Model: {row['gen_model']}, Relative SEM: {row['rel_std']:.4f}")


def plot_dialogue_instability_overlap(df_case_study, top_n=30):
    dialogue_df = df_case_study.groupby(["dialog_id", "judge_model"]).agg(
        {"std": "mean", "rel_std": "mean", "rubric_list_len": "first"}
    ).reset_index()
    pivot_rel_sem = dialogue_df.pivot(index="dialog_id", columns="judge_model", values="rel_std")
    corr_matrix = pivot_rel_sem.corr(method="spearman")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation of Dialogue-level Relative SEM between Judges")
    plt.show()

    top_n_sets = {judge: set(pivot_rel_sem[judge].nlargest(top_n).index) for judge in pivot_rel_sem.columns}
    binary_df = pd.DataFrame(0, index=pivot_rel_sem.index, columns=pivot_rel_sem.columns)
    for judge, ids in top_n_sets.items():
        binary_df.loc[list(ids), judge] = 1
    plot_df = binary_df[binary_df.sum(axis=1) > 0].copy()
    plot_df["count"] = plot_df.sum(axis=1)
    plot_df = plot_df.sort_values(by=["count", "dialog_id"], ascending=[False, True]).drop(columns=["count"])
    plt.figure(figsize=(12, 10))
    sns.heatmap(plot_df, cmap="Blues", cbar=False, yticklabels=True, linewidths=0.5, linecolor="white")
    plt.title(f"Overlap of Top-{top_n} Unstable Dialogues (Sorted by Consensus)")
    plt.xlabel("Judge Model")
    plt.ylabel("Dialogue ID")
    plt.tight_layout()
    plt.show()


def inspect_unstable_dialog_prompts(first_n, dialog_ids=(16, 17, 24, 38)):
    for dialog_id in dialog_ids:
        print(f"Dialog ID {dialog_id} 的 Prompt:{get_query_text(first_n[int(dialog_id)])}")


def analyze_impact_of_gen_model(df_case_study):
    pivot_rel_std = df_case_study.pivot_table(index="judge_model", columns="gen_model", values="rel_std", aggfunc="mean")
    pivot_std = df_case_study.pivot_table(index="judge_model", columns="gen_model", values="std", aggfunc="mean")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(pivot_rel_std, annot=True, cmap="YlOrRd", fmt=".4f", ax=axes[0], vmax=pivot_rel_std.max().max())
    axes[0].set_title("Average Relative SEM")
    sns.heatmap(pivot_std, annot=True, cmap="YlOrRd", fmt=".4f", ax=axes[1], vmax=pivot_std.max().max())
    axes[1].set_title("Average Absolute SEM")
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "gen_model_impact_heatmaps_combined.png", dpi=300)
    plt.show()
