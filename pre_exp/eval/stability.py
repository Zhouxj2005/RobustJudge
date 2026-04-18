from __future__ import annotations

from collections import defaultdict

import numpy as np

from .scoring import _extract_score


def extract_rubric_scores(data, first_n, gen_model, judge_model, min_valid_samples=12):
    rubric_data = []
    rubric_total_scores = []
    for i in data.keys():
        trials = data[i][gen_model][judge_model]
        num_rubrics = len(first_n[int(i)]["Rubrics"])
        for k in range(num_rubrics):
            scores = []
            for trial in trials:
                if len(trial) < num_rubrics:
                    continue
                scores.append(_extract_score(trial[k]))
            if np.sum(~np.isnan(scores)) >= min_valid_samples:
                scores = [score for score in scores if not np.isnan(score)]
                rubric_data.append(scores[:min_valid_samples])
                rubric_total_scores.append(first_n[int(i)]["Rubrics"][k]["points"])
    return np.array(rubric_data), np.array(rubric_total_scores)


def extract_conversation_scores(data, first_n, gen_model, judge_model, min_valid_samples=12):
    valid_conv_data = []
    conversation_total_scores = []
    for i in data.keys():
        trials = data[i][gen_model][judge_model]
        num_rubrics = len(first_n[int(i)]["Rubrics"])
        valid_sampling_scores = []
        for trial in trials:
            if len(trial) < num_rubrics:
                continue
            total_score = sum(_extract_score(trial[k]) for k in range(num_rubrics))
            if not np.isnan(total_score):
                valid_sampling_scores.append(total_score)
        if len(valid_sampling_scores) >= min_valid_samples:
            valid_conv_data.append(valid_sampling_scores[:min_valid_samples])
            conversation_total_scores.append(sum(item["points"] for item in first_n[int(i)]["Rubrics"]))
    return np.array(valid_conv_data), np.array(conversation_total_scores)


def combine_score_matrices(score_matrix_dict, total_score_dict, gen_models, judge_models):
    combined_scores = {}
    combined_totals = {}
    for judge_model in judge_models:
        combined_scores[judge_model] = np.concatenate([score_matrix_dict[g][judge_model] for g in gen_models], axis=0)
        combined_totals[judge_model] = np.concatenate([total_score_dict[g][judge_model] for g in gen_models], axis=0)
    return combined_scores, combined_totals


def calculate_stability(scores_matrix, n_values, B=50, normalized=False, total_scores=None):
    stability_metrics = []
    sample_size = scores_matrix.shape[1]
    for n in n_values:
        bootstrap_means = np.zeros((scores_matrix.shape[0], B))
        for b in range(B):
            indices = np.random.choice(sample_size, n, replace=True)
            bootstrap_means[:, b] = np.nanmean(scores_matrix[:, indices], axis=1)
        stds_of_means = np.nanstd(bootstrap_means, axis=1, ddof=1)
        if normalized and total_scores is not None:
            stds_of_means = stds_of_means / total_scores
        stability_metrics.append(np.nanmean(stds_of_means))
    return stability_metrics


def compute_stability_for_all(result, first_n, gen_models, judge_models, min_valid_samples=12, B=50):
    rubric_scores_matrix = defaultdict(dict)
    rubric_total_scores_matrix = defaultdict(dict)
    conversation_scores_matrix = defaultdict(dict)
    conversation_total_scores_matrix = defaultdict(dict)

    for gen_model in gen_models:
        for judge_model in judge_models:
            rubric_scores_matrix[gen_model][judge_model], rubric_total_scores_matrix[gen_model][judge_model] = (
                extract_rubric_scores(result, first_n, gen_model, judge_model, min_valid_samples=min_valid_samples)
            )
            conversation_scores_matrix[gen_model][judge_model], conversation_total_scores_matrix[gen_model][judge_model] = (
                extract_conversation_scores(result, first_n, gen_model, judge_model, min_valid_samples=min_valid_samples)
            )

    rubric_combined, rubric_total_combined = combine_score_matrices(
        rubric_scores_matrix, rubric_total_scores_matrix, gen_models, judge_models
    )
    conversation_combined, conversation_total_combined = combine_score_matrices(
        conversation_scores_matrix, conversation_total_scores_matrix, gen_models, judge_models
    )

    n_vals = range(1, min_valid_samples + 1)
    sems_rubr, sems_conv = {}, {}
    sems_rubr_norm, sems_conv_norm = {}, {}
    for judge_model in judge_models:
        sems_rubr[judge_model] = calculate_stability(rubric_combined[judge_model], n_vals, B=B)
        sems_conv[judge_model] = calculate_stability(conversation_combined[judge_model], n_vals, B=B)
        sems_rubr_norm[judge_model] = calculate_stability(
            rubric_combined[judge_model], n_vals, B=B, normalized=True, total_scores=rubric_total_combined[judge_model]
        )
        sems_conv_norm[judge_model] = calculate_stability(
            conversation_combined[judge_model],
            n_vals,
            B=B,
            normalized=True,
            total_scores=conversation_total_combined[judge_model],
        )

    return {
        "rubric_scores_matrix": rubric_scores_matrix,
        "rubric_total_scores_matrix": rubric_total_scores_matrix,
        "conversation_scores_matrix": conversation_scores_matrix,
        "conversation_total_scores_matrix": conversation_total_scores_matrix,
        "rubric_scores_matrix_combined": rubric_combined,
        "rubric_total_scores_matrix_combined": rubric_total_combined,
        "conversation_scores_matrix_combined": conversation_combined,
        "conversation_total_scores_matrix_combined": conversation_total_combined,
        "sems_rubr": sems_rubr,
        "sems_conv": sems_conv,
        "sems_rubr_normalized": sems_rubr_norm,
        "sems_conv_normalized": sems_conv_norm,
        "n_vals": list(n_vals),
    }
