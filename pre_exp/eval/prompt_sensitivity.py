from __future__ import annotations

from collections import defaultdict

import numpy as np

from .alignment import compute_alignment_for_all
from .scoring import _extract_score
from .stability import calculate_stability


def extract_valid_scores_v2(data, first_n, gen_model, judge_model, min_valid_samples=16):
    valid_rubric_data, rubric_total_scores = [], []
    for i in data.keys():
        trials = []
        for samplings in data[i][gen_model][judge_model].values():
            trials.extend(samplings)
        num_rubrics = len(first_n[int(i)]["Rubrics"])
        for k in range(num_rubrics):
            scores = []
            for trial in trials:
                if len(trial) < num_rubrics:
                    continue
                scores.append(_extract_score(trial[k]))
            if np.sum(~np.isnan(scores)) >= min_valid_samples:
                scores = [score for score in scores if not np.isnan(score)]
                valid_rubric_data.append(scores[:min_valid_samples])
                rubric_total_scores.append(first_n[int(i)]["Rubrics"][k]["points"])
    return np.array(valid_rubric_data), np.array(rubric_total_scores)


def extract_conversation_scores_v2(data, first_n, gen_model, judge_model, min_valid_samples=16):
    valid_conv_data, conversation_total_scores = [], []
    for i in data.keys():
        trials = []
        for samplings in data[i][gen_model][judge_model].values():
            trials.extend(samplings)
        num_rubrics = len(first_n[int(i)]["Rubrics"])
        valid_sampling_scores = []
        for trial in trials:
            if len(trial) < num_rubrics:
                continue
            total_score = sum(_extract_score(trial[k]) for k in range(num_rubrics))
            if total_score is not np.nan:
                valid_sampling_scores.append(total_score)
        if len(valid_sampling_scores) >= min_valid_samples:
            valid_conv_data.append(valid_sampling_scores[:min_valid_samples])
            conversation_total_scores.append(sum(item["points"] for item in first_n[int(i)]["Rubrics"]))
    return np.array(valid_conv_data), np.array(conversation_total_scores)


def compute_prompt_sensitivity_stability(result2, first_n, gen_models, judge_models, min_valid_samples=16, B=50):
    rubric_scores_matrix_v2 = defaultdict(dict)
    rubric_total_score_matrix_v2 = defaultdict(dict)
    conversation_scores_matrix_v2 = defaultdict(dict)
    conversation_total_scores_matrix_v2 = defaultdict(dict)
    for gen_model in gen_models:
        for judge_model in judge_models:
            rubric_scores_matrix_v2[gen_model][judge_model], rubric_total_score_matrix_v2[gen_model][judge_model] = (
                extract_valid_scores_v2(result2, first_n, gen_model, judge_model, min_valid_samples=min_valid_samples)
            )
            conversation_scores_matrix_v2[gen_model][judge_model], conversation_total_scores_matrix_v2[gen_model][judge_model] = (
                extract_conversation_scores_v2(result2, first_n, gen_model, judge_model, min_valid_samples=min_valid_samples)
            )

    rubric_scores_matrix_combined_v2, rubric_total_score_matrix_combined_v2 = {}, {}
    conversation_scores_matrix_combined_v2, conversation_total_scores_matrix_combined_v2 = {}, {}
    for judge_model in judge_models:
        rubric_scores_matrix_combined_v2[judge_model] = np.concatenate(
            [rubric_scores_matrix_v2[g][judge_model] for g in gen_models], axis=0
        )
        rubric_total_score_matrix_combined_v2[judge_model] = np.concatenate(
            [rubric_total_score_matrix_v2[g][judge_model] for g in gen_models], axis=0
        )
        conversation_scores_matrix_combined_v2[judge_model] = np.concatenate(
            [conversation_scores_matrix_v2[g][judge_model] for g in gen_models], axis=0
        )
        conversation_total_scores_matrix_combined_v2[judge_model] = np.concatenate(
            [conversation_total_scores_matrix_v2[g][judge_model] for g in gen_models], axis=0
        )

    n_vals = range(1, min_valid_samples + 1)
    sems_rubr_v2, sems_conv_v2 = {}, {}
    sems_rubr_normalized_v2, sems_conv_normalized_v2 = {}, {}
    for judge_model in judge_models:
        sems_rubr_v2[judge_model] = calculate_stability(rubric_scores_matrix_combined_v2[judge_model], n_vals, B=B)
        sems_conv_v2[judge_model] = calculate_stability(conversation_scores_matrix_combined_v2[judge_model], n_vals, B=B)
        sems_rubr_normalized_v2[judge_model] = calculate_stability(
            rubric_scores_matrix_combined_v2[judge_model],
            n_vals,
            B=B,
            normalized=True,
            total_scores=rubric_total_score_matrix_combined_v2[judge_model],
        )
        sems_conv_normalized_v2[judge_model] = calculate_stability(
            conversation_scores_matrix_combined_v2[judge_model],
            n_vals,
            B=B,
            normalized=True,
            total_scores=conversation_total_scores_matrix_combined_v2[judge_model],
        )
    return {
        "sems_rubr_v2": sems_rubr_v2,
        "sems_conv_v2": sems_conv_v2,
        "sems_rubr_normalized_v2": sems_rubr_normalized_v2,
        "sems_conv_normalized_v2": sems_conv_normalized_v2,
    }


def flatten_prompt_variant_trials(result2):
    comb_result = {}
    for i in result2:
        comb_result[i] = {}
        for gen in result2[i]:
            comb_result[i][gen] = {}
            for judge in result2[i][gen]:
                comb_result[i][gen][judge] = []
                for variant in result2[i][gen][judge]:
                    comb_result[i][gen][judge].extend(result2[i][gen][judge][variant])
    return comb_result


def compute_prompt_sensitivity_alignment(result2, first_n, ground_truth, gen_models, judge_models, B=50):
    comb_result = flatten_prompt_variant_trials(result2)
    alignment = compute_alignment_for_all(comb_result, first_n, ground_truth, gen_models, judge_models, B=B)
    return {
        "avg_rubric_spearman_v2": alignment["avg_rubric_spearman"],
        "avg_conv_spearman_v2": alignment["avg_conv_spearman"],
    }
