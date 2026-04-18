from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import scipy.stats

from .scoring import _extract_score


def get_spearman(data, first_n, ground_truth, gen_model, judge_model, B=50):
    valid_trials_list, gt_rubric_scores, gt_conv_scores, max_scores_list = [], [], [], []
    for i in data.keys():
        trials = data[i][gen_model][judge_model]
        num_rubrics = len(first_n[int(i)]["Rubrics"])
        max_score = sum(rubric["points"] for rubric in first_n[int(i)]["Rubrics"])
        valid_trial_scores = []
        for trial in trials:
            if len(trial) < num_rubrics:
                continue
            trial_scores = [_extract_score(trial[k]) for k in range(num_rubrics)]
            if not np.any(np.isnan(trial_scores)):
                valid_trial_scores.append(trial_scores)
        if len(valid_trial_scores) >= 12:
            valid_trials_list.append(np.array(valid_trial_scores[:12]))
            gt_score = [score["score"] for score in ground_truth[i][gen_model]]
            gt_rubric_scores.append(np.array(gt_score))
            gt_conv_scores.append(sum(gt_score) / max_score)
            max_scores_list.append(max_score)
        else:
            print(f"对话 {i} 因合法完整采样仅有 {len(valid_trial_scores)} 次(需12次)，已被整体剔除")

    m = len(valid_trials_list)
    gt_conv_scores = np.array(gt_conv_scores)
    max_scores_list = np.array(max_scores_list)
    rubric_spearman_matrix = np.zeros((m, 12))
    conv_spearman_array = np.zeros(12)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in range(1, 13):
            b_rubric_spearman = np.zeros((B, m))
            b_conv_spearman = np.zeros(B)
            for b in range(B):
                b_total_scores = np.zeros(m)
                idx = np.random.randint(0, 12, size=j)
                for i in range(m):
                    sampled_avg = valid_trials_list[i][idx].mean(axis=0)
                    rho_rubric, _ = scipy.stats.spearmanr(sampled_avg, gt_rubric_scores[i])
                    b_rubric_spearman[b, i] = rho_rubric
                    b_total_scores[i] = sampled_avg.sum() / max_scores_list[i]
                rho_conv, _ = scipy.stats.spearmanr(b_total_scores, gt_conv_scores)
                b_conv_spearman[b] = rho_conv
            rubric_spearman_matrix[:, j - 1] = np.nanmean(b_rubric_spearman, axis=0)
            conv_spearman_array[j - 1] = np.nanmean(b_conv_spearman)
    return rubric_spearman_matrix, conv_spearman_array


def compute_alignment_for_all(result, first_n, ground_truth, gen_models, judge_models, B=50):
    rubric_spearman_dict = defaultdict(dict)
    conv_spearman_dict = defaultdict(dict)
    for gen_model in gen_models:
        for judge_model in judge_models:
            rubric_spearman_dict[gen_model][judge_model], conv_spearman_dict[gen_model][judge_model] = get_spearman(
                result, first_n, ground_truth, gen_model, judge_model, B=B
            )

    avg_rubric_spearman, avg_conv_spearman = {}, {}
    for judge_model in judge_models:
        rubric_list = [rubric_spearman_dict[g][judge_model] for g in gen_models]
        conv_list = [conv_spearman_dict[g][judge_model] for g in gen_models]
        avg_rubric_spearman[judge_model] = np.nanmean(np.vstack(rubric_list), axis=0)
        avg_conv_spearman[judge_model] = np.nanmean(np.vstack(conv_list), axis=0)
    return {
        "rubric_spearman_dict": rubric_spearman_dict,
        "conv_spearman_dict": conv_spearman_dict,
        "avg_rubric_spearman": avg_rubric_spearman,
        "avg_conv_spearman": avg_conv_spearman,
    }
