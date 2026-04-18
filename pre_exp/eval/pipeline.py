from __future__ import annotations

import argparse

from .alignment import compute_alignment_for_all
from .case_study import analyze_impact_of_gen_model, analyze_top_unstable_items, extract_case_study_data, inspect_unstable_dialog_prompts, plot_dialogue_instability_overlap
from .client import Get
from .config import BASE_MIN_VALID_SAMPLES, BOOTSTRAP_B, GEN_MODELS, JUDGE_MODELS, PATHS, PROMPT_VARIANT_JUDGE_MODELS, ensure_output_dirs
from .data import load_first_n_dataset, load_json, save_json
from .factors import build_factor_dataframe, plot_factor_analysis
from .plotting import configure_matplotlib, plot_figure
from .prompt_sensitivity import compute_prompt_sensitivity_alignment, compute_prompt_sensitivity_stability
from .scoring import load_prompt_templates, run_or_load_generation, run_or_load_scoring
from .stability import compute_stability_for_all


def load_base_artifacts():
    ensure_output_dirs()
    configure_matplotlib()
    first_n = load_first_n_dataset()
    client = Get()
    prompt_templates = load_prompt_templates(PATHS.prompt)
    responses = run_or_load_generation(first_n, GEN_MODELS, client, PATHS.model_res)
    result = run_or_load_scoring(first_n, responses, GEN_MODELS, JUDGE_MODELS, client, prompt_templates, PATHS.result)
    return first_n, responses, result


def run_all_sections():
    first_n, responses, result = load_base_artifacts()

    stability = compute_stability_for_all(result, first_n, GEN_MODELS, JUDGE_MODELS, min_valid_samples=BASE_MIN_VALID_SAMPLES, B=BOOTSTRAP_B)
    save_json(stability["sems_rubr"], PATHS.stability_item)
    save_json(stability["sems_conv"], PATHS.stability_query)
    plot_figure(stability["n_vals"], stability["sems_rubr"], title="Criteria-level 评分波动性与采样次数", num=stability["rubric_scores_matrix_combined"])
    plot_figure(stability["n_vals"], stability["sems_conv"], title="Query-level 评分波动性与采样次数", num=stability["conversation_scores_matrix_combined"])
    plot_figure(stability["n_vals"], stability["sems_rubr_normalized"], title="Criteria-level 相对评分波动性与采样次数")
    plot_figure(stability["n_vals"], stability["sems_conv_normalized"], title="Query-level 相对评分波动性与采样次数")

    df_case_study = extract_case_study_data(result, first_n, GEN_MODELS, JUDGE_MODELS)
    analyze_top_unstable_items(df_case_study)
    plot_dialogue_instability_overlap(df_case_study)
    inspect_unstable_dialog_prompts(first_n)
    analyze_impact_of_gen_model(df_case_study)

    ground_truth = load_json(PATHS.ground_truth)
    alignment = compute_alignment_for_all(result, first_n, ground_truth, GEN_MODELS, JUDGE_MODELS, B=BOOTSTRAP_B)
    plot_figure(stability["n_vals"], alignment["avg_rubric_spearman"], title="Criteria-level 一致性与采样次数", ylabel="Spearman Correlation")
    plot_figure(stability["n_vals"], alignment["avg_conv_spearman"], title="Item-level 一致性与采样次数", ylabel="Spearman Correlation")

    result2 = load_json(PATHS.prompt_variant_result)
    sensitivity_stability = compute_prompt_sensitivity_stability(result2, first_n, GEN_MODELS, PROMPT_VARIANT_JUDGE_MODELS, B=BOOTSTRAP_B)
    sensitivity_alignment = compute_prompt_sensitivity_alignment(result2, first_n, ground_truth, GEN_MODELS, PROMPT_VARIANT_JUDGE_MODELS, B=BOOTSTRAP_B)
    save_json(sensitivity_stability["sems_rubr_v2"], PATHS.stability_item_v2)
    save_json(sensitivity_stability["sems_conv_v2"], PATHS.stability_query_v2)

    sems_rubr_v2 = {"qwen3-32b-v2": sensitivity_stability["sems_rubr_v2"]["qwen3-32b"][:12], "qwen3-32b": stability["sems_rubr"]["qwen3-32b"]}
    sems_conv_v2 = {"qwen3-32b-v2": sensitivity_stability["sems_conv_v2"]["qwen3-32b"][:12], "qwen3-32b": stability["sems_conv"]["qwen3-32b"]}
    plot_figure(range(1, 13), sems_rubr_v2, title="多prompt下Criteria-level 评分波动性与采样次数")
    plot_figure(range(1, 13), sems_conv_v2, title="多prompt下Query-level 评分波动性与采样次数")

    avg_rubric_spearman_v2 = {"qwen3-32b-v2": sensitivity_alignment["avg_rubric_spearman_v2"]["qwen3-32b"][:12], "qwen3-32b": alignment["avg_rubric_spearman"]["qwen3-32b"]}
    avg_conv_spearman_v2 = {"qwen3-32b-v2": sensitivity_alignment["avg_conv_spearman_v2"]["qwen3-32b"][:12], "qwen3-32b": alignment["avg_conv_spearman"]["qwen3-32b"]}
    plot_figure(range(1, 13), avg_rubric_spearman_v2, title="多prompt Criteria-level 一致性与采样次数", ylabel="Spearman Correlation")
    plot_figure(range(1, 13), avg_conv_spearman_v2, title="多prompt Item-level 一致性与采样次数", ylabel="Spearman Correlation")

    df_factors = build_factor_dataframe(result, ground_truth, responses, first_n, GEN_MODELS, JUDGE_MODELS, min_valid_samples=BASE_MIN_VALID_SAMPLES)
    plot_factor_analysis(df_factors)


def main():
    parser = argparse.ArgumentParser(description="Modular version of exp.ipynb")
    parser.add_argument("--run", choices=["all"], default="all")
    args = parser.parse_args()
    if args.run == "all":
        run_all_sections()


if __name__ == "__main__":
    main()
