# 生成rubric实验代码说明

## 1. 生成 rubric

### `gen_rubric.py`
- 功能：对数据集中前 `100` 个问题生成 `6` 组 rubric-list，并对生成结果中不原子的 rubric 做 split。
- 主要输入：
  - prompt 配置：`rubric_syn_prompt.json`
  - 默认主 prompt：`dedup_oriented_prompt_v4`
  - 默认拆分 prompt：
    - `rubric_split_system_prompt_v2`
    - `rubric_split_prompt_v2`
- 模型调用：
  - `api_qwen32b.py`
- 主要输出：
  - `rubric_with_dedup_oriented_prompt.json`
- 输出内容：
  - 每个问题一条记录，包括：`question_index`、`question`、`filled_prompt`、`rubric_responses`
  - `rubric_responses` 是同一问题生成的 `6` 组 rubric-list

### `api_qwen32b.py`
- 功能：封装生成 rubric / judge rubric 时使用的 Qwen API 调用。
- 输入：prompt、system prompt、temperature、timeout 等。
- 输出：模型原始文本回复。

## 2. rubric 去重得到 matrix

### `get_matrix.py`
- 功能：对同一问题下多组 rubric-list 做语义对齐，得到 `unique_rubric` 集合以及 list-match matrix。
- 主要输入：
  - 生成 rubric 结果：`rubric_with_dedup_oriented_prompt.json`
  - prompt 配置：`rubric_dedup_prompt.json`
  - 默认匹配 prompt：
    - `system_prompt_list_match_with_question_requirement_v3`
    - `list_match_prompt_with_question_requirement_v3`
- 模型调用：
  - `api_kimi.py`
- 主要输出：
  - `rubric_matrix_list_match.json`
- 输出内容：
  - `unique_rubrics`：该问题下去重后的 rubric 集合
  - `sample_match_indices`：每个 rubric-list 中每条 rubric 对应的 `unique_rubric` 编号，索引从 `0` 开始
  - `matrix`：`rubric-list × unique_rubric` 的二值出现矩阵

### `api_kimi.py`
- 功能：封装 rubric 去重/匹配时使用的 Kimi API 调用。
- 输入：prompt、system prompt 等。
- 输出：模型原始文本回复。

## 3. 使用 rubric 进行 judge

### `judge_generated_rubrics.py`
- 功能：使用生成的 rubric-list 对已有 response 进行 judge。
- 主要输入：
  - 生成 rubric：`rubric_with_dedup_oriented_prompt.json`
  - response：`../../model_res.json`
  - judge prompt：`../../prompt.json`
  - 使用的 prompt key：`list-grader-template`
  - judge 模型：固定为 `qwen3-32b`
- 模型调用：
  - `api_qwen32b.py`
- 主要输出：
  - `generated_rubric_judge_result.json`
- 输出内容：
  - 结构为：`question -> gen_model -> sample_idx -> repeated trials`
  - 每个 `sample_idx` 对应一组生成 rubric-list，每组默认重复 judge `8` 次

## 4. 对 judge 结果进行分析

### `query_level_r_sem_analysis.py`
- 功能：计算生成 rubric judge 结果的 query-level / rubric-level `R-SEM`。
- 主要输入：
  - `rubric_with_dedup_oriented_prompt.json`
  - `generated_rubric_judge_result.json`
- 主要输出：
  - `query_level_r_sem_metrics.json`
  - 图：
    - `figures/generated_rubric_query_level_r_sem_vs_n.png`
    - `figures/generated_rubric_criteria_level_r_sem_vs_n.png`
- 输出内容：
  - query-level 独立桶 / 融合桶的 `R-SEM`
  - rubric-level 的 `R-SEM`

### `query_level_spearman_analysis.py`
- 功能：计算生成 rubric judge 的 query-level 分数与标准 GT query-level 分数之间的 Spearman。
- 主要输入：
  - `generated_rubric_judge_result.json`
  - `rubric_with_*prompt*.json`
  - `../../ground_truth.json`
- 主要输出：
  - `query_level_spearman_metrics.json`
  - 图：`figures/generated_rubric_query_level_spearman_vs_n.png`
- 输出内容：
  - query-level 独立桶 / 融合桶随采样次数变化的 Spearman

### `query_level_mae_analysis.py`
- 功能：计算生成 rubric judge 的 query-level MAE，并区分不同独立桶聚合方式。
- 主要输入：
  - `generated_rubric_judge_result.json`
  - `rubric_with_*prompt*.json`
  - `../../ground_truth.json`
- 主要输出：
  - `query_level_mae_metrics.json`
  - 图：`figures/generated_rubric_query_level_mae_vs_n.png`
- 输出内容：
  - 独立桶 MAE 的 `avg / min / max` 聚合结果

### `rubric_context_consistency_analysis.py`
- 功能：检验“同一个 `unique_rubric` 在不同 rubric-list 下，judge 分数分布是否一致”。
- 主要输入：
  - `rubric_matrix_list_match.json`
  - `generated_rubric_judge_result.json`
- 主要输出：
  - `rubric_context_consistency_metrics.json`
  - 图：`figures/generated_rubric_context_consistency_t_obs.png`
- 输出内容：
  - 每个 `(question, gen_model, unique_rubric)` 的 `T_obs`、`p_value`、`context_details`

### `compare_generated_vs_std_query_scores.py`
- 功能：比较“生成 rubric judge 得到的 query-level score”是否高于“标准 rubric + `qwen3-32b` judge 得到的 query-level score”。
- 主要输入：
  - 标准 judge 结果：`../../result.json`
  - 标准 GT：`../../ground_truth.json`
  - 生成 rubric judge 结果：`generated_rubric_judge_result.json`
  - 生成 rubric：`rubric_with_dedup_oriented_prompt.json`
- 主要输出：
  - `generated_vs_std_query_score_comparison.json`
- 输出内容：
  - 每个 `(question, gen_model)` 上生成 rubric 分数与标准 rubric 分数的对比结果

## 5. case study

### `rubric_dedup_case_study.py`
- 功能：把 rubric 去重结果整理成可读 markdown，查看每个问题的 `unique_rubrics` 和各 sample 的对应关系。
- 主要输入：
  - `rubric_matrix_list_match.json`
  - `rubric_with_dedup_oriented_prompt.json`
- 主要输出：
  - `rubric_list_dedup_case_study.md`
- 输出内容：
  - 每个问题的 `unique_rubrics`
  - 每个 sample 的原始 rubric-list 与激活的 unique ids

### `rubric_context_consistency_case_study.py`
- 功能：对 `T_obs` 最高的若干 `(q,g,u)` 做 case study，展开 question、response、local rubric、judge evidence。
- 主要输入：
  - `rubric_context_consistency_metrics.json`
  - `rubric_matrix_list_match.json`
  - `rubric_with_dedup_oriented_prompt.json`
  - `generated_rubric_judge_result.json`
  - `../../model_res.json`
- 主要输出：
  - `rubric_context_consistency_top10_case_study.md`
- 输出内容：
  - Top `(q,g,u)` 的 question、response、matched rubric、score list 和 evidence

### `case_study.py`
- 功能：较早期的去重 case study 脚本，基于文本相似度展示 local rubric 与 unique rubric 的匹配关系。
- 主要输入：
  - `rubric_with_simplified_prompt.json`
  - `rubric_matrix.json`
- 主要输出：
  - `rubric_case_study_first3_v2.md`
- 输出内容：
  - 前几个问题中 local rubric 到 unique rubric 的文本匹配结果

## 6. 其他说明

- `analysis_design.md`
  - 功能：记录分析设计，不直接执行。
- `plan.md`
  - 功能：实验记录/草稿，不是主流程脚本。
- `generated_rubric_judge_result(simplified prompt).json`、`rubric_matrix_100.json` 等
  - 功能：较早期实验产物，不是当前主流程的默认输入。

## 7. 主流程最简版

1. `gen_rubric.py`
   - 生成为每个问题生成多组 rubric-list
2. `get_matrix.py`
   - 对多组 rubric-list 去重并建立 `unique_rubric` / `matrix`
3. `judge_generated_rubrics.py`
   - 用生成 rubric 去 judge response
4. 分析脚本
   - `query_level_r_sem_analysis.py`
   - `query_level_spearman_analysis.py`
   - `query_level_mae_analysis.py`
   - `rubric_context_consistency_analysis.py`
   - `compare_generated_vs_std_query_scores.py`
5. case study 脚本
   - `rubric_dedup_case_study.py`
   - `rubric_context_consistency_case_study.py`
