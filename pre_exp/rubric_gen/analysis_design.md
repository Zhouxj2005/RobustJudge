# Query-Level Analysis Design

涉及两个脚本：

- `pre_exp/rubric_gen/query_level_r_sem_analysis.py`
- `pre_exp/rubric_gen/query_level_spearman_analysis.py`
## 归一化

### Query-level

分母是当前 `rubric-sample` 自己的总分。

### Rubric-level

分母是这个 rubric 自己的 `points`。

## 分桶

### R-SEM 脚本

- Query-level 独立桶：`(question_id, gen_model, rubric_sample_id)`
- Query-level 融合桶：`(question_id, gen_model)`
- Rubric-level 桶：`(question_id, gen_model, rubric_sample_id, rubric_index)`

桶里存的都是已经归一化后的分数。

### Spearman 脚本

- Query-level 独立桶：`(question_id, gen_model, rubric_sample_id)`
- Query-level 融合桶：`(question_id, gen_model)`

每个桶除了 `relative_scores`，还带对应的 `gt_relative_score`。

GT 的 query-level 相对分数定义为：

```text
gt_relative_score = sum(ground_truth_scores) / sum(ground_truth_weights)
```

## R-SEM

对一个桶、一个给定 `n`：

1. 有放回抽样 `n` 次，求这 `n` 次分数的均值
2. 重复 `B` 次，得到 `B` 个 bootstrap means，对这 `B` 个均值求标准差（Standard Error）

这个标准差就是该桶在当前 `n` 下的 R-SEM。

对所有可用桶取平均，得到该 `n` 下的 `mean_r_sem`。

所以：

- 独立桶曲线：固定一套 rubric-list 时的平均波动
- 融合桶曲线：混合使用 rubric-lists 时的平均波动
- Rubric-level 曲线：单个 rubric 的平均波动

## Spearman

Spearman 脚本保留同一个核心算法：

1. 每轮 bootstrap 先为每个比较对象生成一个预测分数，用这些预测分数和 GT 做一次 `spearmanr`
2. 重复 `B` 轮，对 `B` 个 Spearman 结果取平均

### 融合桶 Spearman

每轮对每个 `(question_id, gen_model)` 融合桶：随机抽 `n` 个样本，求均值

然后在 `(question_id, gen_model)` 粒度上与 GT 算 Spearman。

多轮Spearmanr求平均

### 独立桶 Spearman

为了比较“固定一个 rubric-list”与“混合使用 rubric-lists”：

- 每轮先对每个 `(question_id, gen_model)` 随机选一个 `rubric-sample`
- 再从该 sample 的独立桶里抽 `n` 个样本求均值
- 最后在 `(question_id, gen_model)` 粒度上与 GT 算 Spearman

这样两条 Spearman 曲线比较的是：

```text
固定使用一套 rubric-list vs 混合使用 rubric-lists
```

<img src="D:\学校学习\李侃老师科研\Evaluation\LLM-as-a-Judge鲁棒性分析\RobutsEval\RubricHub\figures\generated_rubric_criteria_level_r_sem_vs_n.png" alt="generated_rubric_criteria_level_r_sem_vs_n" style="zoom: 25%;" />

<img src="D:\学校学习\李侃老师科研\Evaluation\LLM-as-a-Judge鲁棒性分析\RobutsEval\RubricHub\figures\generated_rubric_query_level_r_sem_vs_n.png" alt="generated_rubric_query_level_r_sem_vs_n" style="zoom: 25%;" />

<img src="D:\学校学习\李侃老师科研\Evaluation\LLM-as-a-Judge鲁棒性分析\RobutsEval\RubricHub\figures\generated_rubric_query_level_spearman_vs_n.png" alt="generated_rubric_query_level_spearman_vs_n" style="zoom:25%;" />
