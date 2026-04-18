# exp.ipynb 实验内容总结

## 1. 实验目标

`exp.ipynb` 的核心目标是分析 **LLM-as-a-Judge** 在 rubric-based 评测场景下的表现，重点关注以下三个问题：

1. 重复采样是否能够提升 Judge 打分的鲁棒性。
2. 重复采样是否能够提升 Judge 与 Ground Truth 之间的一致性。
3. 输入改写（prompt variant）是否会影响 Judge 的稳定性，以及多 prompt 混合采样是否能够进一步改善评测表现。

整体上，这份实验是在研究：**通过多次采样和输入扰动，能否让 LLM Judge 的评分更稳定、更可靠、更接近参考标准。**

## 2. 数据与实验设置

### 2.1 数据集

实验使用 `RubricHub_v1` 中的 parquet 数据，通过 `datasets.load_dataset` 读取，并从中选取前 100 条 query 作为实验样本。

每条数据主要包含：

- `prompt`：待回答的问题或指令
- `Rubrics`：该问题对应的评分标准列表

### 2.2 回答生成模型

对每条 query，先用多个生成模型生成 response。当前 notebook 中使用的生成模型包括：

- `qwen2.5-72b`
- `gpt-oss-120b`
- `qwen3-235b`

生成结果保存在 `model_res.json` 中；若该文件已存在，则直接加载，避免重复调用 API。

### 2.3 Judge 模型

对于每个 `(query, response, rubric-list)` 组合，使用多个 Judge 模型进行打分：

- `deepseek-v3.2`
- `qwen2.5-7b`
- `qwen3-32b`
- `gpt-4o`

评分结果保存在 `result.json` 中。

### 2.4 评分方式

Notebook 中通过 `prompt.json` 里的 `list-grader-template` 构造评分 prompt，将 query、response 和 rubric-list 填入模板后发送给 Judge 模型。

Judge 返回的结果会被解析为 rubric 级别的评分列表，形成如下结构：

`result[i][generation_model][judge_model][j][k]`

其中：

- `i` 表示第 `i` 条数据
- `generation_model` 表示回答生成模型
- `judge_model` 表示评分模型
- `j` 表示第 `j` 次采样评分
- `k` 表示第 `k` 个 rubric item 的得分

## 3. 实验一：重复采样对鲁棒性的提升

### 3.1 实验目的

这一部分研究的是：对于同一个 response，如果让 Judge 重复打分多次，再对评分结果求平均，是否能降低分数波动、提升稳定性。

### 3.2 采样设置

每个 Judge 对同一条样本重复评分 16 次。

不过在后续分析中，并不是所有采样都会被保留。代码中会过滤不完整或非法的评分结果，仅保留：

- rubric 级别上至少有 12 次有效采样的 item
- 对话级别上至少有 12 次有效采样的对话

为保证维度对齐，最终每个样本只截取前 12 次有效评分用于分析。

### 3.3 分析粒度

这一部分在两个层面上计算稳定性：

- `Criteria-level`：单个 rubric item 的得分波动
- `Query-level`：整条对话总分的波动

其中，对话总分是一个样本下所有 rubric 得分之和。

### 3.4 评分清洗与统一

由于 Judge 返回格式可能存在差异，代码通过 `_extract_score` 统一解析评分：

- 若返回字段中有 `score`，直接使用该数值
- 若返回 `is_met` 和 `weight`，则按照是否满足条件映射为对应得分
- 无法解析时记为 `NaN`

这一步的作用是把不同形式的 Judge 输出转成统一的数值评分。

### 3.5 稳定性指标

鲁棒性部分使用 bootstrap 重采样来模拟“进行 n 次评审并求平均”的过程，并计算平均分的标准差。

具体做法是：

1. 对每个样本的 12 次评分中，有放回地随机抽取 `n` 次。
2. 计算这 `n` 次评分的平均值。
3. 重复上述过程 `B=50` 次，得到 50 个平均值。
4. 计算这 50 个平均值的标准差。
5. 对所有样本取平均，作为当前采样次数 `n` 下的稳定性指标。

代码中将该指标记作“平均值的标准差”，本质上反映的是：**如果只进行 n 次采样并取均值，该均值会有多大波动。**

此外，还计算了归一化版本：

- rubric 级波动除以该 rubric 的总分
- 对话级波动除以该对话的总分

这样可以比较不同满分尺度下的相对稳定性。

### 3.6 当前结果趋势

从 notebook 中已有结果可以看出，随着采样次数从 1 增加到 12，所有 Judge 模型的评分波动都明显下降，说明：

- **重复采样 + 平均** 能够显著提升评测稳定性
- 采样次数越多，分数越不容易受到单次随机性的影响

当前 rubric-level 的绝对波动结果显示：

- `deepseek-v3.2`：从约 `0.497` 下降到 `0.144`
- `gpt-4o`：从约 `0.537` 下降到 `0.152`
- `qwen2.5-7b`：从约 `0.986` 下降到 `0.281`
- `qwen3-32b`：从约 `1.108` 下降到 `0.296`

当前 query-level 的绝对波动结果也呈现同样趋势：

- `deepseek-v3.2`：从约 `8.44` 下降到 `2.67`
- `gpt-4o`：从约 `8.51` 下降到 `2.45`
- `qwen2.5-7b`：从约 `16.76` 下降到 `5.15`
- `qwen3-32b`：从约 `15.76` 下降到 `4.27`

初步来看：

- `deepseek-v3.2` 和 `gpt-4o` 的稳定性整体更好
- `qwen2.5-7b` 和 `qwen3-32b` 的评分波动更大

因此，Judge 模型本身的选择会显著影响最终评测鲁棒性。

### 3.7 可视化结果

Notebook 绘制了四类曲线图：

- `Criteria-level 评分波动性与采样次数`
- `Query-level 评分波动性与采样次数`
- `Criteria-level 相对评分波动性与采样次数`
- `Query-level 相对评分波动性与采样次数`

这些图用于展示不同 Judge 在不同采样次数下的波动变化趋势。

## 4. 实验一补充：Case Study

在稳定性分析基础上，notebook 还做了两个补充研究。

### 4.1 不同 Judge 最不稳定 Item 的共性

代码会提取每个 item 的统计信息，例如：

- 标准差或相对标准差
- rubric-list 长度
- 所属生成模型与 Judge 模型

然后对每个 Judge 找出最不稳定的 top-k item，分析这些 item 是否具有某些共同特征，例如：

- rubric-list 是否更长
- 某类任务是否更容易引起评分波动

### 4.2 生成模型对不稳定性的影响

notebook 还分析了不同 `gen_model` 是否会影响 Judge 的评分稳定性，即：

- 不同生成模型生成的回答，是否会让同一个 Judge 表现出不同程度的波动

这部分通过透视表和热力图统计不同 `(judge_model, gen_model)` 组合下的平均相对波动。

### 4.3 Judge 间相关性分析

代码将对话级不稳定性按 Judge 聚合后，计算不同 Judge 之间的 Spearman 相关性，用于回答：

- 不同 Judge 是否会在相同的对话上同时表现出“不稳定”

这可以帮助判断“不稳定样本”是否具有跨 Judge 的共性。

此外，notebook 还手动查看了一些不稳定对话，如 `16`、`17`、`24`、`38`，用于定性分析其 prompt 特征。

## 5. 实验二：重复采样对一致性的提升

### 5.1 实验目的

实验一主要关注“稳不稳”，而实验二进一步研究：

- 重复采样之后，Judge 的平均分是否会更接近 Ground Truth

也就是说，这部分研究的是评测的**一致性**，而不是仅仅看方差是否减小。

### 5.2 Ground Truth

Notebook 从 `ground_truth.json` 中读取参考评分，结构为：

`ground_truth[i][gen][j]`

表示第 `i` 条对话、由 `gen` 生成的回答、在第 `j` 个 rubric 上的 Ground Truth 分数。

### 5.3 一致性指标

这一部分使用 **Spearman 相关系数** 衡量 Judge 评分与 Ground Truth 之间的一致性。

分析同样分为两个层面：

- `Criteria-level`
  对每条对话内部，比较 rubric 级平均分与 Ground Truth rubric 分数之间的 Spearman 相关性。
- `Item-level / Query-level`
  对所有对话，比较归一化总分与 Ground Truth 总分之间的 Spearman 相关性。

### 5.4 计算过程

代码流程如下：

1. 对每条样本筛选出至少 12 次完整合法评分。
2. 对于采样次数 `j = 1, 2, ..., 12`：
   - 从 12 次评分中有放回抽取 `j` 次
   - 对抽到的评分求平均
   - 将平均结果与 Ground Truth 计算 Spearman 相关
3. 重复 `B=50` 次 bootstrap
4. 取平均相关系数，得到不同采样次数下的一致性曲线

因此，这部分是在回答：

- 如果只采样一次，一致性有多高？
- 如果采样多次再平均，一致性能否继续提升？

### 5.5 输出结果

Notebook 绘制了两张图：

- `Criteria-level 一致性与采样次数`
- `Item-level 一致性与采样次数`

用于展示 Spearman correlation 随采样次数变化的趋势。

如果曲线随采样次数上升，则说明重复采样不仅让分数更稳定，也让评分结果更接近参考标准。

## 6. 实验三：输入敏感性实验

### 6.1 实验目的

这一部分关注 Judge 对输入形式变化的敏感性，即：

- 当 query 或评分 prompt 发生语义等价改写时，Judge 的结果是否会明显变化

进一步地，还研究：

- 如果将不同 prompt 变体下的评分结果混合起来再做重复采样，是否能进一步提升鲁棒性与一致性

### 6.2 Prompt 变体

Notebook 中列出了 4 类 prompt variant：

- 翻译成英文
- 翻译成西班牙语
- 使用 `deepseek v3.2` 改写
- 使用 `qwen2.5 7b` 改写

结果存放在 `exp2_result.json` 中，其结构为：

`result2[i][gen][judge][variant][j][k]`

表示：

- 第 `i` 条对话
- 用 `gen` 生成回答
- 用 `judge` 评分
- 在某个 `variant` 输入下
- 第 `j` 次采样
- 第 `k` 个 rubric 的得分

### 6.3 多 prompt 混合采样

在这一部分中，代码会把所有 variant 下的采样结果展平组合。

设定为：

- 总采样数 `TOTAL_SAMPLES = 20`
- 至少保留 `MIN_VALID_SAMPLES = 16` 次有效采样

也就是说，代码会把多种 prompt 变体下的评分统一看作一组更大的采样池，再从中研究：

- 稳定性是否继续改善
- 与 Ground Truth 的一致性是否继续提高

### 6.4 当前分析对象

在这部分 notebook 中，Judge 模型只保留了：

- `qwen3-32b`

然后将原始单 prompt 条件下的结果与多 prompt 条件下的结果进行对比，用 `qwen3-32b-v2` 表示多 prompt 版本。

### 6.5 输出结果

这一部分同样计算并可视化：

- 多 prompt 下 `Criteria-level` 波动性
- 多 prompt 下 `Query-level` 波动性
- 多 prompt 下归一化波动性
- 多 prompt 下与 Ground Truth 的 Spearman 一致性

因此，实验三实际上是在验证：

- **输入改写会不会引入额外不稳定性**
- **把不同 prompt variant 一起纳入采样后，能否抵消单一 prompt 的偶然性**

## 7. 整体实验逻辑

如果从整体上看，这份 notebook 的实验逻辑可以概括为三步：

1. 先让多个生成模型对 query 生成回答。
2. 再让多个 Judge 模型对回答按 rubric 多次评分。
3. 最后从鲁棒性、一致性、输入敏感性三个角度分析多次采样的价值。

因此，这不是一个单纯的“模型打分”脚本，而是一套较完整的 **Judge 可靠性评估框架**。

## 8. 当前可得的初步结论

基于 notebook 当前已经跑出的结果，可以得到以下初步判断：

1. 重复采样能够显著降低 Judge 打分波动，提高稳定性。
2. 不同 Judge 模型之间的稳定性差异明显，Judge 的选择本身就是重要变量。
3. 稳定性分析不仅可以在 rubric 级别做，也可以在 query 总分级别做，两者都显示出多次采样的收益。
4. notebook 已经进一步尝试分析“不稳定样本”的结构性来源，例如 rubric 长度、生成模型差异、Judge 间共性等。
5. 在引入 Ground Truth 后，可以继续验证多次采样是否不仅更稳，而且更接近参考标准。
6. 输入改写实验说明，Judge 的输出可能对 prompt 形式敏感，而多 prompt 混合采样是一个潜在的缓解方向。

## 9. 代码层面的一个注意点

在生成 response 的部分，如果本地不存在 `model_res.json`，代码中目前有一行：

```python
break # for testing, 先生成一条就保存，后续再生成剩下的
```

这意味着：

- 如果没有现成缓存，当前 notebook 实际上只会生成第一条样本的回答
- 后续实验依赖已有的 `model_res.json`

因此，如果后面要完整复现实验，需要确认：

- `model_res.json` 是否已经完整保存了 100 条数据
- 或者删除这行测试用的 `break`

## 10. 一句话总结

`exp.ipynb` 当前完成的是一套围绕 **LLM-as-a-Judge 鲁棒性分析** 的实验，系统研究了：

- 多次采样是否让评分更稳定
- 多次采样是否让评分更接近 Ground Truth
- prompt 改写是否会影响 Judge，以及多 prompt 混合采样是否能缓解这种敏感性

其核心结论方向是：**重复采样是提升 LLM Judge 可靠性的有效方法，而 Judge 模型选择和输入形式都会显著影响最终评测质量。**


