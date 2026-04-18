# Rubric-based LLM-as-a-Judge 鲁棒性实验的数学解释

结合 [exp.ipynb](d:/学校学习/李侃老师科研/Evaluation/LLM-as-a-Judge鲁棒性分析/RobutsEval/RubricHub/exp.ipynb) 的实验定义，我们主要想解释两个现象：

1. 在“3.1 重复采样对鲁棒性的提升”中，`criteria-level` 的相对 SEM 比 `query-level` 更大。
2. 在“3.3 输入敏感性实验”中，多 prompt 的 `item-level` 一致性随着采样次数增加，出现了“先低后高，最终反超单 prompt”的现象。

---

## 1. 先对齐实验里的指标

根据代码，`stability` 的核心过程是：

1. 对一个对象的多次采样结果，有放回地抽取 $n$ 次。
2. 对这 $n$ 次采样取均值。
3. 重复 $B$ 次，得到 $B$ 个“采样均值”。
4. 计算这 $B$ 个均值的标准差，作为该对象在采样次数 $n$ 下的 SEM。

这里最关键的是，`criteria-level` 和 `query-level` 的“相对 SEM”并不是同一个量的两种写法，而是两套不同的归一化与平均流程。

### 1.1 Criteria-level 相对 SEM

对每个有效的 `(query i, rubric r)`：

- 取这个 rubric 在多次 judge 采样中的得分序列
- bootstrap 计算其“采样 $n$ 次后的均值”的标准差，记为
  $$
  SEM_{ir}(n)
  $$
- 再除以该 rubric 的满分 $P_{ir}$，得到相对 SEM：
  $$
  R^{\text{crit}}_{ir}(n)=\frac{SEM_{ir}(n)}{P_{ir}}
  $$

最后，对所有有效 `(i,r)` 的这个相对 SEM 取平均：

$$
\overline{R}^{\text{crit}}(n)
=
\frac{1}{N_{\text{crit}}}
\sum_{(i,r)\in\mathcal D_{\text{crit}}}
R^{\text{crit}}_{ir}(n).
$$

### 1.2 Query-level 相对 SEM

对每个有效 query $i$：

- 每次采样先把该 query 下所有 rubric 分数求和，得到 query 总分
- 对这个“总分序列”做 bootstrap，计算采样 $n$ 次后的均值标准差，记为
  $$
  SEM^{\text{query}}_i(n)
  $$
- 再除以该 query 的总满分
  $$
  P_i=\sum_{r=1}^{m_i}P_{ir}
  $$
  得到相对 SEM：
  $$
  R^{\text{query}}_i(n)=\frac{SEM^{\text{query}}_i(n)}{P_i}
  $$

最后，对所有有效 query 的这个相对 SEM 取平均：

$$
\overline{R}^{\text{query}}(n)
=
\frac{1}{N_{\text{query}}}
\sum_{i\in\mathcal D_{\text{query}}}
R^{\text{query}}_i(n).
$$

所以真正想论证的是：

$$
\overline{R}^{\text{query}}(n)
<
\overline{R}^{\text{crit}}(n).
$$

也就是：

- `criteria-level`：每个 rubric 单独做“SEM / rubric 满分”，再平均
- `query-level`：每个 query 先聚合总分，再做“SEM / query 总满分”，再平均

---

## 2. 统一概率模型

符号约定：

- $i$：query（或 item）
- $r$：query $i$ 下第 $r$ 个 rubric
- $p\in\mathcal P$：语义等价但 token 不同的 prompt 变体
- $t$：第 $t$ 次采样（解码/打分一次）

层次概率模型（把 prompt 视为外层随机变量，把解码随机性视为内层随机性）：

对给定 $(i,r)$ 与 prompt $p$，Judge 分数 $Y_{irpt}$ 是随机变量，定义其条件矩：

$$
\mu_{ir}(p)=\mathbb E[Y_{irpt}\mid p],\quad
b_{ir}(p)=\mu_{ir}(p)-G_{ir},\quad
\sigma_{ir}^2(p)=\mathrm{Var}(Y_{irpt}\mid p).
$$

因此可以写成加性分解（只是符号重写，不额外假设）：

$$
Y_{irpt}=G_{ir}+b_{ir}(p)+\epsilon_{irpt},\quad \mathbb E[\epsilon_{irpt}\mid p]=0,\quad \mathrm{Var}(\epsilon_{irpt}\mid p)=\sigma_{ir}^2(p).
$$

其中：

- $G_{ir}$：该 rubric 的 GT 分数
- $b_{ir}(p)$：prompt 引入的系统漂移（prompt effect）
- $\epsilon_{irpt}$：给定 prompt 后的解码噪声

两种采样机制：

- Single-prompt：固定 $p=p_0$，仅对解码噪声重复采样
- Multi-prompt：每次采样先抽 $p_t\sim\mathcal P$，再解码得到 $Y_{ir p_t t}$

对同一对象采样 $n$ 次并取均值：

$$
\bar Y_{ir}^{(n)}=\frac{1}{n}\sum_{t=1}^n Y_{ir p_t t}.
$$

在 single-prompt（$p_t\equiv p_0$）且各次采样独立同分布时，

$$
\mathrm{Var}(\bar Y_{ir}^{(n)})=\frac{\sigma_{ir}^2(p_0)}{n}.
$$

这对应实验里“重复采样取平均会使波动按 $1/\sqrt n$ 衰减”的现象。

---

## 3. 为什么 Query-level 相对 SEM 通常小于 Criteria-level 相对 SEM

### 3.1 先把理论对象和代码口径对齐

这部分不能再像普通“总分方差 vs 单项方差”那样直接比，因为你代码里比较的是两个**不同归一化顺序**下的平均相对 SEM。

因此更准确的对象是：

- `criteria-level`
  $$
  \overline{R}^{\text{crit}}(n)
  =
  \text{平均}_{(i,r)}
  \left(
  \frac{SEM_{ir}(n)}{P_{ir}}
  \right)
  $$

- `query-level`
  $$
  \overline{R}^{\text{query}}(n)
  =
  \text{平均}_{i}
  \left(
  \frac{SEM_i^{\text{query}}(n)}{P_i}
  \right)
  $$

其中 $P_i=\sum_r P_{ir}$。


### 3.2 先按“先求 SEM，再归一化”的含义定义量

这里的相对 SEM 有非常明确的解释：

> 在采样 $n$ 次并取平均之后，剩余的随机波动范围占总分的比例。

因此，对第 $i$ 个 query、第 $r$ 个 rubric：

$$
R^{\text{crit}}_{ir}(n)=\frac{SEM_{ir}(n)}{P_{ir}}
$$

表示：

- 这个 rubric 在重复采样并平均之后，还剩多少随机波动
- 这个波动相对于该 rubric 满分 $P_{ir}$ 占多大比例

而对第 $i$ 个 query：

$$
R^{\text{query}}_i(n)=\frac{SEM^{\text{query}}_i(n)}{P_i},
\qquad
P_i=\sum_{r=1}^{m_i}P_{ir}
$$

表示：

- 这个 query 的总分在重复采样并平均之后，还剩多少随机波动
- 这个波动相对于 query 总满分 $P_i$ 占多大比例

所以第 3 节真正比较的是“随机波动占总分比例”的大小，而不是归一化分数本身的方差。

### 3.3 Query-level 相对 SEM 的推导

设第 $i$ 个 query 有 $m_i$ 个 rubric。单次采样下，第 $r$ 个 rubric 的打分为

$$
Y_{irt}=G_{ir}+b_{ir}+\varepsilon_{irt}.
$$

记 query 的单次总分为

$$
S_{it}=\sum_{r=1}^{m_i}Y_{irt}.
$$

对 $n$ 次采样取平均：

$$
\bar S_i^{(n)}=\frac{1}{n}\sum_{t=1}^{n}S_{it}.
$$

由于 bootstrap 里的 SEM 本质上就是“采样均值的标准差”，因此 query-level 的 SEM 可近似写为

$$
SEM_i^{\text{query}}(n)= \sqrt{\mathrm{Var}(\bar S_i^{(n)})}.
$$

又因为

$$
S_{it}=\sum_{r=1}^{m_i}Y_{irt},
$$

所以

$$
\mathrm{Var}(\bar S_i^{(n)})
=
\frac{1}{n}
\mathrm{Var}(S_{i1})
=
\frac{1}{n}
\left[
\sum_{r=1}^{m_i}\sigma_{ir}^2
+
\sum_{r\neq s}\Sigma_i(r,s)
\right],
$$

其中

$$
\sigma_{ir}^2=\mathrm{Var}(\varepsilon_{irt}),
\qquad
\Sigma_i(r,s)=\mathrm{Cov}(\varepsilon_{irt},\varepsilon_{ist}).
$$

因此：

$$
SEM_i^{\text{query}}(n)
=
\frac{1}{\sqrt n}
\sqrt{
\sum_{r=1}^{m_i}\sigma_{ir}^2
+
\sum_{r\neq s}\Sigma_i(r,s)
}.
$$

再除以总满分 $P_i$，得到 query-level 相对 SEM：

$$
R_i^{\text{query}}(n)
=
\frac{1}{P_i\sqrt n}
\sqrt{
\sum_{r=1}^{m_i}\sigma_{ir}^2
+
\sum_{r\neq s}\Sigma_i(r,s)
}.
$$

这一步才是和代码一致的顺序：先求总分均值的波动，再除以总满分。


### 3.4 Criteria-level 相对 SEM 的推导

对单个 `(query i, rubric r)`，其采样均值为

$$
\bar Y_{ir}^{(n)}=\frac{1}{n}\sum_{t=1}^{n}Y_{irt}.
$$

于是

$$
SEM_{ir}(n)= \sqrt{\mathrm{Var}(\bar Y_{ir}^{(n)})}
=
\frac{\sigma_{ir}}{\sqrt n}.
$$

再除以该 rubric 满分 $P_{ir}$，得到单个 rubric 的相对 SEM：

$$
R^{\text{crit}}_{ir}(n)
=
\frac{\sigma_{ir}}{P_{ir}\sqrt n}.
$$

### 3.5 引入权重与相对误差的重构

为了严谨比较 query-level 和 criteria-level 的相对 SEM 大小，我们定义两个核心变量：

1. **单个 rubric 的相对标准差（Relative Noise Level）：**
   记第 $i$ 个 query 下第 $r$ 个 rubric 单次采样的相对标准差为 $c_{ir}$：
   $$
   c_{ir} = \frac{\sigma_{ir}}{P_{ir}}
   $$
   结合 3.4 节，可得 $R^{\text{crit}}_{ir}(n) = \frac{c_{ir}}{\sqrt{n}}$。

2. **单个 rubric 的满分权重（Score Weight）：**
   记第 $r$ 个 rubric 的满分在第 $i$ 个 query 总满分中所占的权重为 $w_{ir}$：
   $$
   w_{ir} = \frac{P_{ir}}{P_i}
   $$
   显然有 $\sum_{r=1}^{m_i} w_{ir} = 1$。

同时，将 rubric 之间的噪声协方差 $\Sigma_i(r,s)$ 用皮尔逊相关系数 $\rho_{ir,is}$ 展开：
$$
\Sigma_i(r,s) = \rho_{ir,is} \sigma_{ir} \sigma_{is}
$$
其中 $\rho_{ir,is} \in [-1, 1]$。

现在，将这些变量代入 3.3 节推导出的 Query-level 相对 SEM 公式中：
$$
R_i^{\text{query}}(n) = \frac{1}{P_i\sqrt n} \sqrt{ \sum_{r=1}^{m_i}\sigma_{ir}^2 + \sum_{r\neq s}\Sigma_i(r,s) }
$$

将 $\sigma_{ir} = c_{ir} P_{ir}$ 代入，并提取 $P_i$ 进根号内，可以得到一个非常优雅的形式：
$$
R_i^{\text{query}}(n) = \frac{1}{\sqrt n} \sqrt{ \sum_{r=1}^{m_i} (w_{ir} c_{ir})^2 + \sum_{r\neq s} \rho_{ir,is} (w_{ir} c_{ir}) (w_{is} c_{is}) }
$$

### 3.6 核心不等式：Query-level 相对 SEM 的严格上界

现在我们要证明，Query-level 的相对 SEM 存在一个严格的上界。

由于相关系数 $\rho_{ir,is} \le 1$，如果我们将根号内部的 $\rho_{ir,is}$ 全部放大替换为 $1$，根号内的值必然变大（或相等）：
$$
\sum_{r=1}^{m_i} (w_{ir} c_{ir})^2 + \sum_{r\neq s} \rho_{ir,is} (w_{ir} c_{ir}) (w_{is} c_{is}) 
\le 
\sum_{r=1}^{m_i} (w_{ir} c_{ir})^2 + \sum_{r\neq s} (w_{ir} c_{ir}) (w_{is} c_{is})
$$

观察不等式右侧，这恰好是多项式完全平方的展开式！即：
$$
\sum_{r=1}^{m_i} (w_{ir} c_{ir})^2 + \sum_{r\neq s} (w_{ir} c_{ir}) (w_{is} c_{is}) = \left( \sum_{r=1}^{m_i} w_{ir} c_{ir} \right)^2
$$

对两边同时开根号，我们得到了 Query-level 相对 SEM 的不等式：
$$
R_i^{\text{query}}(n) \le \frac{1}{\sqrt n} \sum_{r=1}^{m_i} w_{ir} c_{ir} = \sum_{r=1}^{m_i} w_{ir} \left( \frac{c_{ir}}{\sqrt n} \right)
$$

代入 3.4 节的结论，即得：
$$
R_i^{\text{query}}(n) \le \sum_{r=1}^{m_i} w_{ir} R^{\text{crit}}_{ir}(n)
$$

**这里的物理意义极其重要：**
它在数学上严格证明了，一个 Query 总分的相对 SEM，**永远小于或等于**其内部各个 Rubric 相对 SEM 的**加权平均值**（以分数占比为权重）。

**严格不等号成立的条件：**
等号成立的唯一条件是该 query 下所有 rubric 的打分误差完全正相关（对于所有的 $r \neq s$，$\rho_{ir,is} = 1$）。由于大模型（LLM）在不同 rubric 上的解码噪声具有独立性或微弱相关性，这种极端共振在实际中不可能发生。因此，**严格不等号（$<$）必然成立**。

### 3.7 加权平均与算术平均：全局不等式的成立条件

上一步我们证明了在单一 query $i$ 内部，Query-level 相对 SEM 严格小于 Criteria-level 相对 SEM 的**加权平均**：
$$
R_i^{\text{query}}(n) < \sum_{r=1}^{m_i} w_{ir} R^{\text{crit}}_{ir}(n)
$$

然而在代码实现（即 1.1 节）中，`criteria-level` 是对所有的 rubric 直接取**算术平均**（无权重）。在单一 query 内部，算术平均记为：
$$
\overline{R}_i^{\text{crit}}(n) = \frac{1}{m_i} \sum_{r=1}^{m_i} R^{\text{crit}}_{ir}(n)
$$

要使得最终实验现象 $R_i^{\text{query}}(n) < \overline{R}_i^{\text{crit}}(n)$ 严格成立，我们需要考察**加权平均与算术平均的关系**。根据统计学性质，两者之差等于权重与变量的协方差乘以样本数：
$$
\sum_{r=1}^{m_i} w_{ir} R^{\text{crit}}_{ir}(n) - \frac{1}{m_i} \sum_{r=1}^{m_i} R^{\text{crit}}_{ir}(n) = (m_i - 1) \cdot \text{Cov}\left(w_{ir}, R^{\text{crit}}_{ir}(n)\right)
$$

因此，只要满足以下**两个条件之一**，结论即可严格成立：

1. **等权条件**：若同一 query 内的所有 **rubric 满分相同**（即 $P_{ir}$ 皆相等），则 $w_{ir} = 1/m_i$ 恒定。此时**加权平均等于算术平均**，结合 3.6 节的上界，严格不等式必然成立。
2. **误差非正相关条件**：在真实场景中，一个 rubric 的分数权重 $w_{ir}$（比如满分是 10 分还是 2 分）通常与其相对测量误差 $R^{\text{crit}}_{ir}$ 是**不相关的**（即 $\text{Cov} \approx 0$），或者**满分越高的 rubric 相对误差反而越小**（即 $\text{Cov} < 0$）。在这两种常理假设下，加权平均 $\le$ 算术平均。

只要在绝大多数 query 中满足上述条件，当我们在外层对所有 query 进行全局平均时，必然有：
$$
\overline{R}^{\text{query}}(n) = \frac{1}{N_{\text{query}}} \sum_i R^{\text{query}}_i(n) 
\quad < \quad 
\frac{1}{N_{\text{query}}} \sum_i \overline{R}_i^{\text{crit}}(n) = \overline{R}^{\text{crit}}(n)
$$
这就完成了现象 1 的严格数学解释。

### 3.8 可写成论文里的命题 (Theorem)

**命题 1 (聚合评估中的方差抵消效应 / Variance Cancellation in Aggregated Evaluation).**

设 $R^{\text{crit}}_{ir}(n)$ 与 $R^{\text{query}}_i(n)$ 分别为在采样次数 $n$ 下 criteria-level 与 query-level 的相对标准误 (SEM)。假设同一 query 内不同 rubric 之间的评估噪声并非完全正相关（皮尔逊相关系数 $\rho < 1$），且 **rubric 的满分权重与其相对波动率** 不呈显著正相关，则必然严格成立：
$$
\overline{R}^{\text{query}}(n) < \overline{R}^{\text{crit}}(n)
$$

### 3.9 证明思路简述 (Proof Sketch)

本命题的证明核心在于将方差的累加转化为完全平方的上界放缩。
1. 首先，通过引入分数占比权重 $w_{ir} = P_{ir}/P_i$，将 query-level 的相对 SEM 展开为包含协方差项的根式。
2. 其次，利用相关系数 $\rho < 1$ 的性质进行放缩，将根号内凑成完全平方形式，证明 query-level 相对 SEM 严格受上界限定，该上界即为 criteria-level 相对 SEM 的加权平均值。
3. 最后，根据样本协方差的性质，证明在合理假设下（满分权重与相对误差无正相关），加权平均值不大于算术平均值，从而在全局取均值后得到最终的严格不等关系。

------

## 4. 多 prompt 一致性“先低后高”的理论解释

本节直接复用第 2 节的层次模型（prompt 外层随机性 + 解码内层随机性），只给出与结论相关的统计量与判别条件。

### 4.1 统计矩（全期望与全方差）

为简洁起见，把“评估对象”抽象为 $i$（它可以代表某个 query 的总分、也可以代表某个 rubric 分数）。令 $p\in\mathcal P$，并沿用第 2 节定义：

$$
\mu_i(p)=\mathbb E[Y\mid p],\quad b_i(p)=\mu_i(p)-G_i,\quad \sigma_i^2(p)=\mathrm{Var}(Y\mid p).
$$

Single-prompt（固定 $p_0$）下：

$$
\mathbb{E}[\bar{Y}_{s}^{(n)}] = G_i + b_i(p_0),\quad
\mathrm{Var}(\bar{Y}_{s}^{(n)}) = \frac{\sigma_i^2(p_0)}{n}.
$$

Multi-prompt（$p_t\sim\mathcal P$）下（全方差公式）：

$$
\mathbb{E}[\bar{Y}_{m}^{(n)}] = G_i + \mathbb{E}_{p}[b_i(p)],
$$

$$
\mathrm{Var}(\bar{Y}_{m}^{(n)}) = \frac{1}{n}\Big( \mathbb{E}_{p}[\sigma_i^2(p)] + \mathrm{Var}_{p}(\mu_i(p)) \Big).
$$

### 4.2 MSE 与交叉阈值

定义

$$
B_s=b_i(p_0),\quad B_m=\mathbb E_p[b_i(p)],\quad
\Sigma_s^2=\sigma_i^2(p_0),\quad \Sigma_m^2=\mathbb E_p[\sigma_i^2(p)]+\mathrm{Var}_p(\mu_i(p)).
$$

则

$$
\mathrm{MSE}_s(n) = B_s^2 + \frac{\Sigma_s^2}{n},\quad
\mathrm{MSE}_m(n) = B_m^2 + \frac{\Sigma_m^2}{n}.
$$

当 $B_m^2<B_s^2$ 且 $\Sigma_m^2>\Sigma_s^2$ 时，存在唯一交叉阈值

$$
n^\star = \frac{\Sigma_m^2-\Sigma_s^2}{B_s^2-B_m^2},
$$

并得到“先低后高”：$n<n^\star$ 方差主导（multi 劣），$n>n^\star$ 偏差主导（multi 优）。

### 4.3 从 pairwise 排序到 Spearman

考虑任意一对对象 $(i,j)$，设 $\Delta G=G_i-G_j>0$。正确排序条件为 $\Delta \bar Y^{(n)}=\bar Y_i^{(n)}-\bar Y_j^{(n)}>0$。令

$$
\mu_{\Delta s}=\Delta G+\Delta B_s,\quad
\mu_{\Delta m}=\Delta G+\Delta B_m.
$$

在 CLT 条件下（$n$ 充分大）：

$$
\Delta \bar{Y}_s^{(n)} \sim \mathcal{N}\Big(\mu_{\Delta s}, \frac{2\Sigma_s^2}{n}\Big),\quad
\Delta \bar{Y}_m^{(n)} \sim \mathcal{N}\Big(\mu_{\Delta m}, \frac{2\Sigma_m^2}{n}\Big).
$$

因此

$$
P_s(n) \approx \Phi\left( \sqrt{\frac{n}{2\Sigma_s^2}} (\Delta G + \Delta B_s) \right),\quad
P_m(n) \approx \Phi\left( \sqrt{\frac{n}{2\Sigma_m^2}} (\Delta G + \Delta B_m) \right).
$$

Spearman 可视作大量 pairwise 正确率的宏观聚合，因此当多数组对在足够大的 $n$ 下满足 $P_m(n)>P_s(n)$，整体 Spearman 将表现为“先低后高并反超”。





---

## 5. 两个现象可以统一到同一个框架里

两个现象可以统一为“方差随 $n$ 衰减 + 偏差不随 $n$ 衰减”的分解：

- 现象 A：query-level 的相对 SEM 更小来自聚合效应（第 3 节），其核心是“噪声按平方和累积、满分按线性累积”。
- 现象 B：多 prompt 的“先低后高”来自 $B_m^2$ 与 $\Sigma_m^2$ 的 trade-off（第 4 节），小样本看方差，大样本看偏差。

---

## 6. 可以直接写进论文的命题版本

### 命题 1：Query-level 的平均相对 SEM 通常小于 Criteria-level

若同一 query 内各 rubric 的相对评分噪声处于同一量级，且不同 rubric 的噪声协方差不至于接近完全同步正相关，则 query 总分的相对均值不确定性通常小于单个 rubric 分数的相对均值不确定性。因此，在对所有有效 query 与 rubric-entry 分别取平均后，query-level 的平均相对 SEM 通常小于 criteria-level 的平均相对 SEM。

### 命题 2：多 prompt 采样存在 bias-variance crossover

若对同一对象 $i$，单/多 prompt 的偏差与方差系数满足 $B_m^2<B_s^2$ 且 $\Sigma_m^2>\Sigma_s^2$（定义见第 4.2 节），则存在唯一阈值

$$
n^\star=\frac{\Sigma_m^2-\Sigma_s^2}{B_s^2-B_m^2},
$$

使得 $n<n^\star$ 时单 prompt 的 MSE 更小、$n>n^\star$ 时多 prompt 的 MSE 更小。结合第 4.3 节的 pairwise 排序概率，可得到 Spearman 在足够大 $n$ 下的反超现象。

### 命题 3：语义等价 prompt 的平均可视作去偏

若一组语义等价 prompt 所引入的 token-level 扰动在总体上近似中心化，即

$$
\mathbb{E}_p[B_{irp}] \approx 0,
$$

则多 prompt 混合后的系统偏差将小于单 prompt 的系统偏差；但由于不同 prompt 之间存在异质性，其单次采样方差可能更大。因此，多 prompt 表现出“大方差、小偏差”的统计特征，并在较大采样数下优于单 prompt。

---

## 7. 目前这套解释的边界

主要近似与边界：

- 将 bootstrap 的 SEM 与总体 $\mathrm{Var}(\bar Y^{(n)})$ 对齐（大样本时更合理）。
- 使用 CLT 将差值均值近似为正态（$n$ 足够大时成立）。
- 将 Spearman 的变化解释为 pairwise 正确率的宏观聚合（常见且可操作，但不是等式）。

这些假设都很自然，也足够支撑论文中的理论解释，但如果想做更强的“数学证明”，可能还需要额外假设。

---

## 8. 一句话总结

重复采样将不确定性按 $1/\sqrt n$ 衰减；query-level 聚合进一步降低“波动占总分比例”；多 prompt 通过引入 $\mathrm{Var}_p(\mu(p))$ 换取更小的期望偏差，从而在足够大的 $n$ 下体现一致性反超。
