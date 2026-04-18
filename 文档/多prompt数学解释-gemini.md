这是一个非常敏锐且专业的观察。原稿件中的第 4~6 节为了追求直觉上的好懂，确实在数学表达上做了大量简化（例如直接套用加性模型、生硬地把 MSE 转换为 Spearman 的正态假设等），这在正式的学术推导中显得有些“民科”或不够严密。

为了让这部分能直接作为顶级会议/期刊论文的 Theoretical Analysis，我们需要摒弃那些含糊的“约等于”，转而建立一个**层次概率模型 (Hierarchical Probabilistic Model)**，并利用**全方差公式 (Law of Total Variance)** 和**中心极限定理 (CLT)** 来推导一致性“先低后高”的必然性。

以下我为你重新梳理并严格推导的 4~6 节内容，你可以直接替换或参考。

---

### 重新推导：多 Prompt 采样池一致性“先低后高”的理论分析

要严密解释“多 Prompt 混合虽然在单次采样时表现出更差的方差，但在多次采样后能在排序一致性上反超单 Prompt”这一现象，我们需要从**层次生成过程**出发，严格区分“采样噪声”与“Prompt 偏置”，并最终将其映射到排序概率上。

#### 1. 建立层次概率模型 (Hierarchical Probabilistic Model)

我们不再使用简单的加性模型，而是将大模型 (LLM) 的打分过程视为一个条件概率生成过程。

对于给定的评估对象 $i$，其真实的 Ground Truth 分数为 $G_i$。
设 $\mathcal{P}$ 为一个**语义等价但 token 表达不同**的 Prompt 分布空间。
当给定具体的 prompt $p \in \mathcal{P}$ 时，LLM 输出的分数 $Y$ 是一个随机变量。我们定义：
* **Prompt 特定的期望分数**： $\mu_i(p) = \mathbb{E}[Y \mid p]$
* **Prompt 特定的系统偏置**： $b_i(p) = \mu_i(p) - G_i$
* **Prompt 特定的解码方差**： $\sigma_i^2(p) = \mathrm{Var}(Y \mid p)$

**单 Prompt 采样 (Single-Prompt)：**
固定使用某一个 prompt $p_0$。进行 $n$ 次独立重复解码采样，其估计均值记为 $\bar{Y}_{s}^{(n)}$。
根据独立同分布性质，其统计矩为：
$$
\mathbb{E}[\bar{Y}_{s}^{(n)}] = G_i + b_i(p_0)
$$
$$
\mathrm{Var}(\bar{Y}_{s}^{(n)}) = \frac{\sigma_i^2(p_0)}{n}
$$

**多 Prompt 采样 (Multi-Prompt)：**
每次采样不仅引入解码随机性，同时从 $\mathcal{P}$ 中独立抽取 prompt $p_t \sim \mathcal{P}$。进行 $n$ 次采样后取均值，记为 $\bar{Y}_{m}^{(n)}$。
根据**全期望公式**与**全方差公式 (Law of Total Variance)**，其统计矩为：
$$
\mathbb{E}[\bar{Y}_{m}^{(n)}] = \mathbb{E}_{p}[\mathbb{E}[Y \mid p]] = G_i + \mathbb{E}_{p}[b_i(p)]
$$
$$
\mathrm{Var}(\bar{Y}_{m}^{(n)}) = \frac{1}{n} \mathrm{Var}_{p, Y}(Y) = \frac{1}{n} \Big( \mathbb{E}_{p}[\sigma_i^2(p)] + \mathrm{Var}_{p}(\mu_i(p)) \Big)
$$

#### 2. 偏差-方差的严格不等式关系 (The Bias-Variance Trade-off)

为了对比，我们定义两个核心参量：
* 单 Prompt 偏置：$B_s = b_i(p_0)$
* 多 Prompt 偏置（即 Prompt 分布的期望偏置）：$B_m = \mathbb{E}_{p}[b_i(p)]$

**假设 1（多 Prompt 的去偏效应）：** 由于 $\mathcal{P}$ 中的 prompt 在语义上等效且扰动方向具有一定对称性，在边缘化 (Marginalize) prompt 分布后，其系统性偏置会被大幅削弱，即严格满足 $|B_m| < |B_s|$。
**推论 1（多 Prompt 的方差膨胀）：** 比较两者的方差系数，假设 $p_0$ 的方差代表了平均水平（$\sigma_i^2(p_0) \approx \mathbb{E}_{p}[\sigma_i^2(p)]$），那么多 Prompt 的单次方差必然更大，因为它额外引入了 prompt 期望波动的方差项：
$$
\Sigma_m^2 = \mathbb{E}_{p}[\sigma_i^2(p)] + \mathrm{Var}_{p}(\mu_i(p)) > \sigma_i^2(p_0) = \Sigma_s^2
$$

此时，两者的均方误差 (MSE) 随 $n$ 的变化率完全确定：
$$
\mathrm{MSE}_s(n) = B_s^2 + \frac{\Sigma_s^2}{n}
$$
$$
\mathrm{MSE}_m(n) = B_m^2 + \frac{\Sigma_m^2}{n}
$$
由于 $B_m^2 < B_s^2$ 且 $\Sigma_m^2 > \Sigma_s^2$，必然存在一个唯一的交叉阈值 $n^\star = \frac{\Sigma_m^2 - \Sigma_s^2}{B_s^2 - B_m^2}$。在 $n < n^\star$ 时，多 Prompt 误差更大；$n > n^\star$ 时，多 Prompt 误差更小。

#### 3. 从 MSE 严格推导到排序一致性 (Ranking / Spearman)

这一步是原稿件中最薄弱的环节。评价指标（如 Spearman 或 Kendall's Tau）的本质是**成对比较的正确率 (Pairwise Ranking Accuracy)**。我们必须从排序概率来推导，而不是含糊地套用正态分布。

考虑任意一对真实分数不同的评估对象 $(i, j)$，不妨设 $G_i > G_j$，即真实差距 $\Delta G = G_i - G_j > 0$。
评测器给出正确排序的条件是 $\Delta \bar{Y}^{(n)} = \bar{Y}_i^{(n)} - \bar{Y}_j^{(n)} > 0$。

我们分别考察单 Prompt 和多 Prompt 在成对差值上的分布均值与方差：
* **单 Prompt 差值均值**：$\mu_{\Delta s} = \Delta G + (B_{s,i} - B_{s,j}) \equiv \Delta G + \Delta B_s$
* **多 Prompt 差值均值**：$\mu_{\Delta m} = \Delta G + (B_{m,i} - B_{m,j}) \equiv \Delta G + \Delta B_m$

根据中心极限定理 (CLT)，当 $n$ 逐渐增大时，样本均值差 $\Delta \bar{Y}^{(n)}$ 渐进服从正态分布：
$$
\Delta \bar{Y}_s^{(n)} \sim \mathcal{N}\Big(\mu_{\Delta s}, \frac{2\Sigma_s^2}{n}\Big)
$$
$$
\Delta \bar{Y}_m^{(n)} \sim \mathcal{N}\Big(\mu_{\Delta m}, \frac{2\Sigma_m^2}{n}\Big)
$$
（注：这里为简化符号假设两对象方差同质，不同质不影响单调性结论）。

排序正确的概率 $P(\Delta \bar{Y}^{(n)} > 0)$ 即为：
$$
P_s(n) \approx \Phi\left( \sqrt{\frac{n}{2\Sigma_s^2}} (\Delta G + \Delta B_s) \right)
$$
$$
P_m(n) \approx \Phi\left( \sqrt{\frac{n}{2\Sigma_m^2}} (\Delta G + \Delta B_m) \right)
$$
其中 $\Phi$ 为标准正态累积分布函数，它是严格单调递增的。

**核心论证：为什么会出现“先低后高，最后反超”？**

**阶段一：小样本区间（$n$ 较小，方差主导）**
当 $n$ 很小时（尤其是 $n=1$ 或 $2$），由于 $\Sigma_m^2 > \Sigma_s^2$，多 Prompt 差值的分布被大幅“拉平”。即便其均值 $\mu_{\Delta m}$ 可能比 $\mu_{\Delta s}$ 更接近真实的 $\Delta G$，但巨大的方差会导致多 Prompt 分布有更多的面积落入 $<0$ 的错误区间。因此，在小样本下，$P_m(n) < P_s(n)$，多 Prompt 一致性更低。

**阶段二：大样本的渐进行为（$n \to \infty$，偏差主导）**
随着 $n \to \infty$，方差项 $2\Sigma^2/n \to 0$。此时概率的极限行为完全取决于**均值的符号**。
由于多 Prompt 的去偏效应，其偏置远小于真实差距（即 $|\Delta B_m| < \Delta G$），因此 $\Delta G + \Delta B_m > 0$ 恒成立，所以：
$$
\lim_{n \to \infty} P_m(n) = \Phi(+\infty) = 100\%
$$
多 Prompt 在大样本下能够实现趋于完美的渐进排序。

反观单 Prompt，如果该 prompt $p_0$ 在这对对象上产生了**逆转真实差距的恶性偏置**（即 $\Delta B_s < -\Delta G$），其均值 $\mu_{\Delta s}$ 将变为负数！此时：
$$
\lim_{n \to \infty} P_s(n) = \Phi(-\infty) = 0\%
$$
无论单 Prompt 采样多少次，它都会固执地给出错误的排序。即便 $\Delta B_s$ 没有大到逆转排序，只要 $|\Delta B_m| < |\Delta B_s|$，单 Prompt 达到相同排序正确率所需的采样次数 $n$ 将远大于多 Prompt。

**结论：**
因为 Spearman 相关系数本质上是所有 pair 正确率的宏观积分。在低采样次数下，多 Prompt 被其庞大的 Prompt 异质性方差（$\mathrm{Var}_{p}(\mu_i(p))$）拖累，排序错误率高；但随着采样次数增加，均值定理抹平了方差，单 Prompt 陷入了自身固有的系统偏置（Bias）导致的排序“死胡同”，而多 Prompt 依靠其**零均值扰动**的特性（期望偏置更小），最终在宏观 Spearman 指标上实现反超。

---

### 为什么这套推导更好？
1. **摒弃了随意的近似**：直接使用概率论中的 Law of Total Variance 拆解了“为什么多 prompt 方差大”，将其精确归因为 $\mathrm{Var}_{p}(\mu_i(p))$（Prompt 期望之间的差异）。
2. **逻辑闭环**：从最底层的生成机制（CLT）推导到了具体的 pairwise ranking probability，而不是通过 MSE 强行生搬硬套 Spearman。
3. **解释了单 Prompt 的“致命伤”**：不仅说明了多 prompt 会变好，还通过极限理论指出了单 prompt 在大样本下面临的理论上限问题（如果 $\Delta B_s$ 为负且绝对值大，无限采样也是错的），这在论文中是一个非常具有洞察力（Insightful）的论点。