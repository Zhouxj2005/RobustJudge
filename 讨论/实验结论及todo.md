## 结论

1. **R-SEM的计算**：“先计算SEM再归一化”和“先归一化再求SEM”在criteria-level和query-level都是一样的
2. 同一query上，query-level  R-SEM 比 criteria-level R-SEM 的 **加权平均** 一定更小
3. criteria-level R-SEM 的 **加权平均** 小于等于 **算术平均** 的条件是：
   (1) 所有rubric等权（满分points相同）
   (2) rubric-level R-SEM 与 points **不正相关**
4. query-level的 `rubric_count`、`total_points`、`response_words` 越大，`criteria-level R-SEM` 越大，对`query-level R-SEM`和一致性影响小
   `question_words`越大，MAE越大
   `gt_norm_score`越大，**鲁棒性越好（SEM越小），一致性越好**
   `gt_midness`越大，**鲁棒性越差**，一致性略微变差
5. **pair-wise 设定中**，最小化错判概率 <=> 最大化 $\frac{(w^T\mu)^2}{w^T\cdot\Sigma \cdot w}$ 
    $\mu_k=E[Z_k|Y=+1]$ ， $Z_k\in\R$ 表示**任意测评信号**（只需其满足**亚高斯分布**，实际中可用 A 和 B 的评分差） 
   $\Sigma_y(i,j)=Cov(Z_i,Z_j|Y=y)，y\in\{+1,-1\}$
6. 优化生成 Rubric 的方法：加入 **对比样本** 、根据区分度**动态分裂rubric**、用**人类标注做RL**、加入 **weight**、**rubric正交化**
7. 如果**所有rubric正交且 $\mu_k$ 相同**， $k$ 越多， $P(\hat{Y}\neq Y)$ 越小

## Todo

1. 若 **R-SEM与points弱正相关** ，但criteria-level R-SEM 的 **加权平均** 仍小于等于 **算术平均** ，正相关上限是什么？

2. 修正理论模型：$\varepsilon$ 的分布应为 prmopt $p$ 的条件分布。在此修正下重推两个现象的解释。
3. 增设实验修正：
   (1) 同一命名：item->criteria , conv -> query
   (2) rubric_alignment -> mean_rubric_ae / query_ae
   (3) 增设gt_norm_score-rubric_alignment/MAE的散点图
4. 直接生成rubric
   (1) 相同question，rubric越多（直接统计/只统计可区分rubric），criteria-level Judge是否更稳定（上下文过长是否影响判断）？
   (2) 相同question，rubric越多（直接统计/只统计可区分rubric），query-level Judge是否更稳定（多rubric相互抵消）？
   

