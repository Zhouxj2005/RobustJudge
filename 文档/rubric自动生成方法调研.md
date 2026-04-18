### 1. SedarEval: Automated Evaluation using Self-Adaptive Rubrics（2025.1）

#### Method1

构造人工reference rubric，拿model生成的rubric与人工rubric做DPO

#### Method2

GPT-4生成rubric和理想answer，然后用Judge判断rubric和answer是否对的上

----

### 2. EVALAGENT: Discovering Implicit Evaluation Criteria from the Web（2025.4）

1. **用LLM生成解决用户 Instruction 应该去 google 的 query**

```
Instruction: {{ Instruction }}
What should I google so I learn useful advice on writing a great response to the above instruction?
Give a JSON response with three most important google queries as keys. The queries should be
in the form of “how to” or “what are” etc and the value is what we want to learn from the query.
The queries should focus on seeking actionable writing advice instead of focusing on abstract
knowledge concepts.
```

2. **google search取top30网页，LLM质量检测（内容专业性、可信度；任务相关性）**
3. **LLM根据每个筛选后的网页生成query建议，LLM总结所有网页建议**
4. **用LLM对所有建议进行视角对齐、相关性过滤（去除过于细枝末节的rubric）**

---

### 3. Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains（2025.10）

+ 主要聚焦使用 `rubric` 作为奖励信号进行 `RL` , 拓展了 `RLVR` 的范畴

#### $Rubric$ 生成方式

```
Query
   ↓
Reference Answer (来自数据集/人工标注/强模型)
   ↓
[GPT-4o / o3-mini] 
   ↓
Rubric List (7-20 条结构化标准)
```

#### Prompt for Rubric Generation

1. **输入query、reference answer**
2. 分3个重要等级（Essential、Important、Optional）和1个错误等级（Pitfall）
3. 按重要度给1\~5和-2\~-1的weight

<img src="C:\Users\Zhouxj's Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260411101319115.png" alt="image-20260411101319115" style="zoom: 50%;" />

> **提出两种rubric-scores聚合方式：**
>
> 1. 加权平均（与现在相同）
> 2. LLM对rubric-list进行聚合、直接给1~10的分

---

### 4. OpenRubrics: Towards Scalable Synthetic Rubric Generation for Reward Modeling and LLM Alignment（2026.2）

+ **对比式**构造rubric
+ 同时 SFT Generator 和 Judge

─────────────────────────────────────────────────────────────
 **阶段一**：**Rubric生成**   
  ① 构造偏好对                                              											       

​    ├── **有分数数据**：最高分 vs 最低分                                             						         

​    ├── **SFT数据**：多模型生成回答 → 开源Reward模型排序 → 最好 vs 最差      			         

​    └── **可验证数据**：满足条件 vs 不满足条件                    							             

​                        ↓                                  													   

  ② **Contrastive Rubric Generation**（每个样本×5次）          							        

​    ├── 输入：问题 + chosen + rejected（或排序列表）       							        

​    ├── Step 1：根据request提取Hard Rules（显式要求）              								          

​    ├── Step 2：对比response分析具体差异（允许主题相关）               								     

​    └── Step 3：step2的结果强制抽象为Principles（通用原则）               							       

​                        ↓                                													   

  ③ Preference-Label Consistency Filtering                 								         

​    ├── 构造所有对比对 P = {(a,b)}                           									        

​    ├── 用Rubric预测每对的偏好，计算准确率 Acc_i                                 											  

​    └── 保留：Acc_i ≥ τ 且 当前样本预测正确的rubric          							         

​                        ↓                                 													  

  输出：高质量的 (问题, rubric) 对，用于训练Rubric-RM  								    

─────────────────────────────────────────────────────────────

阶段二：**SFT** 训练 Rubric Generator 和 Judge

  ① Generator

```
输入：{prompt}  （用户问题）
输出：{rubric}  （结构化Rubric列表）

目标：最大化 P(rubric | prompt)
```

  ① Judge

```
输入1：用户问题 (Instruction/Prompt)    
输入2：回复A (Response A)               
输入3：回复B (Response B)               
输入4：Rubric（由Generator生成或数据集中的）

目标：最大化 P(偏好标签 | 问题, 回复A, 回复B, Rubric)
```

─────────────────────────────────────────────────────────────

阶段三：性能测试

**Rubric-RM** 是一个 **gen+judge** 的整体

第一部分：看这个judge方法跟别的比judge准确度提升了没有（4B反超基础7B，8B接近GPT-4.1-mini）

第二部分：用 **Rubric-RM** 生成 **Reward** 去 **DPO** 一个小模型的生成能力/问题回答能力（比其他开源Reward模型普遍提升2个点）

---

### 5. Rethinking Rubric Generation for Improving LLM Judge and Reward Modeling for Open-ended Tasks（2026.2）

**只优化偏序概率最大化**

**生成rubric**

1. 让 LLM 根据 prompt 和 $m$ 个样本回答（文中用 $m=8$），生成第一批候选 rubric

2. **核心循环**：

   1. 如果一个 rubric 被 $>n$ 个回答满足（文中 $n=2$），就分解成更细粒度的子 rubric。

   2. 对新产生的子rubric**过滤**：

      （1）方向错误（弱模型 比 强模型 更符合）；

      （2）冗余（语义重叠的）
   3. 连续15次分解后生成的新 rubric 被过滤机制拒绝：退出循环

3. 对rubric**加权**
   信噪比<img src="C:\Users\Zhouxj's Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260411163647396.png" alt="image-20260411163647396" style="zoom: 50%;" />要最大化即可最小化误判概率，<img src="C:\Users\Zhouxj's Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260411163758433.png" alt="image-20260411163758433" style="zoom: 67%;" />，<img src="C:\Users\Zhouxj's Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260411163836502.png" alt="image-20260411163836502" style="zoom:67%;" />

   理论最优<img src="C:\Users\Zhouxj's Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260411163909001.png" alt="image-20260411163909001" style="zoom:80%;" />
   
   <img src="C:\Users\Zhouxj's Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260411163924672.png" alt="image-20260411163924672" style="zoom:80%;" />

> 

---

### 6. RubricRAG: Towards Interpretable and Reliable LLM Evaluation via Domain Knowledge Retrieval for Rubric Generation（2026.3）

**1.构建数据库**

**每条记录包含**：

- 一个query，一个rubric list（每条标准包含description和score）

**数据来源**：**OpenAI HealthBench**、**ResearchRubrics**

**2.推理**

**用当前 query 检索语义相似的历史 query，把那些 query 对应的人类 rubric 作为 few-shot exemplars，然后让模型生成当前 query 的 rubric。**

---

###  7. Learning Query-Specific Rubrics from Human Preferences for DeepResearch Report Generation（2026.2）

1. **人工构造**查询和人类偏好对 $(q, r_{acc}, r_{rej})$，据此 Rubric 生成器输出一个评分标准列表 $y$。
2. **用另一个 LLM 当裁判**，根据 $y$ 里的各项标准给两份报告打分（每项1-10分，再按权重加权平均），得到 $S(r_{acc}|y)$ 和 $S(r_{rej}|y)$。
3. **构造奖励：**
   $R(Pref):$ 如果 $S(r_{acc}) > S(r_{rej})$ 得 +1 分，否则 -1 分
   $R_{LLM}:$ 直接评估rubric质量；
   $R_{fmt}:$ 格式错误-1

----

# 属性提升总结

### 1. 区分度

提升对 **水平相近的回答** 的判断，极大减少了 Judge 打平局或随机打分的概率，提升了对高难度偏好对的**评估一致性**

+ **OpenRubrics (2026.2)**：通过**对比式生成（Contrastive Generation）**，直接让模型观察好/坏回答的区别，提取出最能区分这两者的核心差异。

+ **Rethinking Rubric Generation (2026.2)**：通过**动态分解（Decomposition）**，当一个 rubric 被太多回答满足时，说明它太简单（区分度低），模型会将其继续往下细化拆解。

### 2. 权重感知

+ **Rubrics as Rewards (2025.10)**：引入了**重要度分级（Essential/Pitfall等）与正负权重**
+ **Rethinking Rubric Generation (2026.2)**：引入信噪比（SNR）对 Rubric 进行**数学理论层面的加权**，让判定误差小的 Rubric 占更高权重。
+ **带来的提升**：Judge 能够像人类专家一样**抓主要矛盾**。这极大地提升了 Judge 的**鲁棒性**，使其不再容易被华丽辞藻或冗长但偏题的回答

### 3. 客观知识锚定

- **EVALAGENT (2025.4)**：赋予 Rubric **外部真实世界的实操属性**，通过 Google Search 获取真实专家的 How-to 建议，确保标准是业内公认的。
- **RubricRAG (2026.3)**：赋予 Rubric **历史一致性与领域专业性**，通过 RAG 检索已有的专家标注 Rubric 作为 Few-shot，保证了同一领域内评价标准的延续性。

- **带来的提升**：使得最终的 Judge 是基于“客观事实”而非“主观幻觉”在打分，大幅提升了对专业问题评估的**鲁棒性**，减少了评估过程中的知识性误判。

### 4. 人类偏好对齐度

- **SedarEval (2025.1)**：直接用 DPO 对齐人类构造的 Reference Rubric，提升了 Rubric 本身的**拟人度**。

- **Learning Query-Specific Rubrics... (2026.2)**：将 Rubric 作为强化学习策略优化的**隐变量**，其唯一目标就是让 Judge 使用该 Rubric 算出的分数 $S_{acc}>S_{rej}$
- **OpenRubrics (2026.2)** 中的一致性过滤）：保留能真实预测偏好的 Rubric。
- **带来的提升**：这是提升 **人类一致性** 最直接的手段

### 5. 独立性与无冗余

**痛点**：如果生成的多条 Rubric 在语义上高度重合，Judge 在打分时会对同一缺陷或优点进行多次惩罚/奖励（Double-counting），导致分数极化。

- **Rethinking Rubric Generation (2026.2)**：加入了冗余（语义重叠）过滤机制。
- **EVALAGENT (2025.4)**：进行相关性过滤，去除细枝末节，提取核心总结。
- **带来的提升**：赋予了 Rubric 列表内各项的**正交性**，使得评价维度互不干扰，显著提升了加权总分的**鲁棒性和准确性**。



# 启发

1. **Rubric 的获取不应只是 Prompt -> LLM -> Rubric**：需要引入 **对比信息**（好坏样本的差异）或 **外部知识**（Search/RAG）来增强**区分度**和**专业性**。
2. **Rubric 的结构不应只是 平铺的 List**：必须有**权重**和**致命错误识别**，甚至根据难度动态分裂（Dynamic Granularity），来增强**鲁棒性**。
3. **Rubric 的清洗必须有 Validation 闭环**：生成的 Rubric 不能直接用，必须经过“用它去模拟打分，看是否与先验偏好一致”或“信噪比检测”，通过测试的 Rubric 才能留用。
