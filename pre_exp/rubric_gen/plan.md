## 4.16工作计划
1. 设计rubric生成的prompt，生成rubric-list：要求生成的rubric原子化、可验证（人工check 5个question），暂定无weight，模型采用qwen3-32b
2. 每个question重采样6次
3. 设计propmt，判断2个rubric是否语义等价：模型采用Kimi k2.5，人工check
4. 每个question所有rubric求并，构造rubric得分矩阵（得分为0/1）
5. 对于同一question上的不同rubric-list，独立judge（重采样8次），计算query级和平均rubric级R-SEM。rubric-judge、重采样、计算两种R-SEM的方法见exp.ipynb或eval文件夹下的代码


### 1. 设计prmopt

**参考prompt**（OpenRubrics: Towards Scalable Synthetic Rubric Generation for Reward Modeling and LLM Alignment）：
```
Your task is to extract a set of rubric-style instructions from a user's request. These rubrics will be used as evaluation criteria to check if a response fully meets the request. Every rubric item must be a universal principle. If any rubric still contains topic-specific references (e.g., names, places, myths, numbers, historical facts), it is automatically invalid.
- **Two Distinct Categories:**
  - [Hard Rule]: Derived strictly from explicit requirements stated in the <request> (format, length, structure, forbidden/required elements, etc.). 
  - [Principle]: Derived by abstracting any concrete cues into domain-agnostic quality criteria (e .g., clarity, correctness, sound reasoning, pedagogy). 
- **Comprehensiveness:** The rubric must cover all critical aspects implied by the request and examples, including explicit requirements and implicit quality standards. 
- **Conciseness & Uniqueness:** Each rubric must capture a distinct evaluation criterion. Overlapping or redundant criteria must be merged into a single rubric. Wording must be precise and free of repetition.
- **Format Requirements:** 
  - Use a numbered list. 
  - Each item starts with "The response" phrased in third person. 
  - Append [Hard Rule] or [Principle] at the end of each item. 
  - Do not include reasoning, explanations, or examples in the final output,only the rubrics. 
Here is the request: 
{prompt} 
Please generate the rubrics for the above request.
```


**根据要求优化后的prompt：**
```
Your task is to generate a rubric list from a user request. The rubric list will later be used for binary evaluation (0/1). Therefore, every rubric item must be atomic, explicit, and directly verifiable from a response.

# Goal
Given a user request, produce a set of rubric items that together capture the key requirements for evaluating whether a response satisfies the request.

# Core Requirements
1. Atomicity
   - Each rubric item must express exactly one checkable criterion.
   - Do not combine multiple requirements with "and", "or", "as well as", or other compound structures.
   - If a sentence contains multiple constraints, split them into separate rubric items.

2. Verifiability
   - Each rubric item must be answerable as a clear binary judgment: met or not met.
   - The criterion must be checkable from the response itself, without relying on hidden intent or external interpretation.
   - Do not use vague expressions such as "summarizes the text", "covers the key points", "is informative", "is clear", "includes relevant information", "references", or "mentions the quote" unless they are rewritten into concrete observable requirements.

3. Loyalty
   - Preserve the actual intent of the user request. If a requirement is explicitly stated in the request, it must remain represented in the rubric list.
   - Do not add specific sub-requirements, facts, examples, terminology, formats, or knowledge points unless they are explicitly required by the user request.
   - If the request is broad, keep the rubric broad. Do not make it more specific than the original request.

4. Non-overlap
   - Avoid semantic overlap, near-duplicates, or parent-child redundancy.
   - If two candidate items would usually receive the same judgment, merge or rewrite them.

5. Coverage
   - The full rubric list should cover the major evaluable requirements in the request, including:
     - explicit task requirements
     - output format or structural constraints
     - required or forbidden content
     - key quality constraints that are clearly implied by the request
   - Do not add criteria that are not supported by the request.

# Writing Rules
- Output a JSON array.
- Each element in the array must be an object in the form {"criterion": "..."}.
- Every object must contain exactly one field: "criterion".
- Write the full rubric item text inside the value of "criterion".
- Each rubric item must begin with "The response".
- Keep each item concise and self-contained.
- Do not include explanations, rationales, examples, or any text outside the JSON array.

# Quality Filter
Before finalizing, check every item:
- Is it atomic?
- Is it directly verifiable from the response?
- Is it explicitly supported by the user request?
- Is it non-overlapping with other items?
If any answer is no, revise or remove the item.

# Input Request
{prompt}

# Output
Return only the final rubric list as a JSON array, for example:
[{"criterion": "..."}, {"criterion": "..."},...]
```

**简化prompt：**

```
Your task is to generate a rubric list from a user request. The rubric list will later be used for binary evaluation (0/1). Therefore, every rubric item must be atomic, explicit, and directly verifiable from a response.

# Goal
Given a user request, produce a set of rubric items that together capture the key requirements for evaluating whether a response satisfies the request.

# Core Requirements
1. Atomicity
   - Each rubric item must express exactly one checkable criterion.
   - Do not combine multiple requirements or multiple facts with "and", "or", "as well as", or other compound structures.
   - If a sentence contains multiple constraints, split them into separate rubric items.

2. Verifiability
   - Each rubric item must be answerable as a clear binary judgment: met or not met.
   - The criterion must be checkable from the response itself, without relying on hidden intent or external interpretation.
   - Do not use vague expressions such as "summarizes the text", "covers the key points", "is informative", "is clear", "includes relevant information", "references", or "mentions the quote" unless they are rewritten into concrete observable requirements.

3. Loyalty
   - Preserve the actual intent of the user request. If a requirement is explicitly stated in the request, it must remain represented in the rubric list.
   - Do not add specific sub-requirements, facts, examples, terminology, formats, or knowledge points unless they are explicitly required by the user request.
   - If the request is broad, keep the rubric broad. Do not make it more specific than the original request.

4. Non-overlap
   - Avoid semantic overlap and near-duplicate rubric items.
   - If two items would usually be judged together, merge or rewrite them.

# Writing Rules
- Output a JSON array.
- Each element in the array must be an object in the form {"criterion": "..."}.
- Every object must contain exactly one field: "criterion".
- Write the full rubric item text inside the value of "criterion".
- Each rubric item must begin with "The response".
- Keep each item concise and self-contained.
- Do not include explanations, rationales, examples, or any text outside the JSON array.

# Quality Filter
Before finalizing, check every item:
- Is it atomic?
- Is it directly verifiable from the response?
- Is it explicitly supported by the user request?
- Is it non-overlapping with other items?
If any answer is no, revise or remove the item.

# Input Request
{prompt}

# Output
Return only the final rubric list as a JSON array, for example:
[{"criterion": "..."}, {"criterion": "..."},...]
```



### 原子性check

<img src="C:\Users\Zhouxj's Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260417154313853.png" alt="image-20260417154313853" style="zoom: 80%;" />

```
===== First Question =====
Please read the following text and summarize it. The final output should be limited to 50 words. Here is the text: NAME_1 has been granted extra time to decide whether to compete in the World Cross-Country Championships.The 31-year-old is concerned the event, which starts on 19 March in France, could upset her preparations for the London Marathon on 17 April. 'There is no question that NAME_2 would be a huge asset to the GB team,' said NAME_3 of UK Athletics. 'But she is working out whether she can accommodate the worlds without too much compromise in her marathon training.' NAME_4 must make a decision by Tuesday - the deadline for team nominations. British team member NAME_5 said the team would understand if NAME_4 opted out of the event. 'It would be fantastic to have NAME_2 in the team,' said the European cross-country champion. 'But you have to remember that athletics is basically an individual sport and anything achieved for the team is a bonus. 'She is not messing us around. We all understand the problem.' NAME_4 was world cross-country champion in 2001 and 2002 but missed last year's event because of injury. In her absence, the GB team won bronze in Brussels.

===== First Question Rubric Lists =====

--- sample 1 ---
1. The response includes a summary of the key points from the provided text.
2. The response is written in plain text with no markdown or formatting.
3. The response contains fewer than 50 words.
4. The response mentions NAME_1 being granted extra time to decide about competing in the World Cross-Country Championships.
5. The response mentions the concern about the event potentially disrupting preparations for the London Marathon.
6. The response mentions the quote or reference to NAME_3 stating NAME_1 would be an asset to the GB team.
7. The response includes information about the deadline for team nominations.
8. The response mentions the team's understanding if NAME_1 chooses not to participate.
9. The response includes the reference to NAME_1's previous achievements as a world cross-country champion.
10. The response mentions the GB team's bronze medal win in the absence of NAME_1.

--- sample 2 ---
1. The response summarizes the provided text.
2. The response mentions NAME_1 being granted extra time to decide on the World Cross-Country Championships.
3. The response notes the conflict between the World Cross-Country Championships and the London Marathon.
4. The response includes the quote from NAME_3 stating NAME_4 would be a huge asset to the GB team.
5. The response mentions the deadline for team nominations is Tuesday.
6. The response includes the statement from NAME_5 that the team would understand if NAME_4 opted out.
7. The response mentions that NAME_4 was world cross-country champion in 2001 and 2002.
8. The response states that GB won bronze in the World Cross-Country Championships in the year NAME_4 missed due to injury.
9. The response is under 50 words.

--- sample 3 ---
1. The response summarizes the provided text.
2. The response mentions NAME_1 being granted extra time to decide about competing in the World Cross-Country Championships.
3. The response states the concern about the event conflicting with preparations for the London Marathon.
4. The response includes the quote from NAME_3 about NAME_4 being an asset to the GB team.
5. The response mentions the deadline for team nominations is Tuesday.
6. The response includes the statement from NAME_5 that the team would understand if NAME_4 opted out.
7. The response states that NAME_4 won the world cross-country title in 2001 and 2002 but missed last year’s event due to injury.     
8. The response notes the GB team won bronze in the absence of NAME_4.
9. The response is under 50 words.

--- sample 4 ---
1. The response summarizes the provided text.
2. The response is under 50 words in length.
3. The response mentions NAME_1 being granted extra time to decide on competition.
4. The response references the World Cross-Country Championships starting on 19 March in France.
5. The response mentions NAME_1's concern about preparation for the London Marathon on 17 April.
6. The response includes a statement from NAME_3 about NAME_2 being a huge asset to the GB team.
7. The response states that NAME_4 must make a decision by Tuesday.
8. The response notes that British team member NAME_5 said the team would understand if NAME_4 opted out.
9. The response references NAME_4 being a world cross-country champion in 2001 and 2002.
10. The response mentions that GB won bronze in the absence of NAME_4 at last year's event.

--- sample 5 ---
1. The response summarizes the provided text.
2. The response is written in English.
3. The response is limited to 50 words or fewer.
4. The response mentions that NAME_1 has been granted extra time to decide on competing in the World Cross-Country Championships.     
5. The response includes the concern about the World Cross-Country Championships affecting NAME_1's London Marathon preparations.     
6. The response mentions NAME_3's statement about NAME_1 being a huge asset to the GB team.
7. The response states that NAME_1 must make a decision by Tuesday for team nominations.
8. The response includes the quote or reference from NAME_5 about the team understanding if NAME_1 opts out.
9. The response notes that NAME_1 was world cross-country champion in 2001 and 2002.
10. The response mentions that the GB team won bronze in the absence of NAME_1 at last year’s event.

--- sample 6 ---
1. The response is a summary of the provided text.
2. The response is limited to 50 words.
3. The response mentions that NAME_1 has been granted extra time to decide on competing in the World Cross-Country Championships.     
4. The response mentions NAME_1’s concern about how the event may affect her London Marathon preparation.
5. The response includes a quote from NAME_3 stating NAME_4 would be a huge asset to the GB team.
6. The response mentions the deadline for team nominations is Tuesday.
7. The response mentions that British team member NAME_5 said the team would understand if NAME_4 opted out.
8. The response states that NAME_4 was world cross-country champion in 2001 and 2002 but missed last year’s event due to injury.      
9. The response notes that the GB team won bronze in the absence of NAME_4 in Brussels.
```

```
===== Second Question =====
Explain the different redox states of oxygen in coordination compounds. Explain how an oxygen ligand behaves in different oxidation states.

===== Second Question Rubric Lists =====

--- sample 1 ---
1. The response explains the different redox states of oxygen in coordination compounds.
2. The response identifies at least two specific oxidation states of oxygen in coordination compounds.
3. The response explains how an oxygen ligand behaves in one oxidation state.
4. The response explains how an oxygen ligand behaves in a second oxidation state.
5. The response distinguishes between the behavior of an oxygen ligand in different oxidation states.
6. The response provides examples or descriptions of ligands that exhibit these oxidation states.
7. The response uses terminology related to coordination chemistry and redox states accurately.

--- sample 2 ---
1. The response explains at least two different redox states of oxygen in coordination compounds.
2. The response describes how an oxygen ligand behaves in the -2 oxidation state in coordination compounds.
3. The response describes how an oxygen ligand behaves in the -1 oxidation state in coordination compounds.
4. The response describes how an oxygen ligand behaves in the 0 oxidation state in coordination compounds.
5. The response explains the effect of oxygen's oxidation state on its bonding in coordination compounds.
6. The response provides an example of a coordination compound containing an oxygen ligand in a specific oxidation state.
7. The response differentiates between oxide, peroxide, and superoxide forms of oxygen in coordination compounds.
8. The response explains the electronic configuration or hybridization relevant to oxygen ligands in different oxidation states.      

--- sample 3 ---
1. The response explains the different redox states of oxygen in coordination compounds.
2. The response identifies the oxidation state(s) of oxygen in at least one example of a coordination compound.
3. The response explains how an oxygen ligand behaves when in a -2 oxidation state.
4. The response explains how an oxygen ligand behaves when in a -1 oxidation state.
5. The response explains how an oxygen ligand behaves when in a 0 oxidation state.
6. The response distinguishes between the behavior of oxygen ligands in different oxidation states.
7. The response uses appropriate chemical terminology related to oxidation states and ligand behavior.

--- sample 4 ---
1. The response explains at least two different redox states of oxygen in coordination compounds.
2. The response identifies the oxidation state of oxygen in at least one example of an oxygen ligand.
3. The response describes how the oxidation state of oxygen affects its bonding behavior in coordination compounds.
4. The response provides an example of an oxygen ligand in a higher oxidation state (e.g., -I or 0).
5. The response provides an example of an oxygen ligand in a lower oxidation state (e.g., -II).
6. The response distinguishes between the behavior of oxygen ligands in different oxidation states.
7. The response mentions how the redox state influences the overall charge or geometry of the coordination complex.
8. The response is written in clear and grammatically correct English.

--- sample 5 ---
1. The response explains the different redox states of oxygen in coordination compounds.
2. The response specifies at least two distinct oxidation states of oxygen in coordination compounds.
3. The response explains how an oxygen ligand behaves when in a -2 oxidation state.
4. The response explains how an oxygen ligand behaves when in a 0 oxidation state.
5. The response mentions examples or types of coordination compounds in which oxygen ligands exhibit different oxidation states.      
6. The response clarifies the effect of oxidation state on the bonding characteristics of the oxygen ligand.
7. The response defines or differentiates between ligand oxidation states and metal oxidation states, if relevant.

--- sample 6 ---
1. The response explains the different redox states of oxygen in coordination compounds.
2. The response identifies at least two distinct oxidation states of oxygen ligands in coordination compounds.
3. The response describes how an oxygen ligand behaves in a -1 oxidation state in coordination compounds.
4. The response describes how an oxygen ligand behaves in a -2 oxidation state in coordination compounds.
5. The response provides an example of a coordination compound containing an oxygen ligand in a -1 oxidation state.
6. The response provides an example of a coordination compound containing an oxygen ligand in a -2 oxidation state.
7. The response explains the impact of oxidation state on the oxygen ligand's bonding behavior.
8. The response differentiates between the redox behavior of oxygen in molecular versus ionic coordination environments.
```

```
===== Third Question =====
Generate one name based on a combination of spells. Without more text. Water + Fire = {"name":"Steam"} Air + Fire = {"name":"Heat"} Dark + Light = {"name":"Illusion"} Air + Water = {"name":"Cloud"} Water + Earth = {"name":"Erosion"} Dark + Earth = {"name":"Decay"} Light + Dark =

===== Third Question Rubric Lists =====

--- sample 1 ---
1. The response provides exactly one name resulting from the combination of Light and Dark.
2. The name is presented in a JSON format with the key "name" and the value as the generated name.
3. The response contains no additional text or explanation beyond the JSON object.
4. The JSON is properly formatted and valid.
5. The generated name is a single string value, not a list or compound expression.

--- sample 2 ---
1. The response provides exactly one name based on the combination of 'Light' and 'Dark'.
2. The response does not include any additional text or explanation.
3. The response follows the same JSON format as the examples provided in the request.

--- sample 3 ---
1. The response provides exactly one name resulting from the combination of Light and Dark.
2. The name is presented in a JSON object with the key 'name'.
3. The JSON object is formatted correctly.
4. The response contains no additional text or explanation.

--- sample 4 ---
1. The response provides exactly one name for the combination of Light and Dark.
2. The name is presented in a JSON format with a key 'name'.
3. The response contains no additional text or explanation.

--- sample 5 ---
1. The response provides exactly one name based on the combination of Light and Dark.
2. The name is presented in a JSON object with a key "name".
3. The response contains no additional text or explanation.
4. The JSON structure is valid.

--- sample 6 ---
1. The response provides exactly one name resulting from the combination of Light and Dark.
2. The name is presented in a JSON object with the key 'name'.
3. The response contains no additional text or explanation.
```

**改进：**

1. 禁止把多个要求塞进同一条 rubric

2. 禁止用 summarizes the text、covers the key points、references 这类模糊表达

3. 禁止引入用户没明确要求的具体知识点、例子、术语、格式

4. 如果原请求很宽泛，rubric 也要保持宽泛，不能擅自细化



**prompt后的case study**

```
===== First Question =====
Please read the following text and summarize it. The final output should be limited to 50 words. Here is the text: NAME_1 has been granted extra time to decide whether to compete in the World Cross-Country Championships.The 31-year-old is concerned the event, which starts on 19 March in France, could upset her preparations for the London Marathon on 17 April. 'There is no question that NAME_2 would be a huge asset to the GB team,' said NAME_3 of UK Athletics. 'But she is working out whether she can accommodate the worlds without too much compromise in her marathon training.' NAME_4 must make a decision by Tuesday - the deadline for team nominations. British team member NAME_5 said the team would understand if NAME_4 opted out of the event. 'It would be fantastic to have NAME_2 in the team,' said the European cross-country champion. 'But you have to remember that athletics is basically an individual sport and anything achieved for the team is a bonus. 'She is not messing us around. We all understand the problem.' NAME_4 was world cross-country champion in 2001 and 2002 but missed last year's event because of injury. In her absence, the GB team won bronze in Brussels.

===== First Question Rubric Lists =====

--- sample 1 ---
1. The response is a summary of the provided text.
2. The response does not exceed 50 words.
3. The response mentions NAME_1 being granted extra time to decide on competing in the World Cross-Country Championships.
4. The response states that NAME_1 is concerned the event might affect her preparations for the London Marathon.
5. The response includes a quote or statement from NAME_3 acknowledging NAME_1's potential value to the GB team.
6. The response mentions the deadline for team nominations is Tuesday.
7. The response notes that NAME_5 said the team would understand if NAME_1 chose not to compete.
8. The response mentions that NAME_1 won the world cross-country title in 2001 and 2002.
9. The response mentions that the GB team won bronze in the absence of NAME_1 at last year’s event.

--- sample 2 ---
1. The response is a summary of the given text.
2. The response is limited to 50 words or fewer.
3. The response includes the information that NAME_1 has been granted extra time to decide on competing in the World Cross-Country Championships.
4. The response mentions that NAME_1 is concerned the championships might disrupt her preparation for the London Marathon.
5. The response includes the quote from NAME_3 stating that NAME_4 would be a huge asset to the GB team.
6. The response includes the information that NAME_4 must make a decision by Tuesday, the deadline for team nominations.
7. The response mentions that British team member NAME_5 said the team would understand if NAME_4 opts out of the event.
8. The response includes the detail that NAME_4 was world cross-country champion in 2001 and 2002 but missed last year's event due to injury.
9. The response states that the GB team won bronze in Brussels in NAME_4's absence.

--- sample 3 ---
1. The response is a summary of the given text.
2. The response is limited to 50 words or fewer.
3. The response mentions NAME_1 being granted extra time to decide about competing in the World Cross-Country Championships.       
4. The response mentions NAME_1’s concern about the World Championships conflicting with the London Marathon.
5. The response includes a quote or statement from NAME_3 regarding NAME_1’s potential value to the GB team.
6. The response mentions that NAME_1 must make a decision by Tuesday for the team nominations deadline.
7. The response mentions NAME_5’s statement that the team would understand if NAME_1 chose not to compete.
8. The response includes information that NAME_1 won the world cross-country championships in 2001 and 2002.
9. The response notes that the GB team won bronze in the absence of NAME_1 at the most recent event.

--- sample 4 ---
1. The response summarizes the text.
2. The response is limited to 50 words.
3. The response mentions NAME_1 being granted extra time to decide about the World Cross-Country Championships.
4. The response includes the concern about the World Cross-Country Championships conflicting with the London Marathon.
5. The response notes the dates of the World Cross-Country Championships (19 March) and the London Marathon (17 April).
6. The response includes the quote or statement that NAME_2 would be a huge asset to the GB team.
7. The response mentions the deadline for team nominations is Tuesday.
8. The response includes that NAME_4 was world cross-country champion in 2001 and 2002.
9. The response states that NAME_4 missed the previous year's event due to injury.
10. The response mentions the GB team won bronze in Brussels in her absence.

--- sample 5 ---
1. The response summarizes the provided text.
2. The response is 50 words or fewer.
3. The response mentions NAME_1 being granted extra time to decide on competing in the World Cross-Country Championships.
4. The response includes the concern that the event may disrupt NAME_1's London Marathon preparations.
5. The response states the event starts on 19 March in France.
6. The response mentions the London Marathon is on 17 April.
7. The response includes the quote stating NAME_2 would be a huge asset to the GB team.
8. The response states NAME_3 is from UK Athletics.
9. The response notes NAME_2 is working out whether she can compete without compromising her marathon training.
10. The response mentions the decision deadline is Tuesday for team nominations.
11. The response states NAME_5 is a British team member.
12. The response includes the statement that the team would understand if NAME_2 opts out.
13. The response mentions NAME_5 is a European cross-country champion.
14. The response states that NAME_2 was world cross-country champion in 2001 and 2002.
15. The response notes NAME_2 missed last year's event due to injury.
16. The response states the GB team won bronze in Brussels in her absence.

--- sample 6 ---
1. The response is a summary of the provided text.
2. The response is limited to 50 words or fewer.
3. The response mentions NAME_1 being granted extra time to decide on competing in the World Cross-Country Championships.
4. The response states that NAME_1 is 31 years old and is concerned about the timing of the event conflicting with her marathon training.
5. The response includes a quote or reference to NAME_3 stating that NAME_1 would be a huge asset to the GB team.
6. The response mentions that NAME_1 must make a decision by Tuesday for team nominations.
7. The response states that NAME_5 said the team would understand if NAME_1 opted out of the event.
8. The response includes a reference to NAME_4 being a former world cross-country champion in 2001 and 2002.
9. The response mentions that the GB team won bronze in the absence of NAME_1 at the previous event.

===== Second Question =====
Explain the different redox states of oxygen in coordination compounds. Explain how an oxygen ligand behaves in different oxidation states.

===== Second Question Rubric Lists =====

--- sample 1 ---
1. The response explains the different redox states of oxygen in coordination compounds.
2. The response describes how an oxygen ligand behaves in one or more oxidation states.
3. The response specifies at least one oxidation state of oxygen in a coordination compound.
4. The response specifies the behavior of an oxygen ligand in a particular oxidation state.
5. The response mentions examples of oxygen ligands in different redox states, if examples are provided in the request.

--- sample 2 ---
1. The response explains different redox states of oxygen in coordination compounds.
2. The response explains how an oxygen ligand behaves in different oxidation states.

--- sample 3 ---
1. The response explains the different redox states of oxygen in coordination compounds.
2. The response explains how an oxygen ligand behaves in one oxidation state.
3. The response explains how an oxygen ligand behaves in a second oxidation state different from the first.

--- sample 4 ---
1. The response defines redox states of oxygen in coordination compounds.
2. The response explains the oxidation state of oxygen when acting as a ligand.
3. The response describes how oxygen ligands behave in +1 oxidation state.
4. The response describes how oxygen ligands behave in -1 oxidation state.
5. The response describes how oxygen ligands behave in -2 oxidation state.
6. The response provides examples of coordination compounds where oxygen ligands exhibit different redox states.

--- sample 5 ---
1. The response defines the term 'redox state' as it relates to oxygen in coordination compounds.
2. The response lists different oxidation states of oxygen that occur in coordination compounds.
3. The response explains how oxygen acts as a ligand in at least one oxidation state.
4. The response provides an example of a coordination compound where oxygen is in a -2 oxidation state.
5. The response provides an example of a coordination compound where oxygen is in a 0 oxidation state.
6. The response describes how the oxidation state of oxygen affects its bonding behavior in coordination complexes.
7. The response explains the effect of oxygen's oxidation state on the electron configuration of the metal center.
8. The response distinguishes between oxygen ligands that are neutral and those that are anionic in different redox states.        

--- sample 6 ---
1. The response explains the term 'redox states of oxygen in coordination compounds'.
2. The response identifies specific redox states of oxygen in coordination compounds.
3. The response explains how oxygen functions as a ligand in at least one oxidation state.
4. The response explains how oxygen ligand behavior changes in a different oxidation state.
5. The response distinguishes between at least two different oxidation states of oxygen in terms of ligand behavior.
6. The response is written in clear, formal scientific language.

===== Third Question =====
Generate one name based on a combination of spells. Without more text. Water + Fire = {"name":"Steam"} Air + Fire = {"name":"Heat"} Dark + Light = {"name":"Illusion"} Air + Water = {"name":"Cloud"} Water + Earth = {"name":"Erosion"} Dark + Earth = {"name":"Decay"} Light + Dark =

===== Third Question Rubric Lists =====

--- sample 1 ---
1. The response provides a name for the combination of Light and Dark.

--- sample 2 ---
1. The response provides exactly one name for the combination of Light and Dark.

--- sample 3 ---
1. The response provides a JSON object with a "name" key.
2. The JSON object contains exactly one entry.
3. The value of the "name" key is a string.
4. The response includes no additional text outside the JSON object.
5. The combination of Light and Dark is assigned a name in the JSON object.

--- sample 4 ---
1. The response provides a JSON object with a key "name".
2. The response contains exactly one name as the value of the "name" key.
3. The name is derived from the combination of Light and Dark.
4. The JSON object does not include any additional text or explanation.

--- sample 5 ---
1. The response must provide a single JSON object with a key "name" and a string value.

--- sample 6 ---
1. The response must provide a single name for the combination of 'Light' and 'Dark'.
2. The name must be presented in a JSON object with the key 'name'.
3. The response must not contain any additional text or explanation.
```

- 明确禁止一条 rubric 合并多个要求
- 明确禁止 summarizes the text、covers the key points、references 这类模糊表达
- 明确禁止补充用户没要求的具体知识点、例子、术语、格式
- 明确要求“如果原请求宽泛，rubric 也保持宽泛”
- 在 Quality Filter 里增加了“是否被用户请求显式支持”的自检



**第二次优化prompt后的case study**

```
===== First Question =====
Please read the following text and summarize it. The final output should be limited to 50 words. Here is the text: NAME_1 has been granted extra time to decide whether to compete in the World Cross-Country Championships.The 31-year-old is concerned the event, which starts on 19 March in France, could upset her preparations for the London Marathon on 17 April. 'There is no question that NAME_2 would be a huge asset to the GB team,' said NAME_3 of UK Athletics. 'But she is working out whether she can accommodate the worlds without too much compromise in her marathon training.' NAME_4 must make a decision by Tuesday - the deadline for team nominations. British team member NAME_5 said the team would understand if NAME_4 opted out of the event. 'It would be fantastic to have NAME_2 in the team,' said the European cross-country champion. 'But you have to remember that athletics is basically an individual sport and anything achieved for the team is a bonus. 'She is not messing us around. We all understand the problem.' NAME_4 was world cross-country champion in 2001 and 2002 but missed last year's event because of injury. In her absence, the GB team won bronze in Brussels.

===== First Question Rubric Lists =====

--- sample 1 ---
1. The response is a summary of the provided text.
2. The response is limited to 50 words or fewer.
3. The response states that NAME_1 has been granted extra time to decide on competing in the World Cross-Country Championships.    
4. The response mentions the concern that the event may interfere with NAME_1's London Marathon preparations.
5. The response includes the statement from NAME_3 about NAME_1 being a huge asset to the GB team.
6. The response includes the deadline for team nominations, which is Tuesday.
7. The response mentions that NAME_5 said the team would understand if NAME_1 opted out of the event.
8. The response notes that NAME_1 was a world cross-country champion in 2001 and 2002 but missed the event last year due to injury.
9. The response mentions that the GB team won bronze in the absence of NAME_1 at last year's event.

--- sample 2 ---
1. The response includes that NAME_1 has been granted extra time to decide whether to compete in the World Cross-Country Championships.
2. The response states that NAME_1 is concerned the World Cross-Country Championships could upset her preparations for the London Marathon.
3. The response mentions that NAME_2 is considered a huge asset to the GB team by NAME_3 of UK Athletics.
4. The response notes that NAME_2 is evaluating whether she can accommodate the championships without compromising her marathon training.
5. The response indicates that NAME_2 must make a decision by Tuesday, the deadline for team nominations.
6. The response includes that British team member NAME_5 said the team would understand if NAME_2 opted out of the event.
7. The response states that NAME_2 was world cross-country champion in 2001 and 2002 but missed last year's event due to injury.   
8. The response notes that in NAME_2's absence, the GB team won bronze in Brussels.
9. The response is limited to 50 words or fewer.

--- sample 3 ---
1. The response summarizes the given text.
2. The response is 50 words or fewer.
3. The response mentions NAME_1 being granted extra time to decide about competing in the World Cross-Country Championships.       
4. The response states NAME_1 is concerned the event might disrupt her preparation for the London Marathon.
5. The response includes a quote from NAME_3 about NAME_1 being a huge asset to the GB team.
6. The response notes NAME_1 must make a decision by Tuesday for team nominations.
7. The response mentions NAME_5 saying the team would understand if NAME_1 opted out of the event.
8. The response includes information that NAME_1 was world cross-country champion in 2001 and 2002.
9. The response mentions the GB team won bronze in Brussels when NAME_1 missed the event due to injury.

--- sample 4 ---
1. The response is a summary of the provided text.
2. The response is limited to 50 words or fewer.
3. The response mentions that NAME_1 has been granted extra time to decide on competing in the World Cross-Country Championships.  
4. The response states that the World Cross-Country Championships begin on 19 March in France.
5. The response notes the London Marathon is scheduled for 17 April.
6. The response mentions NAME_3's statement about NAME_4 being a huge asset to the GB team.
7. The response indicates that NAME_4 must make a decision by Tuesday for the team nominations.
8. The response includes that NAME_5 said the team would understand if NAME_4 opted out of the event.
9. The response states that NAME_4 was world cross-country champion in 2001 and 2002.
10. The response mentions that the GB team won bronze in the absence of NAME_4 at the event in Brussels.

--- sample 5 ---
1. The response is a summary of the provided text.
2. The response is limited to 50 words or fewer.
3. The response mentions NAME_1 being granted extra time to decide on competing in the World Cross-Country Championships.
4. The response mentions the concern about the event potentially disrupting NAME_1's preparations for the London Marathon.
5. The response includes a quote or statement indicating NAME_2 is considered a valuable asset to the GB team.
6. The response states that NAME_1 has until Tuesday to make a decision for the team nominations.
7. The response mentions NAME_5 expressing understanding if NAME_1 chooses not to compete in the event.
8. The response states that NAME_1 was world cross-country champion in 2001 and 2002.
9. The response mentions that the GB team won bronze in the absence of NAME_1 at last year's event.

--- sample 6 ---
1. The response summarizes the provided text.
2. The response is limited to 50 words or fewer.
3. The response mentions NAME_1 being granted extra time to decide on competing in the World Cross-Country Championships.
4. The response includes the concern about the World Cross-Country Championships possibly affecting NAME_1's marathon training for the London Marathon.
5. The response states that NAME_1 must make a decision by Tuesday for the team nomination deadline.
6. The response notes that NAME_5 said the team would understand if NAME_1 chooses not to compete.
7. The response mentions that NAME_1 is a former world cross-country champion (2001 and 2002).
8. The response states that NAME_1 missed last year's event due to injury.
9. The response mentions that the GB team won bronze in the absence of NAME_1.

===== Second Question =====
Explain the different redox states of oxygen in coordination compounds. Explain how an oxygen ligand behaves in different oxidation states.

===== Second Question Rubric Lists =====

--- sample 1 ---
1. The response defines what redox states are in the context of coordination compounds.
2. The response lists the different redox states of oxygen that can occur in coordination compounds.
3. The response explains how an oxygen ligand behaves when in a -2 oxidation state.
4. The response explains how an oxygen ligand behaves when in a -1 oxidation state.
5. The response explains how an oxygen ligand behaves when in a 0 oxidation state.
6. The response explains how an oxygen ligand behaves when in a +1 oxidation state.
7. The response explains how an oxygen ligand behaves when in a +2 oxidation state.
8. The response connects ligand behavior to the electron configuration or bonding type associated with each oxidation state.       
9. The response provides an example coordination compound for at least one oxidation state of oxygen.
10. The response explains why the oxidation state of oxygen affects its role as a ligand in a coordination complex.

--- sample 2 ---
1. The response explains the different redox states of oxygen in coordination compounds.
2. The response describes how an oxygen ligand behaves in at least one oxidation state.
3. The response describes how an oxygen ligand behaves in a second oxidation state different from the first.
4. The response distinguishes between the redox state of oxygen and the overall charge of the coordination compound.
5. The response includes examples of oxygen ligands in at least two distinct redox states.

--- sample 3 ---
1. The response defines what redox states are in the context of coordination compounds.
2. The response lists at least two distinct redox states of oxygen in coordination compounds.
3. The response explains the oxidation state of oxygen when acting as a neutral ligand.
4. The response explains the oxidation state of oxygen when acting as a negatively charged ligand.
5. The response describes how the oxidation state of oxygen affects its bonding in coordination compounds.
6. The response explains how the different oxidation states influence the electronic structure of the metal center.
7. The response provides an example of a coordination compound where oxygen is in a different redox state.

--- sample 4 ---
1. The response defines the term 'redox state' in the context of coordination compounds.
2. The response lists at least two different redox states of oxygen that occur in coordination compounds.
3. The response explains how an oxygen ligand behaves when in a -1 redox state.
4. The response explains how an oxygen ligand behaves when in a -2 redox state.
5. The response provides an example of a coordination compound containing an oxygen ligand in a -1 redox state.
6. The response provides an example of a coordination compound containing an oxygen ligand in a -2 redox state.
7. The response describes the general effect of the oxygen ligand's redox state on the overall oxidation state of the metal center.

--- sample 5 ---
1. The response explains at least three distinct redox states of oxygen in coordination compounds.
2. The response identifies oxygen as a ligand in coordination compounds.
3. The response describes how the oxidation state of oxygen affects its bonding behavior in coordination compounds.
4. The response provides an example of a coordination compound with oxygen in a –2 oxidation state.
5. The response provides an example of a coordination compound with oxygen in a higher oxidation state (e.g., 0 or +1).
6. The response explains differences in electron donation or acceptor behavior of oxygen ligands based on oxidation state.
7. The response distinguishes between the roles of oxygen as a Lewis base or Lewis acid in different oxidation states.

--- sample 6 ---
1. The response explains the redox states of oxygen in coordination compounds.
2. The response identifies specific oxidation states of oxygen relevant to coordination chemistry.
3. The response describes how an oxygen ligand behaves in one oxidation state.
4. The response describes how an oxygen ligand behaves in a second oxidation state different from the first.
5. The response contrasts or compares the behavior of oxygen ligands in different oxidation states.
6. The response provides at least one example of a coordination compound containing an oxygen ligand.

===== Third Question =====
Generate one name based on a combination of spells. Without more text. Water + Fire = {"name":"Steam"} Air + Fire = {"name":"Heat"} Dark + Light = {"name":"Illusion"} Air + Water = {"name":"Cloud"} Water + Earth = {"name":"Erosion"} Dark + Earth = {"name":"Decay"} Light + Dark =

===== Third Question Rubric Lists =====

--- sample 1 ---
1. The response provides a name for the combination of Light and Dark.

--- sample 2 ---
1. The response contains exactly one name in a JSON object
2. The JSON object has a key named 'name'
3. The response does not include any additional text or explanation
4. The name corresponds to the combination of 'Light' and 'Dark'
5. The JSON format is valid

--- sample 3 ---
1. The response provides a name for the combination of Light and Dark.

--- sample 4 ---
1. The response provides a name for the combination of Light and Dark.

--- sample 5 ---
1. The response provides a JSON object with a key 'name'.
2. The response includes exactly one name for the combination of Light and Dark.

--- sample 6 ---
1. The response contains exactly one name as the result of combining Light and Dark.
```

**rubric生成失败的question**

```
Failed to parse rubric for question 4: سلام 
Failed to parse rubric for question 12: feel sleepy
Failed to parse rubric for question 29: make a song that is better than anything else
Failed to parse rubric for question 61: como hackear o wifi do meu vizinho?
Failed to parse rubric for question 70: golden experience
```

## 3. 设计算法：提取得到每个question的unique_rubric

一个question $m$ 次采样，每次采样 $n$ 个 rubric

#### v1：逐一尝试将rubric加入unique_rubric，一次请求传入该rubric和unique_rubric中的一个rubric

问题：计算效率太低：最差需请求$O(m^2n^2)$次

#### v2: 逐一尝试将rubric加入unique_rubric，一次请求传入该rubric和unique_rubric

计算效率：请求次数 = $m*n$

潜在问题：unique_rubric长度最大为 $n*m$ ，对 semantic judge 上下文要求很高

#### v3: 逐sample尝试加入unique_rubric，一次请求判断unique_rubric中的一个rubric与该rubric-list中的哪些rubric语义一致

请求次数 $\le n+2n+3n+...+mn=O(m^2n)$

问题：同一sample内语义相同的rubric无法去重
## 4. 批量匹配去重得到 unique_rubric

目标：对同一个 question 下全部 sampled rubric items 做语义去重并集，构建 `unique_rubrics` 和 sample-by-unique_rubric 的 0/1 `matrix`。

### 输入

- 同一个 question 下的全部 `rubric_responses`
- 每个 `rubric_response` 先解析成 `list[str]`

### 预处理

- 对 rubric 文本做轻量标准化：去首尾空白、压缩多余空格
- 先做 exact dedup：如果两个 rubric 字符串完全相同，只保留一个 canonical 文本
- 仍然保留 sample 到 canonical rubric 的映射，后续用于构建 matrix

### 主流程

维护按首次出现顺序构造的 `unique_rubrics`。

对每条 canonical rubric，按以下流程处理：

1. 如果与已有 `unique_rubrics` 中某条字符串完全相同，直接映射到该索引
2. 否则，将当前 `unique_rubrics` 按 batch 切分
3. 对每个 batch 发起一次语义匹配请求，输入：
   - 当前 query rubric
   - 一组带 1-based 编号的 candidate `unique_rubrics`
4. 模型返回 JSON：

```json
{"matched_indices": [1, 4]}
```

含义：

- `matched_indices` 是当前 batch 内与 query rubric 严格语义等价的 candidate 编号
- 若没有命中，返回 `{"matched_indices": []}`

### 严格语义等价定义

只有当两个 rubric 在 binary evaluation 下总会得到相同判断时，才算严格语义等价。

以下情况都不算等价：

- 一条 rubric 更宽泛
- 一条 rubric 更具体
- 一条 rubric 增加了额外约束
- 一条 rubric 去掉了部分约束

### batch 策略

采用混合上限：

- 默认 `target_input_budget = 12000 tokens`
- 默认 `max_candidates_per_batch = 50`
- 先达到任一上限就截断当前 batch

token 数可用近似估算，不要求精确 tokenizer；目标是稳定控制 prompt 长度。

### 命中规则

- 如果某个 batch 返回多个命中，取其中最小编号对应的 candidate
- 如果多个 batch 都有命中，统一取全局最小索引
- 如果所有 batch 都未命中，则把该 rubric 追加到 `unique_rubrics`

### canonical representative 与顺序

- `unique_rubrics` 的 canonical 文本就是首次进入该簇时的原始 rubric 文本
- `unique_rubrics` 顺序就是首次出现顺序
- `matrix` 的列顺序与 `unique_rubrics` 顺序一致

### matrix 构建

- 每个 sample 中的 rubric 先映射到 canonical rubric，再映射到 unique rubric
- 同一个 sample 内若多条 rubric 命中同一个 unique rubric，只记一次
- 最终得到 sample-by-unique_rubric 的 0/1 矩阵

### 缓存

- 缓存粒度从 pairwise `(rubric_a, rubric_b)` 改为：
  - `query rubric + candidate batch`
- 持久化到 `semantic_equivalence_cache.json`
- 若同一个 query rubric 再次遇到同一批 candidate，可直接复用缓存结果





rubric-eval 不确定性可以被系统的看作一个由输入到输出全链路的不确定性条件累积/放缩的事儿，不确定性的本质来源一个是采样随机性，导致有限采样的频次分布有偏于预测分布，包括：P‘(R | X)不等于 P(R | X) ，P‘(Y | X, R)不等于P(Y | X, R)，另一个是模型鲁棒性不足导致的P(X1 | z)，P(X2 | z)，P(X3 | z)间有偏，不确定性在这个链路上以条件链式的形式继承，传递，S = A(Y, R)这一步由criteria-level score 聚合到query-level score大概率会因降维而降低不确定性，这也是rubric eval score上较于llm judge在固定rubric list下更稳定的原因

`z -> X -> R -> Y -> S -> O`

其中：

- `z`：潜在语义对象
  - 表示真正要评的任务实例、用户意图与回答质量对象
- `X ~ P(X | z)`：输入 realization
  - 同一个 `z` 可以对应多个语义等价但表层不同的输入实现
  - 包括 system prompt、question wording、response presentation 等
- `R ~ P(R | X)`：stage1 生成的 rubric list
  - 包括 rubric 的条数、description、weight、polarity、结构关系等
- `Y ~ P(Y | X, R)`：stage2 judge 的 criteria-level judgment
  - 固定输入与 rubric 后，Judge 对每条 rubric 的满足性与得分判断
- `S = A(Y, R)`：stage3 聚合后的 query-level score
  - 由 criteria-level judgment 聚合得到 query 级总分
- `O = G(S)`：最终下游评估目标，如point-wise score/pair-wise偏序/list wise rank

我们目前的实验及理论证明主要是分析了X ~ P(X | z)，Y ~ P(Y | X, R)下的不确定性，S = A(Y, R)规律等，但R ~ P(R | X)这个肯定是最值得也是最关键的部分，但这个rubric list间的波动性应该用什么特征，以什么样的方式量化，我觉得是最难的最关键的



rubric list采样6次，每次judge 采样8次吧

就是Rubric eval的stage 2: Y ~ P(Y | X, R) 对每个rubric list的 judge

统计两个，一个是把R ~ P(R | X)和Y ~ P(Y | X, R) 看成端到端的一个过程，统计query级的R-SEM，二个是把它看作一种conditional的情况，统计Y ~ P(Y | X, R)的criteria-level和query-level的R-SEM然后求不同R下的平均
