from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import BASE_NUM_SAMPLES
from .data import get_query_text, load_json, save_json


def load_prompt_templates(prompt_path: Path) -> dict[str, Any]:
    return load_json(prompt_path)


def run_or_load_generation(first_n, gen_models, model_client, output_path: Path):
    if output_path.exists():
        print("成功加载")
        return load_json(output_path)

    print("正在生成...")
    responses: list[dict[str, str]] = []
    for item in first_n:
        responses.append({})
        query = get_query_text(item)
        for gen_model in gen_models:
            generated, _ = model_client.calc(query, model=gen_model, n=1)
            responses[-1][gen_model] = generated[0]
    save_json(responses, output_path)
    return responses


def get_score(query, response, rubric_list, judge_model, n, model_client, prompt_templates):
    template = prompt_templates["list-grader-template"]
    prompt = template.replace("{{QUERY}}", query).replace("{{RESPONSE}}", response).replace("{{RUBRIC}}", rubric_list)
    results, _ = model_client.calc(prompt, model=judge_model, n=n)
    return [json.loads(item)["scoring_details"] for item in results]


def run_or_load_scoring(
    first_n,
    responses,
    gen_models,
    judge_models,
    model_client,
    prompt_templates,
    output_path: Path,
    n_samples: int = BASE_NUM_SAMPLES,
):
    if output_path.exists():
        print("成功加载results")
        return load_json(output_path)

    print("未找到现有results，将重新计算")
    result: dict[str, dict[str, dict[str, list]]] = {}
    for i, item in enumerate(first_n):
        key = str(i)
        result[key] = {}
        query = get_query_text(item)
        rubric_str = json.dumps(item["Rubrics"], ensure_ascii=False)

        for gen_model in gen_models:
            result[key][gen_model] = {}
            response = responses[i][gen_model]
            for judge_model in judge_models:
                result[key][gen_model][judge_model] = get_score(
                    query=query,
                    response=response,
                    rubric_list=rubric_str,
                    judge_model=judge_model,
                    n=n_samples,
                    model_client=model_client,
                    prompt_templates=prompt_templates,
                )
    save_json(result, output_path)
    return result


def _extract_score(rubric):
    invalid_values = [None, np.nan]
    if "score" in rubric and rubric["score"] not in invalid_values:
        return float(rubric["score"])
    if (
        "is_met" in rubric
        and "weight" in rubric
        and rubric["is_met"] not in invalid_values
        and rubric["weight"] not in invalid_values
    ):
        return float(rubric["weight"]) if str(rubric["is_met"]).lower() == "true" else 0.0
    return np.nan
