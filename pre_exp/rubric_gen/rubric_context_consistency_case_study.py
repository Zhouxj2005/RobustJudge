from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .rubric_context_consistency_analysis import normalize_rubric_score
except ImportError:
    from rubric_context_consistency_analysis import normalize_rubric_score

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS_PATH = BASE_DIR / "rubric_context_consistency_metrics.json"
DEFAULT_MATCH_PATH = BASE_DIR / "rubric_matrix_list_match.json"
DEFAULT_RUBRIC_PATH = BASE_DIR / "rubric_with_dedup_oriented_prompt.json"
DEFAULT_RESULT_PATH = BASE_DIR / "generated_rubric_judge_result.json"
DEFAULT_RESPONSE_PATH = BASE_DIR.parent.parent / "model_res.json"
DEFAULT_OUTPUT_MD = BASE_DIR / "rubric_context_consistency_top10_case_study.md"
DEFAULT_TOP_K = 10


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_rubric_response(text: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def format_score_list(scores: list[float]) -> str:
    return "[" + ", ".join(f"{score:.3f}" for score in scores) + "]"


def collect_context_rubric_blocks(
    q_idx: int,
    gen_model: str,
    unique_rubric_idx: int,
    sample_idx: int,
    sample_mapping: list[int],
    rubric_item: dict[str, Any],
    judge_result: dict[str, Any],
) -> list[dict[str, Any]]:
    local_positions = [idx for idx, mapped_idx in enumerate(sample_mapping) if mapped_idx == unique_rubric_idx]
    local_rubrics = parse_rubric_response(rubric_item["rubric_responses"][sample_idx - 1])
    trials = judge_result[str(q_idx)][gen_model].get(str(sample_idx), [])
    blocks = []

    for local_idx in local_positions:
        local_rubric_text = ""
        if local_idx < len(local_rubrics):
            local_rubric_text = str(local_rubrics[local_idx].get("criterion", ""))

        per_trial_scores: list[float] = []
        per_trial_evidences: list[str] = []
        for trial_idx, trial in enumerate(trials, start=1):
            if local_idx >= len(trial):
                continue
            rubric = trial[local_idx]
            normalized_score = normalize_rubric_score(rubric)
            if normalized_score != normalized_score:
                continue
            per_trial_scores.append(float(normalized_score))
            evidence = rubric.get("evidence")
            if isinstance(evidence, str) and evidence.strip():
                per_trial_evidences.append(f"trial {trial_idx}: {evidence.strip()}")
            else:
                per_trial_evidences.append(f"trial {trial_idx}: ")

        blocks.append(
            {
                "sample_idx": sample_idx,
                "local_idx": local_idx,
                "local_rubric_text": local_rubric_text,
                "scores": per_trial_scores,
                "evidences": per_trial_evidences,
            }
        )

    return blocks


def generate_case_study_markdown(
    metrics: dict[str, Any],
    match_data: list[dict[str, Any]],
    rubric_data: list[dict[str, Any]],
    judge_result: dict[str, Any],
    responses: list[dict[str, str]],
    top_k: int,
) -> str:
    records = metrics.get("records", [])[:top_k]
    match_lookup = {int(item["question_index"]): item for item in match_data}
    rubric_lookup = {int(item["question_index"]): item for item in rubric_data}

    lines: list[str] = [
        "# Top T_obs Case Studies",
        "",
        f"Generated from `{DEFAULT_METRICS_PATH.name}`.",
        "",
        f"Top {len(records)} `(question, gen_model, unique_rubric)` records sorted by `T_obs` descending.",
        f"Current consistency config: `required_samples_per_context = {metrics.get('config', {}).get('required_samples_per_context')}`",
        "",
    ]

    for rank, record in enumerate(records, start=1):
        q_idx = int(record["question_index"])
        gen_model = str(record["gen_model"])
        unique_rubric_idx = int(record["unique_rubric_index"])
        match_item = match_lookup[q_idx]
        rubric_item = rubric_lookup[q_idx]
        response_text = responses[q_idx][gen_model]
        question_text = rubric_item["question"]

        context_rubric_blocks = []
        for context_detail in record["context_details"]:
            sample_idx = int(context_detail["sample_idx"])
            blocks = collect_context_rubric_blocks(
                q_idx=q_idx,
                gen_model=gen_model,
                unique_rubric_idx=unique_rubric_idx,
                sample_idx=sample_idx,
                sample_mapping=match_item["sample_match_indices"][sample_idx - 1],
                rubric_item=rubric_item,
                judge_result=judge_result,
            )
            context_rubric_blocks.extend(blocks)

        lines.extend(
            [
                f"## Rank {rank}: q={q_idx}, gen={gen_model}, u={unique_rubric_idx}",
                "",
                "### Unique Rubric",
                record["unique_rubric_criterion"],
                "",
                "### Question",
                question_text,
                "",
                "### Response",
                "```text",
                response_text.strip(),
                "```",
                "",
                "### Matched Rubrics",
            ]
        )

        for block in context_rubric_blocks:
            lines.append(f"#### Sample {block['sample_idx']} / Local Rubric {block['local_idx']}")
            lines.append("")
            lines.append("Rubric:")
            lines.append(block["local_rubric_text"])
            lines.append("")
            lines.append(f"Score list: `{format_score_list(block['scores'])}`")
            lines.append("")
            lines.append("Judge evidence:")
            for evidence in block["evidences"]:
                lines.append(f"- {evidence}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown case studies for the top T_obs rubric-context consistency records.")
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--match-path", type=Path, default=DEFAULT_MATCH_PATH)
    parser.add_argument("--rubric-path", type=Path, default=DEFAULT_RUBRIC_PATH)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH)
    parser.add_argument("--response-path", type=Path, default=DEFAULT_RESPONSE_PATH)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    markdown = generate_case_study_markdown(
        metrics=load_json(args.metrics_path),
        match_data=load_json(args.match_path),
        rubric_data=load_json(args.rubric_path),
        judge_result=load_json(args.result_path),
        responses=load_json(args.response_path),
        top_k=args.top_k,
    )
    args.output_md.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
