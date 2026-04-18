from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "RubricHub_v1"
FIGURE_DIR = ROOT_DIR / "figures"


@dataclass(frozen=True)
class ExperimentPaths:
    prompt: Path = ROOT_DIR / "prompt.json"
    model_res: Path = ROOT_DIR / "model_res.json"
    result: Path = ROOT_DIR / "result.json"
    ground_truth: Path = ROOT_DIR / "ground_truth.json"
    prompt_variant_result: Path = ROOT_DIR / "exp2_result.json"
    stability_item: Path = ROOT_DIR / "stability_results.json"
    stability_query: Path = ROOT_DIR / "stability_results_conversation.json"
    stability_item_v2: Path = ROOT_DIR / "stability_results_v2.json"
    stability_query_v2: Path = ROOT_DIR / "stability_results_conversation_v2.json"


PATHS = ExperimentPaths()

GEN_MODELS = ["qwen2.5-72b", "gpt-oss-120b", "qwen3-235b"]
JUDGE_MODELS = ["deepseek-v3.2", "qwen2.5-7b", "qwen3-32b", "gpt-4o"]
PROMPT_VARIANT_JUDGE_MODELS = ["qwen3-32b"]

FIRST_N = 100
BASE_NUM_SAMPLES = 16
BASE_MIN_VALID_SAMPLES = 12
PROMPT_VARIANT_TOTAL_SAMPLES = 20
PROMPT_VARIANT_MIN_VALID_SAMPLES = 16
BOOTSTRAP_B = 50


def ensure_output_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

