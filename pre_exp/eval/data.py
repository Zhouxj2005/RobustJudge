from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from .config import DATASET_DIR, FIRST_N


def load_streaming_dataset(base_dir: Path = DATASET_DIR) -> DatasetDict:
    return load_dataset(
        "parquet",
        data_files={"train": str(base_dir / "**/*.parquet")},
        streaming=True,
    )


def load_first_n_dataset(n: int = FIRST_N) -> Dataset:
    dataset = load_streaming_dataset()
    first_n_list = list(islice(dataset["train"], n))
    return Dataset.from_list(first_n_list)


def get_query_text(item: dict[str, Any]) -> str:
    prompt = item["prompt"]
    if isinstance(prompt, list):
        return prompt[0]["content"]
    return prompt


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
