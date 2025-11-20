from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ProblemRecord:
    problem_id: str
    prompt: str


def load_problem(dataset_path: Path, problem_id: Optional[str] = None) -> ProblemRecord:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    df = pd.read_parquet(dataset_path)

    if problem_id is not None:
        subset = df[df["ID"] == problem_id]
        if subset.empty:
            raise ValueError(f"Problem ID {problem_id} not found in dataset.")
        row = subset.iloc[0]
    else:
        row = df.sample(1, random_state=random.randint(0, 9999)).iloc[0]

    return ProblemRecord(problem_id=row["ID"], prompt=row["Problem"])

