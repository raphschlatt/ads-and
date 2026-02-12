from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

MENTION_REQUIRED_COLUMNS = [
    "mention_id",
    "bibcode",
    "author_idx",
    "author_raw",
    "title",
    "abstract",
    "year",
    "source_type",
    "block_key",
]

PAIR_REQUIRED_COLUMNS = [
    "pair_id",
    "mention_id_1",
    "mention_id_2",
    "block_key",
    "split",
]

PAIR_SCORE_REQUIRED_COLUMNS = [
    "pair_id",
    "mention_id_1",
    "mention_id_2",
    "block_key",
    "cosine_sim",
    "distance",
]

CLUSTER_REQUIRED_COLUMNS = ["mention_id", "block_key", "author_uid"]


def validate_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {context}: {missing}")


def save_parquet(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=index)
    return p


def read_parquet(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_parquet(p)
