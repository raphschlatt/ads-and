from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

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


def validate_pair_score_ranges(df: pd.DataFrame) -> dict[str, Any]:
    validate_columns(df, ["cosine_sim", "distance"], "pair_scores")

    cosine = pd.to_numeric(df["cosine_sim"], errors="coerce")
    distance = pd.to_numeric(df["distance"], errors="coerce")

    cosine_non_finite = int(cosine.isna().sum())
    distance_non_finite = int(distance.isna().sum())

    cosine_finite = cosine[cosine.notna()]
    distance_finite = distance[distance.notna()]

    cosine_out_of_range = int(((cosine_finite < -1.0) | (cosine_finite > 1.0)).sum())
    negative_distance = int((distance_finite < 0.0).sum())
    distance_above_max = int((distance_finite > 2.0).sum())

    return {
        "pair_score_range_ok": bool(
            cosine_non_finite == 0
            and distance_non_finite == 0
            and cosine_out_of_range == 0
            and negative_distance == 0
            and distance_above_max == 0
        ),
        "cosine_min": float(cosine_finite.min()) if len(cosine_finite) else None,
        "cosine_max": float(cosine_finite.max()) if len(cosine_finite) else None,
        "distance_min": float(distance_finite.min()) if len(distance_finite) else None,
        "distance_max": float(distance_finite.max()) if len(distance_finite) else None,
        "cosine_non_finite_count": cosine_non_finite,
        "distance_non_finite_count": distance_non_finite,
        "cosine_out_of_range_count": cosine_out_of_range,
        "negative_distance_count": negative_distance,
        "distance_above_max_count": distance_above_max,
    }


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
