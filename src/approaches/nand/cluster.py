from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from src.common.io_schema import CLUSTER_REQUIRED_COLUMNS, validate_columns, save_parquet


def _first_name_token(author_raw: str) -> str:
    text = (author_raw or "").strip().lower()
    if not text:
        return ""
    if "," in text:
        _, right = text.split(",", 1)
        token = right.strip().split()[0] if right.strip() else ""
    else:
        token = text.split()[0]
    token = re.sub(r"[^a-z0-9]", "", token)
    return token


def _name_conflict(a: str, b: str) -> bool:
    ta = _first_name_token(a)
    tb = _first_name_token(b)
    if not ta or not tb:
        return False
    # Ignore one-letter initials; enforce conflict mainly for explicit differing names.
    if len(ta) == 1 or len(tb) == 1:
        return False
    return ta != tb


def _build_distance_matrix(block_mentions: pd.DataFrame, block_scores: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
    mention_ids = block_mentions["mention_id"].astype(str).tolist()
    n = len(mention_ids)
    idx = {m: i for i, m in enumerate(mention_ids)}

    dist = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(dist, 0.0)

    for row in block_scores.itertuples(index=False):
        i = idx.get(str(row.mention_id_1))
        j = idx.get(str(row.mention_id_2))
        if i is None or j is None:
            continue
        d = float(row.distance)
        dist[i, j] = d
        dist[j, i] = d

    return dist, idx


def _apply_constraints(dist: np.ndarray, block_mentions: pd.DataFrame, constraints: Dict) -> np.ndarray:
    if not constraints or not constraints.get("enabled", False):
        return dist

    out = dist.copy()
    max_year_gap = int(constraints.get("max_year_gap", 30))
    enforce_name_conflict = bool(constraints.get("enforce_name_conflict", True))

    authors = block_mentions["author_raw"].fillna("").astype(str).tolist()
    years = block_mentions["year"].tolist()

    n = len(block_mentions)
    for i in range(n):
        for j in range(i + 1, n):
            forced = False
            if enforce_name_conflict and _name_conflict(authors[i], authors[j]):
                forced = True
            yi, yj = years[i], years[j]
            if (yi is not None and yj is not None) and not (pd.isna(yi) or pd.isna(yj)):
                if abs(int(yi) - int(yj)) > max_year_gap:
                    forced = True
            if forced:
                out[i, j] = 1.0
                out[j, i] = 1.0

    np.fill_diagonal(out, 0.0)
    return out


def cluster_blockwise_dbscan(
    mentions: pd.DataFrame,
    pair_scores: pd.DataFrame,
    cluster_config: Dict,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    eps = float(cluster_config.get("eps", 0.35))
    min_samples = int(cluster_config.get("min_samples", 1))
    metric = str(cluster_config.get("metric", "precomputed"))
    constraints = cluster_config.get("constraints", {})

    rows = []

    for block_key, block_mentions in mentions.groupby("block_key", sort=False):
        block_mentions = block_mentions.reset_index(drop=True)
        block_scores = pair_scores[pair_scores["block_key"] == block_key]

        if len(block_mentions) == 1:
            m = str(block_mentions.iloc[0]["mention_id"])
            rows.append({"mention_id": m, "block_key": str(block_key), "author_uid": f"{block_key}::0"})
            continue

        dist, idx = _build_distance_matrix(block_mentions, block_scores)
        dist = _apply_constraints(dist, block_mentions, constraints)

        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = model.fit_predict(dist)

        for mention_id, label in zip(block_mentions["mention_id"].astype(str).tolist(), labels.tolist()):
            uid = f"{block_key}::{int(label)}"
            rows.append({"mention_id": mention_id, "block_key": str(block_key), "author_uid": uid})

    out = pd.DataFrame(rows)
    validate_columns(out, CLUSTER_REQUIRED_COLUMNS, "clusters")

    if output_path is not None:
        save_parquet(out, output_path, index=False)

    return out
