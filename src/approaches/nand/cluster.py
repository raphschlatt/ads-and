from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from src.common.io_schema import CLUSTER_REQUIRED_COLUMNS, validate_columns, save_parquet

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SURNAME_PARTICLES = {
    "da",
    "de",
    "del",
    "della",
    "der",
    "di",
    "du",
    "la",
    "le",
    "van",
    "von",
}


def _ascii_fold(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text or "")
    return norm.encode("ascii", "ignore").decode("ascii").lower().strip()


def _clean_token(token: str) -> str:
    return _NON_ALNUM.sub("", token.lower())


def _is_initial_token(token: str) -> bool:
    t = _clean_token(token)
    return 0 < len(t) <= 2


def _split_name_parts(author_raw: str) -> tuple[list[str], list[str]]:
    raw = _ascii_fold(author_raw)
    if not raw:
        return [], []

    if "," in raw:
        left, right = raw.split(",", 1)
        surname_tokens = [_clean_token(t) for t in left.split() if _clean_token(t)]
        given_tokens = [_clean_token(t) for t in right.split() if _clean_token(t)]
        return surname_tokens, given_tokens

    parts = [_clean_token(t) for t in raw.split() if _clean_token(t)]
    if len(parts) <= 1:
        return parts, parts

    trailing_initial_count = 0
    for token in reversed(parts):
        if _is_initial_token(token):
            trailing_initial_count += 1
        else:
            break

    if 0 < trailing_initial_count < len(parts):
        surname_tokens = parts[:-trailing_initial_count]
        given_tokens = parts[-trailing_initial_count:]
        return surname_tokens, given_tokens

    # Fallback "Firstname Lastname".
    return [parts[-1]], [parts[0]]


def _given_name_token(author_raw: str) -> str:
    _, given_tokens = _split_name_parts(author_raw)
    return given_tokens[0] if given_tokens else ""


def _surname_token(author_raw: str) -> str:
    surname_tokens, _ = _split_name_parts(author_raw)
    if not surname_tokens:
        return ""
    if len(surname_tokens) > 1 and surname_tokens[0] in _SURNAME_PARTICLES:
        surname_tokens = surname_tokens[1:]
    return surname_tokens[-1] if surname_tokens else ""

def _name_conflict(a: str, b: str) -> bool:
    ga = _given_name_token(a)
    gb = _given_name_token(b)
    sa = _surname_token(a)
    sb = _surname_token(b)

    if sa and sb and sa != sb and len(sa) > 2 and len(sb) > 2:
        return True

    if not ga or not gb:
        return False

    if ga == gb:
        return False

    if _is_initial_token(ga) or _is_initial_token(gb):
        return ga[0] != gb[0]

    if ga[0] != gb[0]:
        return True

    if ga.startswith(gb) or gb.startswith(ga):
        return False

    return ga != gb


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
    constraint_mode = str(constraints.get("constraint_mode", "soft")).lower()
    name_conflict_min_distance = float(constraints.get("name_conflict_min_distance", 1.0))
    year_gap_min_distance = float(constraints.get("year_gap_min_distance", 1.0))

    authors = block_mentions["author_raw"].fillna("").astype(str).tolist()
    years = block_mentions["year"].tolist()

    n = len(block_mentions)
    for i in range(n):
        for j in range(i + 1, n):
            force_name = False
            force_year = False
            if enforce_name_conflict and _name_conflict(authors[i], authors[j]):
                force_name = True
            yi, yj = years[i], years[j]
            if (yi is not None and yj is not None) and not (pd.isna(yi) or pd.isna(yj)):
                if abs(int(yi) - int(yj)) > max_year_gap:
                    force_year = True

            if not (force_name or force_year):
                continue

            if constraint_mode == "hard":
                out[i, j] = 1.0
                out[j, i] = 1.0
                continue

            if force_name:
                out[i, j] = max(out[i, j], name_conflict_min_distance)
                out[j, i] = max(out[j, i], name_conflict_min_distance)
            if force_year:
                out[i, j] = max(out[i, j], year_gap_min_distance)
                out[j, i] = max(out[j, i], year_gap_min_distance)

    np.fill_diagonal(out, 0.0)
    return out


def resolve_dbscan_eps(cluster_config: Dict[str, Any], cosine_threshold: float | None = None) -> tuple[float, Dict[str, Any]]:
    eps_mode = str(cluster_config.get("eps_mode", "fixed")).lower()
    fixed_eps = float(cluster_config.get("eps", 0.35))
    eps_min = float(cluster_config.get("eps_min", 0.0))
    eps_max = float(cluster_config.get("eps_max", 1.0))

    if eps_mode == "from_threshold" and cosine_threshold is not None:
        raw_eps = 1.0 - float(cosine_threshold)
        source = "from_threshold"
    else:
        raw_eps = fixed_eps
        source = "fixed"

    resolved = float(np.clip(raw_eps, eps_min, eps_max))
    meta = {
        "eps_mode": eps_mode,
        "source": source,
        "cosine_threshold": cosine_threshold,
        "raw_eps": float(raw_eps),
        "eps_min": eps_min,
        "eps_max": eps_max,
        "resolved_eps": resolved,
    }
    return resolved, meta


def cluster_blockwise_dbscan(
    mentions: pd.DataFrame,
    pair_scores: pd.DataFrame,
    cluster_config: Dict,
    output_path: str | Path | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    eps = float(cluster_config.get("eps", 0.35))
    min_samples = int(cluster_config.get("min_samples", 1))
    metric = str(cluster_config.get("metric", "precomputed"))
    constraints = cluster_config.get("constraints", {})

    rows = []

    grouped = mentions.groupby("block_key", sort=False)
    iterator = grouped
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(grouped, total=int(mentions["block_key"].nunique()), desc="Cluster blocks", leave=False)
        except Exception:
            pass

    for block_key, block_mentions in iterator:
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
