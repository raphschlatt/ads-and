from __future__ import annotations

from pathlib import Path

import pandas as pd

from author_name_disambiguation.common.config import find_project_root, resolve_existing_path
from author_name_disambiguation.common.io_schema import MENTION_REQUIRED_COLUMNS, validate_columns, save_parquet
from author_name_disambiguation.data.build_blocks import create_block_key
from author_name_disambiguation.data.build_mentions import make_mention_id


def load_lspo_raw(parquet_path: str | Path, h5_path: str | Path | None = None) -> pd.DataFrame:
    project_root = find_project_root(Path.cwd())
    checked = []

    p_parquet = resolve_existing_path(parquet_path, project_root=project_root)
    checked.append(str(parquet_path))
    if p_parquet is not None:
        checked.append(str(p_parquet))
        return pd.read_parquet(p_parquet)

    if h5_path is not None:
        p_h5 = resolve_existing_path(h5_path, project_root=project_root)
        checked.append(str(h5_path))
        if p_h5 is not None:
            checked.append(str(p_h5))
            return pd.read_hdf(p_h5)

    raise FileNotFoundError(f"LSPO raw file not found. Checked: {checked}")


def normalize_lspo_mentions(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy().reset_index(drop=True)

    # LSPO row is already one author mention.
    df["bibcode"] = [f"LSPO:{i:07d}" for i in range(len(df))]
    df["author_idx"] = 0
    df["author_raw"] = df.get("author", "").fillna("").astype(str)
    df["title"] = df.get("title", "").fillna("").astype(str)
    df["abstract"] = df.get("abstract", "").fillna("").astype(str)
    df["year"] = None
    df["source_type"] = "lspo"

    if "block" in df.columns:
        df["block_key"] = df["block"].fillna("").astype(str)
    else:
        df["block_key"] = df["author_raw"].map(create_block_key)

    df["mention_id"] = [make_mention_id(b, 0) for b in df["bibcode"]]

    out_cols = [
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
    out = df[out_cols].copy()
    out["orcid"] = df.get("@path")
    out["aff"] = df.get("aff")

    validate_columns(out, MENTION_REQUIRED_COLUMNS, "lspo_mentions")
    return out


def prepare_lspo_mentions(
    parquet_path: str | Path,
    output_path: str | Path,
    h5_path: str | Path | None = None,
) -> pd.DataFrame:
    raw = load_lspo_raw(parquet_path=parquet_path, h5_path=h5_path)
    mentions = normalize_lspo_mentions(raw)
    save_parquet(mentions, output_path, index=False)
    return mentions
