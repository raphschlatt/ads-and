from __future__ import annotations

from pathlib import Path

import pandas as pd

from author_name_disambiguation.common.io_schema import MENTION_REQUIRED_COLUMNS, validate_columns, save_parquet
from author_name_disambiguation.data.build_blocks import create_block_key
from author_name_disambiguation.data.build_mentions import make_mention_id


def load_lspo_raw(parquet_path: str | Path, h5_path: str | Path | None = None) -> pd.DataFrame:
    parquet_candidate = Path(parquet_path).expanduser().resolve()
    if parquet_candidate.exists():
        return pd.read_parquet(parquet_candidate)

    checked = [str(parquet_candidate)]
    if h5_path is not None:
        h5_candidate = Path(h5_path).expanduser().resolve()
        checked.append(str(h5_candidate))
        if h5_candidate.exists():
            return pd.read_hdf(h5_candidate)

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
