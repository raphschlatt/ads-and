from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from author_name_disambiguation.common.io_schema import MENTION_REQUIRED_COLUMNS, validate_columns, save_parquet
from author_name_disambiguation.data.build_blocks import create_block_key
from author_name_disambiguation.data.build_mentions import make_mention_id


LspoRawSourceKind = Literal["parquet", "h5"]


@dataclass(slots=True)
class LspoRawSourceInfo:
    parquet_path: Path | None
    parquet_exists: bool
    h5_path: Path | None
    h5_exists: bool
    selected_source: LspoRawSourceKind | None
    selected_path: Path | None


def _normalize_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _missing_lspo_raw_source_message(info: LspoRawSourceInfo) -> str:
    checked: list[str] = []
    if info.parquet_path is not None:
        checked.append(f"parquet={info.parquet_path}")
    if info.h5_path is not None:
        checked.append(f"h5={info.h5_path}")
    checked_text = ", ".join(checked) if checked else "no paths"
    return (
        "LSPO raw source not found. "
        f"Checked {checked_text}. "
        "Provide either --raw-lspo-parquet or --raw-lspo-h5. "
        "The Zenodo LSPO release can be used through --raw-lspo-h5."
    )


def inspect_lspo_raw_source(
    parquet_path: str | Path | None = None,
    h5_path: str | Path | None = None,
) -> LspoRawSourceInfo:
    parquet_candidate = _normalize_optional_path(parquet_path)
    h5_candidate = _normalize_optional_path(h5_path)
    parquet_exists = parquet_candidate is not None and parquet_candidate.exists()
    h5_exists = h5_candidate is not None and h5_candidate.exists()
    selected_source: LspoRawSourceKind | None = None
    selected_path: Path | None = None
    if parquet_exists:
        selected_source = "parquet"
        selected_path = parquet_candidate
    elif h5_exists:
        selected_source = "h5"
        selected_path = h5_candidate
    return LspoRawSourceInfo(
        parquet_path=parquet_candidate,
        parquet_exists=bool(parquet_exists),
        h5_path=h5_candidate,
        h5_exists=bool(h5_exists),
        selected_source=selected_source,
        selected_path=selected_path,
    )


def resolve_lspo_raw_source(
    parquet_path: str | Path | None = None,
    h5_path: str | Path | None = None,
) -> LspoRawSourceInfo:
    info = inspect_lspo_raw_source(parquet_path=parquet_path, h5_path=h5_path)
    if info.selected_source is None or info.selected_path is None:
        raise FileNotFoundError(_missing_lspo_raw_source_message(info))
    return info


def load_lspo_raw(parquet_path: str | Path | None = None, h5_path: str | Path | None = None) -> pd.DataFrame:
    source = resolve_lspo_raw_source(parquet_path=parquet_path, h5_path=h5_path)
    if source.selected_source == "parquet":
        return pd.read_parquet(source.selected_path)
    return pd.read_hdf(source.selected_path)


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
    parquet_path: str | Path | None,
    output_path: str | Path,
    h5_path: str | Path | None = None,
) -> pd.DataFrame:
    raw = load_lspo_raw(parquet_path=parquet_path, h5_path=h5_path)
    mentions = normalize_lspo_mentions(raw)
    save_parquet(mentions, output_path, index=False)
    return mentions
