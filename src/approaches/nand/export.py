from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.io_schema import save_parquet


def build_publication_author_mapping(
    mentions: pd.DataFrame,
    clusters: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    out = mentions[["bibcode", "author_idx", "mention_id", "source_type"]].copy()
    out = out.merge(clusters[["mention_id", "author_uid"]], on="mention_id", how="left")
    out = out.sort_values(["bibcode", "author_idx"]).reset_index(drop=True)

    if output_path is not None:
        save_parquet(out, output_path, index=False)

    return out
