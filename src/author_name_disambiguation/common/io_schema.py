from __future__ import annotations

import json
from pathlib import Path
import shutil
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


def available_disk_bytes(path: str | Path) -> int | None:
    target = Path(path)
    probe = target if target.exists() else target.parent
    if not str(probe):
        probe = Path(".")
    try:
        usage = shutil.disk_usage(probe)
    except Exception:
        return None
    return int(usage.free)


def _sql_quote_path(path: Path) -> str:
    return str(path).replace("'", "''")


def sort_parquet_file(
    input_path: str | Path,
    *,
    order_by: Iterable[str],
    output_path: str | Path | None = None,
) -> Path:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    order = [str(column).strip() for column in order_by if str(column).strip()]
    if not order:
        return source

    destination = Path(output_path) if output_path is not None else source
    temp_output = destination.with_suffix(destination.suffix + ".sorted.tmp")
    if temp_output.exists():
        temp_output.unlink()

    try:
        import duckdb  # type: ignore
    except Exception:
        frame = pd.read_parquet(source)
        frame = frame.sort_values(order).reset_index(drop=True)
        frame.to_parquet(temp_output, index=False)
    else:
        order_sql = ", ".join(f'"{column}"' for column in order)
        conn = duckdb.connect(database=":memory:")
        try:
            conn.execute(
                "COPY (SELECT * FROM read_parquet('{src}') ORDER BY {order_sql}) "
                "TO '{dst}' (FORMAT PARQUET)".format(
                    src=_sql_quote_path(source),
                    order_sql=order_sql,
                    dst=_sql_quote_path(temp_output),
                )
            )
        finally:
            conn.close()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_output.replace(destination)
    return destination


def write_parquet_block_manifest(
    parquet_path: str | Path,
    manifest_path: str | Path,
    *,
    group_key: str = "block_key",
) -> dict[str, Any]:
    source = Path(parquet_path)
    if not source.exists():
        raise FileNotFoundError(source)

    try:
        import duckdb  # type: ignore
    except Exception:
        frame = pd.read_parquet(source, columns=[group_key]) if source.stat().st_size else pd.DataFrame(columns=[group_key])
        block_counts = (
            frame.groupby(group_key).size().reset_index(name="row_count").sort_values(group_key).to_dict(orient="records")
            if len(frame)
            else []
        )
        row_count = int(len(frame))
    else:
        conn = duckdb.connect(database=":memory:")
        try:
            rows = conn.execute(
                "SELECT {group_key}, COUNT(*) AS row_count "
                "FROM read_parquet('{src}') "
                "GROUP BY {group_key} "
                "ORDER BY {group_key}".format(
                    group_key=f'"{group_key}"',
                    src=_sql_quote_path(source),
                )
            ).fetchall()
            block_counts = [{group_key: row[0], "row_count": int(row[1])} for row in rows]
            row_count = int(sum(int(row["row_count"]) for row in block_counts))
        finally:
            conn.close()

    payload = {
        "parquet_path": str(source),
        "group_key": str(group_key),
        "row_count": int(row_count),
        "block_count": int(len(block_counts)),
        "blocks": block_counts,
    }
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload
