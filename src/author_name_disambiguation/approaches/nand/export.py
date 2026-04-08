from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from author_name_disambiguation.common.io_schema import save_parquet
from author_name_disambiguation.data.build_mentions import split_author_field


ASSIGNMENT_COLUMNS = [
    "source_type",
    "source_row_idx",
    "bibcode",
    "author_idx",
    "author_raw",
    "author_uid",
    "author_uid_local",
    "author_display_name",
    "assignment_kind",
    "canonical_mention_id",
]

AUTHOR_ENTITY_COLUMNS = [
    "author_uid",
    "author_uid_local",
    "author_display_name",
    "aliases",
    "mention_count",
    "document_count",
    "unique_mention_count",
    "display_name_method",
]


def _iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                yield json.loads(text)
        return

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        if "Bibcode" in payload or "bibcode" in payload:
            yield payload
            return
        for value in payload.values():
            if isinstance(value, dict):
                yield value
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item


def _clean_alias(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _apply_uid_scope_to_local_uid(local_uid: str, uid_scope: str, uid_namespace: str | None) -> str:
    if uid_scope == "local":
        return str(local_uid)
    if uid_namespace is None:
        raise ValueError(f"uid_namespace is required when uid_scope={uid_scope!r}.")
    return f"{uid_namespace}::{local_uid}"


def _explode_source_authors(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) == 0:
        return pd.DataFrame(columns=["source_type", "source_row_idx", "bibcode", "author_idx", "author_raw", "mention_id"])

    working = frame[["source_type", "source_row_idx", "bibcode", "authors"]].copy()
    working = working.explode("authors", ignore_index=True)
    working = working[working["authors"].notna()].copy()
    if len(working) == 0:
        return pd.DataFrame(columns=["source_type", "source_row_idx", "bibcode", "author_idx", "author_raw", "mention_id"])

    working["author_raw"] = working["authors"].astype(str).str.strip()
    working = working[working["author_raw"] != ""].copy()
    if len(working) == 0:
        return pd.DataFrame(columns=["source_type", "source_row_idx", "bibcode", "author_idx", "author_raw", "mention_id"])

    working["author_idx"] = (
        working.groupby(["source_type", "source_row_idx"], sort=False).cumcount().astype("int64")
    )
    working["mention_id"] = working["bibcode"].astype(str) + "::" + working["author_idx"].astype(str)
    return working[["source_type", "source_row_idx", "bibcode", "author_idx", "author_raw", "mention_id"]].reset_index(drop=True)


def _compute_author_entities(assignments: pd.DataFrame) -> pd.DataFrame:
    if len(assignments) == 0:
        raise ValueError("assignments must be non-empty.")

    working = assignments[
        ["author_uid", "author_uid_local", "author_raw", "canonical_mention_id", "source_type", "source_row_idx"]
    ].copy()
    working["alias_clean"] = working["author_raw"].map(_clean_alias)

    alias_counter = (
        working.groupby(["author_uid", "alias_clean"], sort=False)
        .size()
        .rename("alias_count")
        .reset_index()
        .sort_values(["author_uid", "alias_count", "alias_clean"], ascending=[True, False, True], kind="stable")
    )
    aliases = alias_counter.groupby("author_uid", sort=True)["alias_clean"].agg(list).rename("aliases")
    display_names = (
        alias_counter.drop_duplicates(subset=["author_uid"], keep="first")
        .set_index("author_uid")["alias_clean"]
        .rename("author_display_name")
    )

    local_counter = (
        working.groupby(["author_uid", "author_uid_local"], sort=False)
        .size()
        .rename("local_count")
        .reset_index()
        .sort_values(["author_uid", "local_count", "author_uid_local"], ascending=[True, False, True], kind="stable")
    )
    local_ids = (
        local_counter.drop_duplicates(subset=["author_uid"], keep="first")
        .set_index("author_uid")["author_uid_local"]
        .rename("author_uid_local")
    )

    mention_count = working.groupby("author_uid", sort=True).size().rename("mention_count")
    document_count = (
        working[["author_uid", "source_type", "source_row_idx"]]
        .drop_duplicates()
        .groupby("author_uid", sort=True)
        .size()
        .rename("document_count")
    )
    unique_mention_count = (
        working[["author_uid", "canonical_mention_id"]]
        .drop_duplicates()
        .groupby("author_uid", sort=True)
        .size()
        .rename("unique_mention_count")
    )

    entities = pd.concat(
        [local_ids, display_names, aliases, mention_count, document_count, unique_mention_count],
        axis=1,
    ).reset_index()
    entities["display_name_method"] = "most_frequent_alias"
    return entities[AUTHOR_ENTITY_COLUMNS].sort_values("author_uid").reset_index(drop=True)


def build_source_author_assignments(
    *,
    publications: pd.DataFrame,
    references: pd.DataFrame,
    canonical_records: pd.DataFrame,
    clusters: pd.DataFrame,
    uid_scope: str,
    uid_namespace: str | None,
    output_path: str | Path | None = None,
    return_author_entities: bool = False,
    return_runtime_meta: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, dict[str, Any]] | tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    build_started_at = perf_counter()
    exploded_frames = [_explode_source_authors(publications), _explode_source_authors(references)]
    out = pd.concat(exploded_frames, ignore_index=True) if exploded_frames else pd.DataFrame()
    if len(out) == 0:
        raise ValueError("No source author assignments were created.")

    cluster_lookup = clusters[["mention_id", "author_uid", "author_uid_local"]].drop_duplicates(subset=["mention_id"])
    out = out.merge(cluster_lookup, on="mention_id", how="left")

    canonical_lookup = canonical_records[["bibcode", "canonical_source_type", "canonical_source_row_idx"]].drop_duplicates(
        subset=["bibcode"]
    )
    out = out.merge(canonical_lookup, on="bibcode", how="left")

    fallback_mask = out["author_uid"].isna() | out["author_uid_local"].isna()
    source_type_series = out["source_type"].astype(str)
    source_row_idx_series = out["source_row_idx"].astype("int64").astype(str)
    author_idx_series = out["author_idx"].astype("int64").astype(str)
    fallback_uid_local = "src." + source_type_series + "." + source_row_idx_series + "." + author_idx_series
    fallback_canonical_id = "src::" + source_type_series + "::" + source_row_idx_series + "::" + author_idx_series
    if uid_scope == "local":
        fallback_uid = fallback_uid_local
    else:
        if uid_namespace is None:
            raise ValueError(f"uid_namespace is required when uid_scope={uid_scope!r}.")
        fallback_uid = f"{uid_namespace}::" + fallback_uid_local

    out["author_uid_local"] = out["author_uid_local"].where(~fallback_mask, fallback_uid_local)
    out["author_uid"] = out["author_uid"].where(~fallback_mask, fallback_uid)

    canonical_row_idx = pd.to_numeric(out["canonical_source_row_idx"], errors="coerce").fillna(-1).astype("int64")
    is_canonical_row = (
        (source_type_series == out["canonical_source_type"].fillna("").astype(str))
        & (out["source_row_idx"].astype("int64") == canonical_row_idx)
    )
    out["assignment_kind"] = "projected_duplicate"
    out.loc[fallback_mask, "assignment_kind"] = "fallback_unmatched"
    out.loc[~fallback_mask & is_canonical_row, "assignment_kind"] = "canonical"
    out["canonical_mention_id"] = out["mention_id"].where(~fallback_mask, fallback_canonical_id)

    assignments_ready_at = perf_counter()
    author_entities = _compute_author_entities(out)
    author_entities_seconds = perf_counter() - assignments_ready_at
    out = out.merge(
        author_entities[["author_uid", "author_display_name"]],
        on="author_uid",
        how="left",
    )
    out = out.drop(columns=["mention_id", "canonical_source_type", "canonical_source_row_idx"])
    out = out[ASSIGNMENT_COLUMNS].sort_values(["source_type", "source_row_idx", "author_idx"]).reset_index(drop=True)

    if out["author_uid"].isna().any():
        raise RuntimeError("source_author_assignments contains null author_uid values.")
    if out["canonical_mention_id"].isna().any():
        raise RuntimeError("source_author_assignments contains null canonical_mention_id values.")

    if output_path is not None:
        save_parquet(out, output_path, index=False)
    runtime_meta = {
        "build_assignments_seconds": float(assignments_ready_at - build_started_at),
        "build_author_entities_seconds": float(author_entities_seconds),
    }
    if return_author_entities and return_runtime_meta:
        return out, author_entities, runtime_meta
    if return_author_entities:
        return out, author_entities
    if return_runtime_meta:
        return out, runtime_meta
    return out


def build_author_entities(assignments: pd.DataFrame, output_path: str | Path | None = None) -> pd.DataFrame:
    out = _compute_author_entities(assignments)
    if output_path is not None:
        save_parquet(out, output_path, index=False)
    return out


def _source_export_stats() -> dict[str, Any]:
    return {
        "rows_total": 0,
        "authors_total": 0,
        "authors_mapped": 0,
        "authors_unmapped": 0,
        "authors_fallback": 0,
        "coverage_rate": 0.0,
    }


def _finalize_source_stats(stats: dict[str, Any]) -> dict[str, Any]:
    authors_total = int(stats["authors_total"])
    stats["coverage_rate"] = float(stats["authors_mapped"] / max(1, authors_total))
    return stats


def _normalize_source_authors(value: Any) -> list[str]:
    return split_author_field(value)


def _aggregate_assignments_for_source(assignments: pd.DataFrame, source_type: str) -> pd.DataFrame:
    subset = assignments[assignments["source_type"] == source_type].copy()
    if len(subset) == 0:
        return pd.DataFrame(
            columns=["source_row_idx", "AuthorUID", "AuthorDisplayName", "mapped_count", "authors_fallback"]
        )

    subset = subset.sort_values(["source_row_idx", "author_idx"], kind="stable").reset_index(drop=True)
    author_uid_lists = subset.groupby("source_row_idx", sort=False)["author_uid"].agg(list).rename("AuthorUID")
    display_name_lists = (
        subset.groupby("source_row_idx", sort=False)["author_display_name"].agg(list).rename("AuthorDisplayName")
    )
    mapped_count = subset.groupby("source_row_idx", sort=False).size().rename("mapped_count")
    fallback_count = (
        (subset["assignment_kind"] == "fallback_unmatched")
        .groupby(subset["source_row_idx"], sort=False)
        .sum()
        .astype(np.int64)
        .rename("authors_fallback")
    )
    return (
        pd.concat([author_uid_lists, display_name_lists, mapped_count, fallback_count], axis=1)
        .reset_index()
        .sort_values("source_row_idx", kind="stable")
        .reset_index(drop=True)
    )


def _assignment_rows_by_source(assignments: pd.DataFrame, source_type: str) -> dict[int, pd.DataFrame]:
    subset = assignments[assignments["source_type"] == source_type].copy()
    grouped = {}
    for row_idx, group in subset.groupby("source_row_idx", sort=False):
        grouped[int(row_idx)] = group.sort_values("author_idx").reset_index(drop=True)
    return grouped


def _iter_raw_records_with_index(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        for row_idx, row in enumerate(frame.itertuples(index=False)):
            yield int(row_idx), row._asdict()
        return
    for row_idx, payload in enumerate(_iter_json_records(path)):
        yield int(row_idx), payload


def _get_record_authors(record: dict[str, Any]) -> list[str]:
    author_raw = record.get("Author", record.get("author"))
    return split_author_field(author_raw)


def _resolve_export_row_assignments(
    *,
    source_type: str,
    row_idx: int,
    authors: list[str],
    group: pd.DataFrame | None,
) -> tuple[list[str], list[str], int]:
    if group is None:
        if not authors:
            return [], [], 0
        raise RuntimeError(f"Missing source assignments for {source_type}[{row_idx}].")

    author_uids = group["author_uid"].astype(str).tolist()
    display_names = group["author_display_name"].astype(str).tolist()
    if len(author_uids) != len(authors):
        raise RuntimeError(
            f"Assignment length mismatch for {source_type}[{row_idx}]: "
            f"expected {len(authors)} authors, got {len(author_uids)}."
        )
    return author_uids, display_names, int((group["assignment_kind"] == "fallback_unmatched").sum())


def _export_source_file_json(
    *,
    source_type: str,
    input_path: Path,
    output_path: Path,
    assignments: pd.DataFrame,
) -> dict[str, Any]:
    stats = _source_export_stats()
    grouped = _assignment_rows_by_source(assignments, source_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row_idx, record in _iter_raw_records_with_index(input_path):
            authors = _get_record_authors(record)
            group = grouped.get(row_idx)
            author_uids, display_names, authors_fallback = _resolve_export_row_assignments(
                source_type=source_type,
                row_idx=int(row_idx),
                authors=authors,
                group=group,
            )
            stats["rows_total"] += 1
            stats["authors_total"] += len(author_uids)
            stats["authors_mapped"] += len(author_uids)
            stats["authors_fallback"] += authors_fallback

            out_row = dict(record)
            out_row["AuthorUID"] = author_uids
            out_row["AuthorDisplayName"] = display_names
            handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    return _finalize_source_stats(stats)


def _export_source_file_parquet(
    *,
    source_type: str,
    input_path: Path,
    output_path: Path,
    assignments: pd.DataFrame,
    source_frame: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stats = _source_export_stats()
    reread_started_at = perf_counter()
    if source_frame is None:
        out = pd.read_parquet(input_path).reset_index(drop=True)
        source_reread_seconds = perf_counter() - reread_started_at
        mirror_mode = "parquet_reread"
    else:
        out = source_frame.reset_index(drop=True).copy()
        source_reread_seconds = 0.0
        mirror_mode = "parquet_frame_reuse"

    mirror_started_at = perf_counter()
    assignments_agg = _aggregate_assignments_for_source(assignments, source_type)
    existing_export_columns = [
        column
        for column in ["AuthorUID", "AuthorDisplayName", "mapped_count", "authors_fallback"]
        if column in out.columns
    ]
    if existing_export_columns:
        out = out.drop(columns=existing_export_columns)
    out["source_row_idx"] = np.arange(len(out), dtype=np.int64)
    author_col = "Author" if "Author" in out.columns else "author" if "author" in out.columns else None
    if author_col is None:
        out["__authors"] = [[] for _ in range(len(out))]
    else:
        out["__authors"] = out[author_col].map(_normalize_source_authors)
    out["__expected_author_count"] = out["__authors"].map(len).astype(np.int64)
    out = out.merge(assignments_agg, on="source_row_idx", how="left")
    out["mapped_count"] = pd.to_numeric(out["mapped_count"], errors="coerce").fillna(0).astype(np.int64)
    out["authors_fallback"] = pd.to_numeric(out["authors_fallback"], errors="coerce").fillna(0).astype(np.int64)

    authored_mask = out["__expected_author_count"] > 0
    missing_mask = authored_mask & out["AuthorUID"].isna()
    if bool(missing_mask.any()):
        row_idx = int(out.loc[missing_mask, "source_row_idx"].iloc[0])
        raise RuntimeError(f"Missing source assignments for {source_type}[{row_idx}].")

    mismatch_mask = authored_mask & (out["mapped_count"] != out["__expected_author_count"])
    if bool(mismatch_mask.any()):
        row = out.loc[mismatch_mask].iloc[0]
        raise RuntimeError(
            f"Assignment length mismatch for {source_type}[{int(row['source_row_idx'])}]: "
            f"expected {int(row['__expected_author_count'])} authors, got {int(row['mapped_count'])}."
        )

    empty_list_mask = out["AuthorUID"].isna()
    if bool(empty_list_mask.any()):
        empty_index = out.index[empty_list_mask]
        out.loc[empty_index, "AuthorUID"] = pd.Series([[] for _ in range(len(empty_index))], index=empty_index, dtype=object)
        out.loc[empty_index, "AuthorDisplayName"] = pd.Series(
            [[] for _ in range(len(empty_index))], index=empty_index, dtype=object
        )

    stats["rows_total"] = int(len(out))
    stats["authors_total"] = int(out["__expected_author_count"].sum())
    stats["authors_mapped"] = int(out["mapped_count"].sum())
    stats["authors_fallback"] = int(out["authors_fallback"].sum())

    out = out.drop(columns=["source_row_idx", "__authors", "__expected_author_count", "mapped_count", "authors_fallback"])
    save_parquet(out, output_path, index=False)
    return _finalize_source_stats(stats), {
        "mirror_seconds": float(perf_counter() - mirror_started_at),
        "source_reread_seconds": float(source_reread_seconds),
        "mirror_mode": mirror_mode,
    }


def export_source_mirrored_outputs(
    *,
    assignments: pd.DataFrame,
    publications_path: str | Path,
    references_path: str | Path | None,
    publications_output_path: str | Path,
    references_output_path: str | Path | None = None,
    publications_frame: pd.DataFrame | None = None,
    references_frame: pd.DataFrame | None = None,
    return_runtime_meta: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    runtime_meta = {
        "mirror_publications_seconds": 0.0,
        "mirror_references_seconds": 0.0,
        "source_reread_seconds": 0.0,
        "mirror_mode": None,
    }
    pubs_in = Path(publications_path)
    pubs_out = Path(publications_output_path)
    if pubs_in.suffix.lower() == ".parquet":
        pubs_stats, pubs_runtime = _export_source_file_parquet(
            source_type="publication",
            input_path=pubs_in,
            output_path=pubs_out,
            assignments=assignments,
            source_frame=publications_frame,
        )
        runtime_meta["mirror_publications_seconds"] = float(pubs_runtime["mirror_seconds"])
        runtime_meta["source_reread_seconds"] += float(pubs_runtime["source_reread_seconds"])
        runtime_meta["mirror_mode"] = pubs_runtime["mirror_mode"]
    else:
        pubs_stats = _export_source_file_json(
            source_type="publication",
            input_path=pubs_in,
            output_path=pubs_out,
            assignments=assignments,
        )
        runtime_meta["mirror_mode"] = "json_streaming"

    refs_stats = _source_export_stats()
    refs_present = references_path is not None and references_output_path is not None
    if refs_present:
        refs_in = Path(references_path)
        refs_out = Path(references_output_path)
        if refs_in.suffix.lower() == ".parquet":
            refs_stats, refs_runtime = _export_source_file_parquet(
                source_type="reference",
                input_path=refs_in,
                output_path=refs_out,
                assignments=assignments,
                source_frame=references_frame,
            )
            runtime_meta["mirror_references_seconds"] = float(refs_runtime["mirror_seconds"])
            runtime_meta["source_reread_seconds"] += float(refs_runtime["source_reread_seconds"])
            runtime_meta["mirror_mode"] = (
                runtime_meta["mirror_mode"]
                if runtime_meta["mirror_mode"] == refs_runtime["mirror_mode"]
                else "mixed"
            )
        else:
            refs_stats = _export_source_file_json(
                source_type="reference",
                input_path=refs_in,
                output_path=refs_out,
                assignments=assignments,
            )
            runtime_meta["mirror_mode"] = (
                runtime_meta["mirror_mode"]
                if runtime_meta["mirror_mode"] == "json_streaming"
                else "mixed"
            )

    authors_total = int(pubs_stats["authors_total"] + refs_stats["authors_total"])
    authors_mapped = int(pubs_stats["authors_mapped"] + refs_stats["authors_mapped"])
    authors_unmapped = int(pubs_stats["authors_unmapped"] + refs_stats["authors_unmapped"])
    authors_fallback = int(pubs_stats["authors_fallback"] + refs_stats["authors_fallback"])
    qc = {
        "publications": pubs_stats,
        "references": refs_stats,
        "references_present": bool(refs_present),
        "rows_total": int(pubs_stats["rows_total"] + refs_stats["rows_total"]),
        "authors_total": authors_total,
        "authors_mapped": authors_mapped,
        "authors_unmapped": authors_unmapped,
        "authors_fallback": authors_fallback,
        "coverage_rate": float(authors_mapped / max(1, authors_total)),
    }
    if return_runtime_meta:
        return qc, runtime_meta
    return qc
