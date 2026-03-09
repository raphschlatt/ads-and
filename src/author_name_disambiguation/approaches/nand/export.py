from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

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


def build_publication_author_mapping(
    mentions: pd.DataFrame,
    clusters: pd.DataFrame,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    out = mentions[["bibcode", "author_idx", "mention_id", "source_type"]].copy()
    cluster_cols = ["mention_id", "author_uid"]
    if "author_uid_local" in clusters.columns:
        cluster_cols.append("author_uid_local")
    out = out.merge(clusters[cluster_cols], on="mention_id", how="left")
    out = out.sort_values(["bibcode", "author_idx"]).reset_index(drop=True)
    if output_path is not None:
        save_parquet(out, output_path, index=False)
    return out


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


def build_source_author_assignments(
    *,
    publications: pd.DataFrame,
    references: pd.DataFrame,
    canonical_records: pd.DataFrame,
    clusters: pd.DataFrame,
    uid_scope: str,
    uid_namespace: str | None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    mention_map = (
        clusters[["mention_id", "author_uid", "author_uid_local"]]
        .drop_duplicates(subset=["mention_id"])
        .set_index("mention_id")
        .to_dict("index")
    )
    canonical_row_map = canonical_records.set_index("bibcode")[["canonical_source_type", "canonical_source_row_idx"]].to_dict("index")

    rows: list[dict[str, Any]] = []
    for frame in [publications, references]:
        if len(frame) == 0:
            continue
        for rec in frame.itertuples(index=False):
            payload = rec._asdict()
            bibcode = str(payload.get("bibcode", "")).strip()
            source_type = str(payload.get("source_type", "")).strip()
            source_row_idx = int(payload.get("source_row_idx", 0))
            authors = payload.get("authors") or []
            if isinstance(authors, str):
                authors = split_author_field(authors)
            for author_idx, author_raw in enumerate(authors):
                mention_id = f"{bibcode}::{author_idx}"
                author = str(author_raw).strip()
                cluster_info = mention_map.get(mention_id)
                canonical_info = canonical_row_map.get(bibcode, {})
                is_canonical_row = (
                    str(canonical_info.get("canonical_source_type", "")) == source_type
                    and int(canonical_info.get("canonical_source_row_idx", -1)) == source_row_idx
                )

                if cluster_info is None:
                    author_uid_local = f"src.{source_type}.{source_row_idx}.{author_idx}"
                    author_uid = _apply_uid_scope_to_local_uid(author_uid_local, uid_scope=uid_scope, uid_namespace=uid_namespace)
                    assignment_kind = "fallback_unmatched"
                    canonical_mention_id = f"src::{source_type}::{source_row_idx}::{author_idx}"
                else:
                    author_uid_local = str(cluster_info["author_uid_local"])
                    author_uid = str(cluster_info["author_uid"])
                    assignment_kind = "canonical" if is_canonical_row else "projected_duplicate"
                    canonical_mention_id = mention_id

                rows.append(
                    {
                        "source_type": source_type,
                        "source_row_idx": int(source_row_idx),
                        "bibcode": bibcode,
                        "author_idx": int(author_idx),
                        "author_raw": author,
                        "author_uid": author_uid,
                        "author_uid_local": author_uid_local,
                        "author_display_name": None,
                        "assignment_kind": assignment_kind,
                        "canonical_mention_id": canonical_mention_id,
                    }
                )

    out = pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS)
    if len(out) == 0:
        raise ValueError("No source author assignments were created.")

    author_entities = build_author_entities(out)
    out = out.drop(columns=["author_display_name"]).merge(
        author_entities[["author_uid", "author_display_name"]],
        on="author_uid",
        how="left",
    )
    out = out[ASSIGNMENT_COLUMNS].sort_values(["source_type", "source_row_idx", "author_idx"]).reset_index(drop=True)

    if out["author_uid"].isna().any():
        raise RuntimeError("source_author_assignments contains null author_uid values.")
    if out["canonical_mention_id"].isna().any():
        raise RuntimeError("source_author_assignments contains null canonical_mention_id values.")

    if output_path is not None:
        save_parquet(out, output_path, index=False)
    return out


def build_author_entities(assignments: pd.DataFrame, output_path: str | Path | None = None) -> pd.DataFrame:
    if len(assignments) == 0:
        raise ValueError("assignments must be non-empty.")

    working = assignments.copy()
    working["alias_clean"] = working["author_raw"].map(_clean_alias)
    alias_counter = (
        working.groupby(["author_uid", "alias_clean"])
        .size()
        .rename("alias_count")
        .reset_index()
        .sort_values(["author_uid", "alias_count", "alias_clean"], ascending=[True, False, True])
    )

    alias_map = alias_counter.groupby("author_uid")["alias_clean"].agg(list).to_dict()
    local_counter = (
        working.groupby(["author_uid", "author_uid_local"])
        .size()
        .rename("local_count")
        .reset_index()
        .sort_values(["author_uid", "local_count", "author_uid_local"], ascending=[True, False, True])
    )
    local_map = local_counter.groupby("author_uid")["author_uid_local"].first().to_dict()

    rows: list[dict[str, Any]] = []
    for author_uid, group in working.groupby("author_uid", sort=True):
        aliases = [alias for alias in alias_map.get(author_uid, []) if alias]
        if not aliases:
            aliases = [_clean_alias(group["author_raw"].iloc[0])]
        rows.append(
            {
                "author_uid": str(author_uid),
                "author_uid_local": str(local_map.get(author_uid, group["author_uid_local"].iloc[0])),
                "author_display_name": aliases[0],
                "aliases": aliases,
                "mention_count": int(len(group)),
                "document_count": int(group[["source_type", "source_row_idx"]].drop_duplicates().shape[0]),
                "unique_mention_count": int(group["canonical_mention_id"].astype(str).nunique()),
                "display_name_method": "most_frequent_alias",
            }
        )

    out = pd.DataFrame(rows, columns=AUTHOR_ENTITY_COLUMNS).sort_values("author_uid").reset_index(drop=True)
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
            if group is None:
                raise RuntimeError(f"Missing source assignments for {source_type}[{row_idx}].")
            author_uids = group["author_uid"].astype(str).tolist()
            display_names = group["author_display_name"].astype(str).tolist()
            if len(author_uids) != len(authors):
                raise RuntimeError(
                    f"Assignment length mismatch for {source_type}[{row_idx}]: "
                    f"expected {len(authors)} authors, got {len(author_uids)}."
                )
            stats["rows_total"] += 1
            stats["authors_total"] += len(author_uids)
            stats["authors_mapped"] += len(author_uids)
            stats["authors_fallback"] += int((group["assignment_kind"] == "fallback_unmatched").sum())

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
) -> dict[str, Any]:
    stats = _source_export_stats()
    grouped = _assignment_rows_by_source(assignments, source_type)
    frame = pd.read_parquet(input_path)

    author_uid_rows: list[list[str]] = []
    display_name_rows: list[list[str]] = []
    for row_idx, row in enumerate(frame.itertuples(index=False)):
        authors = _get_record_authors(row._asdict())
        group = grouped.get(row_idx)
        if group is None:
            raise RuntimeError(f"Missing source assignments for {source_type}[{row_idx}].")
        author_uids = group["author_uid"].astype(str).tolist()
        display_names = group["author_display_name"].astype(str).tolist()
        if len(author_uids) != len(authors):
            raise RuntimeError(
                f"Assignment length mismatch for {source_type}[{row_idx}]: "
                f"expected {len(authors)} authors, got {len(author_uids)}."
            )

        author_uid_rows.append(author_uids)
        display_name_rows.append(display_names)
        stats["rows_total"] += 1
        stats["authors_total"] += len(author_uids)
        stats["authors_mapped"] += len(author_uids)
        stats["authors_fallback"] += int((group["assignment_kind"] == "fallback_unmatched").sum())

    out = frame.copy()
    out["AuthorUID"] = author_uid_rows
    out["AuthorDisplayName"] = display_name_rows
    save_parquet(out, output_path, index=False)
    return _finalize_source_stats(stats)


def export_source_mirrored_outputs(
    *,
    assignments: pd.DataFrame,
    publications_path: str | Path,
    references_path: str | Path | None,
    publications_output_path: str | Path,
    references_output_path: str | Path | None = None,
) -> dict[str, Any]:
    pubs_in = Path(publications_path)
    pubs_out = Path(publications_output_path)
    if pubs_in.suffix.lower() == ".parquet":
        pubs_stats = _export_source_file_parquet(
            source_type="publication",
            input_path=pubs_in,
            output_path=pubs_out,
            assignments=assignments,
        )
    else:
        pubs_stats = _export_source_file_json(
            source_type="publication",
            input_path=pubs_in,
            output_path=pubs_out,
            assignments=assignments,
        )

    refs_stats = _source_export_stats()
    refs_present = references_path is not None and references_output_path is not None
    if refs_present:
        refs_in = Path(references_path)
        refs_out = Path(references_output_path)
        if refs_in.suffix.lower() == ".parquet":
            refs_stats = _export_source_file_parquet(
                source_type="reference",
                input_path=refs_in,
                output_path=refs_out,
                assignments=assignments,
            )
        else:
            refs_stats = _export_source_file_json(
                source_type="reference",
                input_path=refs_in,
                output_path=refs_out,
                assignments=assignments,
            )

    authors_total = int(pubs_stats["authors_total"] + refs_stats["authors_total"])
    authors_mapped = int(pubs_stats["authors_mapped"] + refs_stats["authors_mapped"])
    authors_unmapped = int(pubs_stats["authors_unmapped"] + refs_stats["authors_unmapped"])
    authors_fallback = int(pubs_stats["authors_fallback"] + refs_stats["authors_fallback"])
    return {
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
