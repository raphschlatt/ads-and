from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.common.io_schema import save_parquet
from src.data.build_mentions import split_author_field


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
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload
        return

    # Accept line-delimited JSON in misnamed *.json files.
    with path.open("r", encoding="utf-8") as f:
        first = ""
        second = ""
        for line in f:
            text = line.strip()
            if not text:
                continue
            if not first:
                first = text
            elif not second:
                second = text
                break
    if first.startswith("{") and second.startswith("{"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload
        return

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
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


def _resolve_bibcode(record: dict[str, Any]) -> str:
    return str(record.get("Bibcode") or record.get("bibcode") or "").strip()


def _source_export_stats() -> dict[str, Any]:
    return {
        "rows_total": 0,
        "authors_total": 0,
        "authors_mapped": 0,
        "authors_unmapped": 0,
        "coverage_rate": 0.0,
    }


def _compute_author_uids(
    *,
    record: dict[str, Any],
    mention_to_uid: dict[str, Any],
    stats: dict[str, Any],
) -> list[str | None]:
    bibcode = _resolve_bibcode(record)
    author_raw = record.get("Author")
    if author_raw is None:
        author_raw = record.get("author")
    if author_raw is None:
        author_raw = []
    authors = split_author_field(author_raw)
    author_uids: list[str | None] = []
    for idx, _author in enumerate(authors):
        mention_id = f"{bibcode}::{idx}" if bibcode else ""
        uid = mention_to_uid.get(mention_id)
        author_uids.append(None if uid is None else str(uid))
        stats["authors_total"] += 1
        if uid is None:
            stats["authors_unmapped"] += 1
        else:
            stats["authors_mapped"] += 1
    return author_uids


def _finalize_source_stats(stats: dict[str, Any]) -> dict[str, Any]:
    authors_total = int(stats["authors_total"])
    stats["coverage_rate"] = float(stats["authors_mapped"] / max(1, authors_total))
    return stats


def _export_source_file_json(
    *,
    input_path: Path,
    output_path: Path,
    mention_to_uid: dict[str, Any],
) -> dict[str, Any]:
    stats = _source_export_stats()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for record in _iter_json_records(input_path):
            stats["rows_total"] += 1
            author_uids = _compute_author_uids(record=record, mention_to_uid=mention_to_uid, stats=stats)

            out_row = dict(record)
            out_row["AuthorUID"] = author_uids
            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    return _finalize_source_stats(stats)


def _export_source_file_parquet(
    *,
    input_path: Path,
    output_path: Path,
    mention_to_uid: dict[str, Any],
) -> dict[str, Any]:
    stats = _source_export_stats()
    df = pd.read_parquet(input_path)
    author_uid_rows: list[list[str | None]] = []
    for row in df.itertuples(index=False):
        stats["rows_total"] += 1
        record = row._asdict()
        author_uid_rows.append(_compute_author_uids(record=record, mention_to_uid=mention_to_uid, stats=stats))

    out = df.copy()
    out["AuthorUID"] = author_uid_rows
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(out, output_path, index=False)
    return _finalize_source_stats(stats)


def _is_parquet_path(path: Path) -> bool:
    return path.suffix.lower() == ".parquet"


def export_source_mirrored_outputs(
    *,
    clusters: pd.DataFrame,
    publications_path: str | Path,
    references_path: str | Path | None,
    publications_output_path: str | Path,
    references_output_path: str | Path | None = None,
) -> dict[str, Any]:
    mention_to_uid = {
        str(row["mention_id"]): str(row["author_uid"])
        for _, row in clusters[["mention_id", "author_uid"]].iterrows()
    }
    pubs_in = Path(publications_path)
    pubs_out = Path(publications_output_path)
    if _is_parquet_path(pubs_in):
        pubs_stats = _export_source_file_parquet(
            input_path=pubs_in,
            output_path=pubs_out,
            mention_to_uid=mention_to_uid,
        )
    else:
        pubs_stats = _export_source_file_json(
            input_path=pubs_in,
            output_path=pubs_out,
            mention_to_uid=mention_to_uid,
        )

    refs_stats = _source_export_stats()
    refs_present = references_path is not None and references_output_path is not None
    if refs_present:
        refs_in = Path(str(references_path))
        refs_out = Path(str(references_output_path))
        if _is_parquet_path(refs_in):
            refs_stats = _export_source_file_parquet(
                input_path=refs_in,
                output_path=refs_out,
                mention_to_uid=mention_to_uid,
            )
        else:
            refs_stats = _export_source_file_json(
                input_path=refs_in,
                output_path=refs_out,
                mention_to_uid=mention_to_uid,
            )

    total_authors = int(pubs_stats["authors_total"] + refs_stats["authors_total"])
    total_mapped = int(pubs_stats["authors_mapped"] + refs_stats["authors_mapped"])
    total_unmapped = int(pubs_stats["authors_unmapped"] + refs_stats["authors_unmapped"])
    return {
        "publications": pubs_stats,
        "references": refs_stats,
        "references_present": bool(refs_present),
        "rows_total": int(pubs_stats["rows_total"] + refs_stats["rows_total"]),
        "authors_total": total_authors,
        "authors_mapped": total_mapped,
        "authors_unmapped": total_unmapped,
        "coverage_rate": float(total_mapped / max(1, total_authors)),
    }
