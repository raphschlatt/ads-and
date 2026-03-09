from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from author_name_disambiguation.common.io_schema import MENTION_REQUIRED_COLUMNS, save_parquet, validate_columns
from author_name_disambiguation.data.build_mentions import explode_records_to_mentions, parse_year, split_author_field


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)


def _iter_parquet_records(path: Path):
    frame = pd.read_parquet(path)
    for row_idx, row in enumerate(frame.itertuples(index=False)):
        payload = row._asdict()
        payload["_source_row_idx"] = int(row_idx)
        yield payload


def _iter_json_records(path: Path):
    if path.suffix.lower() == ".jsonl":
        for row_idx, payload in enumerate(_iter_jsonl(path)):
            payload["_source_row_idx"] = int(row_idx)
            yield payload
        return

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        for row_idx, item in enumerate(payload):
            if isinstance(item, dict):
                item["_source_row_idx"] = int(row_idx)
                yield item
        return
    if isinstance(payload, dict):
        if "Bibcode" in payload or "bibcode" in payload:
            payload["_source_row_idx"] = 0
            yield payload
            return
        row_idx = 0
        for value in payload.values():
            if isinstance(value, dict):
                value["_source_row_idx"] = row_idx
                row_idx += 1
                yield value
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        item["_source_row_idx"] = row_idx
                        row_idx += 1
                        yield item


def _iter_ads_records(path: Path):
    if path.suffix.lower() == ".parquet":
        yield from _iter_parquet_records(path)
        return
    yield from _iter_json_records(path)


def _pick_text(record: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _pick_optional_list_or_scalar(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
            return [None if item is None else str(item).strip() for item in value]
        text = str(value).strip()
        if text:
            return text
    return None


def _pick_embedding(record: dict[str, Any]) -> list[float] | None:
    for key in ("precomputed_embedding", "embedding"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
            values = [float(item) for item in value]
            if values:
                return values
    return None


def _normalize_ads_record(record: dict[str, Any], source_type: str) -> dict[str, Any]:
    source_row_idx = int(record.get("_source_row_idx", 0))
    bibcode = str(record.get("Bibcode") or record.get("bibcode") or "").strip()
    if not bibcode:
        raise ValueError(f"{source_type}[{source_row_idx}] is missing Bibcode.")

    author_raw = record.get("Author", record.get("author"))
    authors = split_author_field(author_raw)
    if not authors:
        raise ValueError(f"{source_type}[{source_row_idx}] is missing Author.")

    title = _pick_text(record, ["Title_en", "Title", "title"])
    if not title:
        raise ValueError(f"{source_type}[{source_row_idx}] is missing Title_en/Title.")

    abstract = _pick_text(record, ["Abstract_en", "Abstract", "abstract"])
    if not abstract:
        raise ValueError(f"{source_type}[{source_row_idx}] is missing Abstract_en/Abstract.")

    year = parse_year(record.get("Year") or record.get("year"))
    if year is None:
        raise ValueError(f"{source_type}[{source_row_idx}] is missing or has invalid Year.")

    return {
        "bibcode": bibcode,
        "title": title,
        "abstract": abstract,
        "year": year,
        "aff": _pick_optional_list_or_scalar(record, ["Affiliation", "Affilliation", "aff"]),
        "authors": authors,
        "source_type": source_type,
        "source_row_idx": source_row_idx,
        "precomputed_embedding": _pick_embedding(record),
    }


def load_ads_records(path: str | Path, source_type: str) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input file not found: {resolved}")

    rows: list[dict[str, Any]] = []
    for record in _iter_ads_records(resolved):
        rows.append(_normalize_ads_record(record, source_type=source_type))

    return pd.DataFrame(
        rows,
        columns=[
            "bibcode",
            "title",
            "abstract",
            "year",
            "aff",
            "authors",
            "source_type",
            "source_row_idx",
            "precomputed_embedding",
        ],
    )


def deduplicate_ads_records(publications: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    pub = publications.copy()
    ref = references.copy()
    pub["_priority"] = 0
    ref["_priority"] = 1

    all_records = pd.concat([pub, ref], ignore_index=True)
    if len(all_records) == 0:
        return pd.DataFrame(
            columns=[
                "bibcode",
                "title",
                "abstract",
                "year",
                "aff",
                "authors",
                "source_type",
                "source_row_idx",
                "precomputed_embedding",
                "canonical_source_type",
                "canonical_source_row_idx",
            ]
        )

    all_records = all_records.sort_values(["bibcode", "_priority", "source_row_idx"]).reset_index(drop=True)

    dedup_rows: list[dict[str, Any]] = []
    for bibcode, group in all_records.groupby("bibcode", sort=False):
        first = group.iloc[0].to_dict()
        source_set = set(group["source_type"].dropna().astype(str).tolist())
        for field in ["title", "abstract", "year", "aff", "precomputed_embedding", "authors"]:
            current = first.get(field)
            if current in (None, "", []) or (field == "year" and pd.isna(current)):
                for _, row in group.iterrows():
                    candidate = row.get(field)
                    if candidate not in (None, "", []) and not (field == "year" and pd.isna(candidate)):
                        first[field] = candidate
                        break

        first["bibcode"] = bibcode
        first["source_type"] = "+".join(sorted(source_set)) if source_set else "ads"
        first["canonical_source_type"] = str(group.iloc[0]["source_type"])
        first["canonical_source_row_idx"] = int(group.iloc[0]["source_row_idx"])
        dedup_rows.append(first)

    out = pd.DataFrame(dedup_rows)
    keep_cols = [
        "bibcode",
        "title",
        "abstract",
        "year",
        "aff",
        "authors",
        "source_type",
        "source_row_idx",
        "precomputed_embedding",
        "canonical_source_type",
        "canonical_source_row_idx",
    ]
    return out[keep_cols]


def prepare_ads_source_data(
    publications_path: str | Path,
    references_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    publications = load_ads_records(publications_path, source_type="publication")
    references = (
        pd.DataFrame(columns=publications.columns.tolist())
        if references_path is None
        else load_ads_records(references_path, source_type="reference")
    )
    canonical_records = deduplicate_ads_records(publications, references)
    mentions = explode_records_to_mentions(canonical_records, source_type_default="ads")
    if len(mentions) == 0:
        raise ValueError("No source mentions created. Check input files and author parsing.")
    validate_columns(mentions, MENTION_REQUIRED_COLUMNS, "source_mentions")
    return {
        "publications": publications,
        "references": references,
        "canonical_records": canonical_records,
        "mentions": mentions,
    }


def normalize_ads_mentions(
    publications_path: str | Path,
    references_path: str | Path | None = None,
) -> pd.DataFrame:
    return prepare_ads_source_data(
        publications_path=publications_path,
        references_path=references_path,
    )["mentions"]


def prepare_ads_mentions(
    publications_path: str | Path,
    references_path: str | Path | None,
    output_path: str | Path,
) -> pd.DataFrame:
    mentions = normalize_ads_mentions(
        publications_path=publications_path,
        references_path=references_path,
    )
    save_parquet(mentions, output_path, index=False)
    return mentions
