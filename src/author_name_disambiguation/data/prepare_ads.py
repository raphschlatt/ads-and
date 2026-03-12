from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
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

    # Preserve compatibility with existing ADS drops that contain JSONL content
    # but still use a .json suffix.
    with path.open("r", encoding="utf-8") as handle:
        first = ""
        second = ""
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if not first:
                first = text
                continue
            second = text
            break
    if first.startswith("{") and second.startswith("{"):
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


def _is_present_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return any(
            item is not None and not (isinstance(item, float) and pd.isna(item)) and str(item).strip()
            for item in value
        )
    return bool(str(value).strip())


def _normalize_optional_list_or_scalar_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return [None if item is None else str(item).strip() for item in value]
    text = str(value).strip()
    return text or None


def _normalize_embedding_value(value: Any) -> list[float] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        values = [float(item) for item in value]
        return values or None
    return None


def _coalesce_text_columns(frame: pd.DataFrame, keys: Iterable[str]) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype=object)
    remaining = pd.Series(True, index=frame.index, dtype=bool)
    for key in keys:
        if key not in frame.columns:
            continue
        candidate = frame[key]
        if candidate.dtype != object:
            candidate = candidate.astype(object)
        candidate = candidate.fillna("").astype(str).str.strip()
        take_mask = remaining & candidate.ne("")
        if bool(take_mask.any()):
            result.loc[take_mask] = candidate.loc[take_mask]
            remaining.loc[take_mask] = False
    return result


def _coalesce_object_columns(frame: pd.DataFrame, keys: Iterable[str]) -> pd.Series:
    result = pd.Series([None] * len(frame), index=frame.index, dtype=object)
    remaining = pd.Series(True, index=frame.index, dtype=bool)
    for key in keys:
        if key not in frame.columns:
            continue
        candidate = frame[key]
        take_mask = remaining & candidate.map(_is_present_value)
        if bool(take_mask.any()):
            result.loc[take_mask] = candidate.loc[take_mask]
            remaining.loc[take_mask] = False
    return result


def _normalize_ads_parquet_frame(raw_frame: pd.DataFrame, source_type: str) -> pd.DataFrame:
    bibcode = _coalesce_text_columns(raw_frame, ["Bibcode", "bibcode"])
    authors_raw = _coalesce_object_columns(raw_frame, ["Author", "author"])
    authors = authors_raw.map(split_author_field)
    title = _coalesce_text_columns(raw_frame, ["Title_en", "Title", "title"])
    abstract = _coalesce_text_columns(raw_frame, ["Abstract_en", "Abstract", "abstract"])
    year = _coalesce_object_columns(raw_frame, ["Year", "year"]).map(parse_year)
    aff = _coalesce_object_columns(raw_frame, ["Affiliation", "Affilliation", "aff"]).map(
        _normalize_optional_list_or_scalar_value
    )
    precomputed_embedding = _coalesce_object_columns(raw_frame, ["precomputed_embedding", "embedding"]).map(
        _normalize_embedding_value
    )

    out = pd.DataFrame(
        {
            "bibcode": bibcode,
            "title": title,
            "abstract": abstract,
            "year": year,
            "aff": aff,
            "authors": authors,
            "source_type": str(source_type),
            "source_row_idx": np.arange(len(raw_frame), dtype=np.int64),
            "precomputed_embedding": precomputed_embedding,
        }
    )
    out = out[out["bibcode"].astype(str).str.strip() != ""].copy()
    out = out[out["authors"].map(bool)].copy()
    return out.reset_index(drop=True)


def _normalize_ads_record(record: dict[str, Any], source_type: str) -> dict[str, Any] | None:
    source_row_idx = int(record.get("_source_row_idx", 0))
    bibcode = str(record.get("Bibcode") or record.get("bibcode") or "").strip()
    if not bibcode:
        return None

    author_raw = record.get("Author", record.get("author"))
    authors = split_author_field(author_raw)
    if not authors:
        return None

    title = _pick_text(record, ["Title_en", "Title", "title"])
    abstract = _pick_text(record, ["Abstract_en", "Abstract", "abstract"])
    year = parse_year(record.get("Year") or record.get("year"))

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


def load_ads_records(
    path: str | Path,
    source_type: str,
    *,
    return_raw_source: bool = False,
    return_meta: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input file not found: {resolved}")

    read_started_at = perf_counter()
    normalize_started_at = None
    raw_source: pd.DataFrame | None = None

    if resolved.suffix.lower() == ".parquet":
        raw_source = pd.read_parquet(resolved)
        read_seconds = perf_counter() - read_started_at
        normalize_started_at = perf_counter()
        out = _normalize_ads_parquet_frame(raw_source, source_type=source_type)
        normalize_seconds = perf_counter() - normalize_started_at
        mode = "parquet_vectorized"
    else:
        rows: list[dict[str, Any]] = []
        normalize_started_at = perf_counter()
        for record in _iter_ads_records(resolved):
            normalized = _normalize_ads_record(record, source_type=source_type)
            if normalized is not None:
                rows.append(normalized)
        read_seconds = 0.0
        normalize_seconds = perf_counter() - normalize_started_at
        out = pd.DataFrame(
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
        mode = "record_iter"

    meta = {
        "mode": mode,
        "read_seconds": float(read_seconds),
        "normalize_seconds": float(normalize_seconds),
        "rows": int(len(out)),
    }
    if return_meta:
        return out, (raw_source if return_raw_source else None), meta
    return out


def _empty_canonical_records_frame() -> pd.DataFrame:
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


def _deduplicate_value_is_valid(field: str, value: Any) -> bool:
    if field == "year":
        return bool(pd.notna(value))
    return _is_present_value(value)


def _finalize_deduplicated_group(accumulator: dict[str, Any]) -> dict[str, Any]:
    has_publication = bool(accumulator["has_publication"])
    has_reference = bool(accumulator["has_reference"])
    if has_publication and has_reference:
        source_type = "publication+reference"
    elif has_publication:
        source_type = "publication"
    elif has_reference:
        source_type = "reference"
    else:
        source_type = "ads"

    return {
        "bibcode": accumulator["bibcode"],
        "title": accumulator["title"],
        "abstract": accumulator["abstract"],
        "year": accumulator["year"],
        "aff": accumulator["aff"],
        "authors": accumulator["authors"],
        "source_type": source_type,
        "source_row_idx": accumulator["source_row_idx"],
        "precomputed_embedding": accumulator["precomputed_embedding"],
        "canonical_source_type": accumulator["canonical_source_type"],
        "canonical_source_row_idx": accumulator["canonical_source_row_idx"],
    }


def deduplicate_ads_records(
    publications: pd.DataFrame,
    references: pd.DataFrame,
    *,
    return_meta: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    pub = publications.copy()
    ref = references.copy()
    pub["_priority"] = 0
    ref["_priority"] = 1

    all_records = pd.concat([pub, ref], ignore_index=True)
    if len(all_records) == 0:
        empty = _empty_canonical_records_frame()
        meta = {
            "deduplicate_mode": "single_pass_sorted",
            "input_record_count": 0,
            "duplicate_bibcode_count": 0,
            "max_records_per_bibcode": 0,
        }
        return (empty, meta) if return_meta else empty

    all_records = all_records.sort_values(["bibcode", "_priority", "source_row_idx"], kind="stable").reset_index(drop=True)
    fields = ["title", "abstract", "year", "aff", "precomputed_embedding", "authors"]
    bibcodes = all_records["bibcode"].astype(str).tolist()
    source_types = all_records["source_type"].astype(str).tolist()
    source_row_indices = all_records["source_row_idx"].astype(np.int64).tolist()
    values = {field: all_records[field].tolist() for field in fields}

    output_rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_count = 0
    duplicate_bibcode_count = 0
    max_records_per_bibcode = 0

    def _flush() -> None:
        nonlocal current, current_count, duplicate_bibcode_count, max_records_per_bibcode
        if current is None:
            return
        output_rows.append(_finalize_deduplicated_group(current))
        if current_count > 1:
            duplicate_bibcode_count += 1
        if current_count > max_records_per_bibcode:
            max_records_per_bibcode = current_count
        current = None
        current_count = 0

    for idx, bibcode in enumerate(bibcodes):
        source_type = source_types[idx]
        source_row_idx = int(source_row_indices[idx])
        if current is None or bibcode != current["bibcode"]:
            _flush()
            current = {
                "bibcode": bibcode,
                "title": None,
                "abstract": None,
                "year": None,
                "aff": None,
                "authors": None,
                "source_type": source_type,
                "source_row_idx": source_row_idx,
                "precomputed_embedding": None,
                "canonical_source_type": source_type,
                "canonical_source_row_idx": source_row_idx,
                "has_publication": source_type == "publication",
                "has_reference": source_type == "reference",
            }
            current_count = 1
        else:
            current_count += 1
            if source_type == "publication":
                current["has_publication"] = True
            elif source_type == "reference":
                current["has_reference"] = True

        for field in fields:
            if current[field] is not None:
                continue
            candidate = values[field][idx]
            if _deduplicate_value_is_valid(field, candidate):
                current[field] = candidate

    _flush()

    out = pd.DataFrame.from_records(output_rows, columns=_empty_canonical_records_frame().columns)
    meta = {
        "deduplicate_mode": "single_pass_sorted",
        "input_record_count": int(len(all_records)),
        "duplicate_bibcode_count": int(duplicate_bibcode_count),
        "max_records_per_bibcode": int(max_records_per_bibcode),
    }
    return (out, meta) if return_meta else out


def prepare_ads_source_data(
    publications_path: str | Path,
    references_path: str | Path | None = None,
    *,
    return_raw_sources: bool = False,
    return_runtime_meta: bool = False,
) -> dict[str, pd.DataFrame]:
    pubs_result = load_ads_records(
        publications_path,
        source_type="publication",
        return_raw_source=return_raw_sources,
        return_meta=True,
    )
    publications, raw_publications, pubs_meta = pubs_result
    if references_path is None:
        references = pd.DataFrame(columns=publications.columns.tolist())
        raw_references = None
        refs_meta = {"mode": "not_present", "read_seconds": 0.0, "normalize_seconds": 0.0, "rows": 0}
    else:
        refs_result = load_ads_records(
            references_path,
            source_type="reference",
            return_raw_source=return_raw_sources,
            return_meta=True,
        )
        references, raw_references, refs_meta = refs_result

    deduplicate_started_at = perf_counter()
    deduplicate_result = deduplicate_ads_records(publications, references, return_meta=True)
    canonical_records, deduplicate_meta = deduplicate_result
    deduplicate_seconds = perf_counter() - deduplicate_started_at

    explode_started_at = perf_counter()
    mentions = explode_records_to_mentions(canonical_records, source_type_default="ads")
    explode_mentions_seconds = perf_counter() - explode_started_at
    if len(mentions) == 0:
        raise ValueError("No source mentions created. Check input files and author parsing.")
    validate_columns(mentions, MENTION_REQUIRED_COLUMNS, "source_mentions")
    runtime_meta = {
        "read_publications_seconds": float(pubs_meta["read_seconds"]),
        "read_references_seconds": float(refs_meta["read_seconds"]),
        "normalize_seconds": float(pubs_meta["normalize_seconds"]) + float(refs_meta["normalize_seconds"]),
        "normalize_publications_seconds": float(pubs_meta["normalize_seconds"]),
        "normalize_references_seconds": float(refs_meta["normalize_seconds"]),
        "deduplicate_seconds": float(deduplicate_seconds),
        **dict(deduplicate_meta),
        "explode_mentions_seconds": float(explode_mentions_seconds),
        "publications_mode": pubs_meta["mode"],
        "references_mode": refs_meta["mode"],
    }
    result = {
        "publications": publications,
        "references": references,
        "canonical_records": canonical_records,
        "mentions": mentions,
    }
    if return_raw_sources:
        result["raw_publications"] = raw_publications
        result["raw_references"] = raw_references
    if return_runtime_meta:
        result["runtime"] = runtime_meta
    return result


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
