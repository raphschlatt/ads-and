from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import pandas as pd

from src.common.config import find_project_root, resolve_existing_path
from src.common.io_schema import MENTION_REQUIRED_COLUMNS, validate_columns, save_parquet
from src.data.build_mentions import parse_year, split_author_field, explode_records_to_mentions


def _iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _iter_json_records(path: Path) -> Iterator[Dict]:
    # Fast-path for JSONL (including misnamed *.json files that are line-delimited JSON).
    if path.suffix.lower() == ".jsonl":
        yield from _iter_jsonl(path)
        return

    with path.open("r", encoding="utf-8") as f:
        first = ""
        second = ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not first:
                first = line
            elif not second:
                second = line
                break
    if first.startswith("{") and second.startswith("{"):
        yield from _iter_jsonl(path)
        return

    # Try regular JSON (list/dict), then fallback to JSONL.
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(payload, dict):
            # Could be {id: obj} or single object.
            if "Bibcode" in payload or "bibcode" in payload:
                yield payload
            else:
                for value in payload.values():
                    if isinstance(value, dict):
                        yield value
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                yield item
            return
    except json.JSONDecodeError:
        pass

    yield from _iter_jsonl(path)


def _pick_text(record: Dict, keys: Iterable[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _normalize_ads_record(record: Dict, source_type: str) -> Dict | None:
    bibcode = str(record.get("Bibcode") or record.get("bibcode") or "").strip()
    if not bibcode:
        return None

    author_raw = record.get("Author") or record.get("author") or []
    authors = split_author_field(author_raw)

    emb = record.get("embedding")
    if not (isinstance(emb, list) and emb):
        emb = None

    return {
        "bibcode": bibcode,
        "title": _pick_text(record, ["Title_en", "Title", "title"]),
        "abstract": _pick_text(record, ["Abstract_en", "Abstract", "abstract"]),
        "year": parse_year(record.get("Year") or record.get("year")),
        "aff": _pick_text(record, ["Affiliation", "Affilliation", "aff"]),
        "authors": authors,
        "source_type": source_type,
        "precomputed_embedding": emb,
    }


def load_ads_records(path: str | Path, source_type: str) -> pd.DataFrame:
    project_root = find_project_root(Path.cwd())
    p = resolve_existing_path(path, project_root=project_root)
    if p is None:
        raise FileNotFoundError(f"ADS input file not found: {path}")

    rows: List[Dict] = []
    for rec in _iter_json_records(p):
        norm = _normalize_ads_record(rec, source_type=source_type)
        if norm is not None:
            rows.append(norm)

    if not rows:
        return pd.DataFrame(
            columns=["bibcode", "title", "abstract", "year", "aff", "authors", "source_type", "precomputed_embedding"]
        )

    return pd.DataFrame(rows)


def deduplicate_ads_records(publications: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    pub = publications.copy()
    ref = references.copy()
    pub["_priority"] = 0
    ref["_priority"] = 1

    all_df = pd.concat([pub, ref], ignore_index=True)
    all_df = all_df.sort_values(["bibcode", "_priority"]).reset_index(drop=True)

    dedup_rows = []
    for bibcode, grp in all_df.groupby("bibcode", sort=False):
        first = grp.iloc[0].to_dict()
        source_set = set(grp["source_type"].dropna().astype(str).tolist())

        # Fill missing title/abstract/year/aff from lower-priority rows if needed.
        for field in ["title", "abstract", "year", "aff", "precomputed_embedding", "authors"]:
            if first.get(field) in (None, "", []) or (field == "year" and pd.isna(first.get(field))):
                for _, row in grp.iterrows():
                    value = row.get(field)
                    if value not in (None, "", []) and not (field == "year" and pd.isna(value)):
                        first[field] = value
                        break

        first["source_type"] = "+".join(sorted(source_set)) if source_set else "ads"
        dedup_rows.append(first)

    out = pd.DataFrame(dedup_rows)
    keep_cols = ["bibcode", "title", "abstract", "year", "aff", "authors", "source_type", "precomputed_embedding"]
    return out[keep_cols]


def normalize_ads_mentions(
    publications_path: str | Path,
    references_path: str | Path | None = None,
) -> pd.DataFrame:
    pubs = load_ads_records(publications_path, source_type="publication")
    if references_path is None:
        refs = pd.DataFrame(columns=pubs.columns.tolist())
    else:
        refs = load_ads_records(references_path, source_type="reference")
    dedup = deduplicate_ads_records(pubs, refs)
    mentions = explode_records_to_mentions(dedup, source_type_default="ads")

    if len(mentions) == 0:
        raise ValueError("No ADS mentions created. Check input files and author parsing.")

    validate_columns(mentions, MENTION_REQUIRED_COLUMNS, "ads_mentions")
    return mentions


def prepare_ads_mentions(
    publications_path: str | Path,
    references_path: str | Path | None,
    output_path: str | Path,
) -> pd.DataFrame:
    mentions = normalize_ads_mentions(publications_path=publications_path, references_path=references_path)
    save_parquet(mentions, output_path, index=False)
    return mentions
