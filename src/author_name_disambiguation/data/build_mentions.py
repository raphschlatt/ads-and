from __future__ import annotations

from collections.abc import Iterable
import re
from typing import List

import numpy as np
import pandas as pd

from author_name_disambiguation.data.build_blocks import create_block_key


_SPLIT_RE = re.compile(r"\s*[;,]\s*")


def parse_year(value) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        year = int(value)
    except Exception:
        return None
    return year if 1000 <= year <= 2500 else None


def split_author_field(author_field) -> List[str]:
    if author_field is None:
        return []
    if isinstance(author_field, Iterable) and not isinstance(author_field, (str, bytes, dict)):
        return [str(a).strip() for a in author_field if str(a).strip()]

    text = str(author_field).strip()
    if not text:
        return []

    parts = [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]
    # ADS strings are usually "Lastname FN, Lastname FN".
    # If splitting produced only one element, keep as-is.
    if not parts:
        return []
    return parts


def make_mention_id(bibcode: str, author_idx: int) -> str:
    return f"{bibcode}::{author_idx}"


def _resolve_affiliation_value(raw_aff, author_idx: int):
    if isinstance(raw_aff, Iterable) and not isinstance(raw_aff, (str, bytes, dict)):
        values = list(raw_aff)
        if 0 <= int(author_idx) < len(values):
            value = values[int(author_idx)]
            return None if value is None else str(value).strip()
        return None
    if raw_aff is None:
        return None
    text = str(raw_aff).strip()
    return text or None


def _normalize_author_list(author_field) -> list[str]:
    if author_field is None or (isinstance(author_field, float) and pd.isna(author_field)):
        return []
    if isinstance(author_field, Iterable) and not isinstance(author_field, (str, bytes, dict)):
        return [str(a).strip() for a in author_field if str(a).strip()]
    return split_author_field(author_field)


def _normalize_affiliation_values(raw_aff, author_count: int) -> list[str | None]:
    if author_count <= 0:
        return []
    if isinstance(raw_aff, Iterable) and not isinstance(raw_aff, (str, bytes, dict)):
        values = [None if value is None else str(value).strip() or None for value in raw_aff]
        if len(values) < author_count:
            values.extend([None] * (author_count - len(values)))
        return values[:author_count]
    if raw_aff is None or (isinstance(raw_aff, float) and pd.isna(raw_aff)):
        return [None] * author_count
    text = str(raw_aff).strip() or None
    return [text] * author_count


def explode_records_to_mentions(
    records: pd.DataFrame,
    source_type_default: str,
    authors_col: str = "authors",
) -> pd.DataFrame:
    if len(records) == 0:
        return pd.DataFrame(
            columns=[
                "mention_id",
                "canonical_record_id",
                "bibcode",
                "author_idx",
                "author_raw",
                "title",
                "abstract",
                "year",
                "source_type",
                "block_key",
                "aff",
                "orcid",
            ]
        )

    working = records.copy()
    working["bibcode"] = working.get("bibcode", pd.Series(index=working.index, dtype=object)).astype(str).str.strip()
    working = working[working["bibcode"] != ""].copy()
    if len(working) == 0:
        return pd.DataFrame(
            columns=[
                "mention_id",
                "canonical_record_id",
                "bibcode",
                "author_idx",
                "author_raw",
                "title",
                "abstract",
                "year",
                "source_type",
                "block_key",
                "aff",
                "orcid",
            ]
        )

    author_lists = working.get(authors_col, pd.Series(index=working.index, dtype=object)).map(_normalize_author_list)
    working = working.assign(__authors=author_lists)
    working = working[working["__authors"].map(bool)].copy()
    if len(working) == 0:
        return pd.DataFrame(
            columns=[
                "mention_id",
                "canonical_record_id",
                "bibcode",
                "author_idx",
                "author_raw",
                "title",
                "abstract",
                "year",
                "source_type",
                "block_key",
                "aff",
                "orcid",
            ]
        )

    author_counts = working["__authors"].map(len)
    working["__aff_list"] = [
        _normalize_affiliation_values(raw_aff, int(author_count))
        for raw_aff, author_count in zip(working.get("aff", pd.Series(index=working.index, dtype=object)), author_counts)
    ]
    working["__record_id"] = np.arange(len(working), dtype=np.int64)
    exploded = working.explode(["__authors", "__aff_list"], ignore_index=True)
    exploded["author_raw"] = exploded["__authors"].astype(str).str.strip()
    exploded = exploded[exploded["author_raw"] != ""].copy()
    exploded["author_idx"] = exploded.groupby("__record_id", sort=False).cumcount().astype(np.int64)
    exploded["mention_id"] = exploded["bibcode"].astype(str) + "::" + exploded["author_idx"].astype(str)
    if "canonical_record_id" not in exploded.columns:
        exploded["canonical_record_id"] = exploded["__record_id"].astype(np.int64)
    exploded["title"] = exploded.get("title", pd.Series(index=exploded.index, dtype=object)).fillna("").astype(str)
    exploded["abstract"] = exploded.get("abstract", pd.Series(index=exploded.index, dtype=object)).fillna("").astype(str)
    exploded["year"] = exploded.get("year", pd.Series(index=exploded.index, dtype=object)).map(parse_year)
    exploded["source_type"] = (
        exploded.get("source_type", pd.Series(index=exploded.index, dtype=object))
        .fillna(source_type_default)
        .replace("", source_type_default)
        .astype(str)
    )
    exploded["block_key"] = exploded["author_raw"].map(create_block_key)
    exploded["aff"] = exploded["__aff_list"]
    if "orcid" not in exploded.columns:
        exploded["orcid"] = None

    return exploded[
        [
            "mention_id",
            "canonical_record_id",
            "bibcode",
            "author_idx",
            "author_raw",
            "title",
            "abstract",
            "year",
            "source_type",
            "block_key",
            "aff",
            "orcid",
        ]
    ].reset_index(drop=True)
