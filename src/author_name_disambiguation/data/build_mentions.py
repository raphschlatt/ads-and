from __future__ import annotations

from collections.abc import Iterable
import re
from typing import List

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


def explode_records_to_mentions(
    records: pd.DataFrame,
    source_type_default: str,
    authors_col: str = "authors",
) -> pd.DataFrame:
    rows = []
    for rec in records.itertuples(index=False):
        rec_dict = rec._asdict()
        bibcode = str(rec_dict.get("bibcode", "")).strip()
        if not bibcode:
            continue

        authors = rec_dict.get(authors_col) or []
        if isinstance(authors, str):
            authors = split_author_field(authors)
        if not isinstance(authors, Iterable):
            authors = []

        for author_idx, author_raw in enumerate(authors):
            author = str(author_raw).strip()
            if not author:
                continue

            rows.append(
                {
                    "mention_id": make_mention_id(bibcode, author_idx),
                    "bibcode": bibcode,
                    "author_idx": int(author_idx),
                    "author_raw": author,
                    "title": rec_dict.get("title", "") or "",
                    "abstract": rec_dict.get("abstract", "") or "",
                    "year": parse_year(rec_dict.get("year")),
                    "source_type": rec_dict.get("source_type", source_type_default) or source_type_default,
                    "block_key": create_block_key(author),
                    "aff": _resolve_affiliation_value(rec_dict.get("aff"), author_idx),
                    "orcid": rec_dict.get("orcid"),
                    "precomputed_embedding": rec_dict.get("precomputed_embedding"),
                }
            )

    return pd.DataFrame(rows)
