from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from author_name_disambiguation.data.prepare_ads import deduplicate_ads_records, prepare_ads_source_data


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_ads_dataset(tmp_path: Path, suffix: str) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    pubs_rows = [
        {
            "Bibcode": "bib1",
            "Author": ["Doe J", "Roe A"],
            "Title_en": "Paper 1",
            "Abstract_en": "Abstract 1",
            "Year": 2020,
            "Affiliation": ["Inst A", "Inst B"],
            "precomputed_embedding": [0.1, 0.2],
        },
        {
            "Bibcode": "bib2",
            "Author": ["Solo X"],
            "Title_en": "Paper 2",
            "Abstract_en": "Abstract 2",
            "Year": 2021,
            "Affiliation": ["Inst C"],
        },
    ]
    refs_rows = [
        {
            "Bibcode": "bib1",
            "Author": ["Doe J", "Roe A"],
            "Title_en": "",
            "Abstract_en": "",
            "Year": None,
            "Affiliation": ["Inst A", "Inst B"],
        },
        {
            "Bibcode": "bib3",
            "Author": ["Ref X"],
            "Title_en": "Paper 3",
            "Abstract_en": "Abstract 3",
            "Year": 2022,
            "Affiliation": ["Inst R"],
        },
    ]

    pubs_path = tmp_path / f"publications{suffix}"
    refs_path = tmp_path / f"references{suffix}"
    if suffix == ".parquet":
        pd.DataFrame(pubs_rows).to_parquet(pubs_path, index=False)
        pd.DataFrame(refs_rows).to_parquet(refs_path, index=False)
    else:
        _write_jsonl(pubs_path, pubs_rows)
        _write_jsonl(refs_path, refs_rows)
    return pubs_path, refs_path


def test_prepare_ads_source_data_parquet_matches_jsonl(tmp_path: Path):
    parquet_pubs, parquet_refs = _write_ads_dataset(tmp_path / "parquet", ".parquet")
    json_pubs, json_refs = _write_ads_dataset(tmp_path / "jsonl", ".jsonl")

    parquet_result = prepare_ads_source_data(parquet_pubs, parquet_refs)
    json_result = prepare_ads_source_data(json_pubs, json_refs)

    sort_keys = {
        "publications": ["bibcode", "source_row_idx"],
        "references": ["bibcode", "source_row_idx"],
        "canonical_records": ["bibcode"],
        "mentions": ["mention_id"],
    }
    for key in ["publications", "references", "canonical_records", "mentions"]:
        left = parquet_result[key].sort_values(sort_keys[key]).reset_index(drop=True)
        right = json_result[key].sort_values(sort_keys[key]).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)


def test_prepare_ads_source_data_can_return_runtime_and_raw_sources(tmp_path: Path):
    pubs_path, refs_path = _write_ads_dataset(tmp_path, ".parquet")

    result = prepare_ads_source_data(
        pubs_path,
        refs_path,
        return_raw_sources=True,
        return_runtime_meta=True,
    )

    assert isinstance(result["raw_publications"], pd.DataFrame)
    assert isinstance(result["raw_references"], pd.DataFrame)
    assert result["runtime"]["publications_mode"] == "parquet_vectorized"
    assert result["runtime"]["read_publications_seconds"] >= 0.0
    assert result["runtime"]["deduplicate_seconds"] >= 0.0
    assert result["runtime"]["deduplicate_mode"] == "single_pass_sorted"
    assert result["runtime"]["input_record_count"] == 4
    assert result["runtime"]["duplicate_bibcode_count"] == 1
    assert result["runtime"]["max_records_per_bibcode"] == 2
    assert result["runtime"]["explode_mentions_seconds"] >= 0.0


def _legacy_deduplicate_ads_records(publications: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    def _is_present_value(value) -> bool:
        if value is None:
            return False
        if isinstance(value, float) and pd.isna(value):
            return False
        if isinstance(value, list):
            return any(
                item is not None and not (isinstance(item, float) and pd.isna(item)) and str(item).strip()
                for item in value
            )
        return bool(str(value).strip())

    pub = publications.copy()
    ref = references.copy()
    pub["_priority"] = 0
    ref["_priority"] = 1
    all_records = pd.concat([pub, ref], ignore_index=True)
    all_records = all_records.sort_values(["bibcode", "_priority", "source_row_idx"], kind="stable").reset_index(drop=True)
    grouped = all_records.groupby("bibcode", sort=False)

    first_rows = all_records.drop_duplicates(subset=["bibcode"], keep="first").copy()
    has_publication = grouped["source_type"].transform(lambda s: bool((s.astype(str) == "publication").any()))
    has_reference = grouped["source_type"].transform(lambda s: bool((s.astype(str) == "reference").any()))
    first_rows["source_type"] = np.select(
        [
            has_publication.loc[first_rows.index].to_numpy(dtype=bool)
            & has_reference.loc[first_rows.index].to_numpy(dtype=bool),
            has_publication.loc[first_rows.index].to_numpy(dtype=bool),
            has_reference.loc[first_rows.index].to_numpy(dtype=bool),
        ],
        ["publication+reference", "publication", "reference"],
        default="ads",
    )

    for field in ["title", "abstract", "year", "aff", "precomputed_embedding", "authors"]:
        candidate = all_records[field]
        valid = candidate.notna() if field == "year" else candidate.map(_is_present_value)
        first_valid = candidate.where(valid, None).groupby(all_records["bibcode"], sort=False).transform("first")
        first_rows[field] = first_valid.loc[first_rows.index].to_list()

    first_rows["canonical_source_type"] = all_records.loc[first_rows.index, "source_type"].astype(str).to_list()
    first_rows["canonical_source_row_idx"] = all_records.loc[first_rows.index, "source_row_idx"].astype(int).to_list()
    return first_rows[
        [
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
    ].reset_index(drop=True)


def test_deduplicate_ads_records_matches_legacy_semantics_on_mixed_inputs():
    publications = pd.DataFrame(
        [
            {
                "bibcode": "dup-1",
                "title": "",
                "abstract": "pub-abstract",
                "year": None,
                "aff": None,
                "authors": ["Pub A"],
                "source_type": "publication",
                "source_row_idx": 4,
                "precomputed_embedding": None,
            },
            {
                "bibcode": "pub-only",
                "title": "pub-title",
                "abstract": "",
                "year": 2020,
                "aff": ["Pub Inst"],
                "authors": ["Pub B"],
                "source_type": "publication",
                "source_row_idx": 1,
                "precomputed_embedding": [0.1, 0.2],
            },
        ]
    )
    references = pd.DataFrame(
        [
            {
                "bibcode": "dup-1",
                "title": "ref-title",
                "abstract": "",
                "year": 2022,
                "aff": ["Ref Inst"],
                "authors": ["Ref A"],
                "source_type": "reference",
                "source_row_idx": 2,
                "precomputed_embedding": [0.5, 0.6],
            },
            {
                "bibcode": "ref-only",
                "title": "ref-only-title",
                "abstract": "ref-only-abstract",
                "year": 2021,
                "aff": ["Ref Only"],
                "authors": ["Ref B"],
                "source_type": "reference",
                "source_row_idx": 9,
                "precomputed_embedding": None,
            },
        ]
    )

    expected = _legacy_deduplicate_ads_records(publications, references)
    actual, meta = deduplicate_ads_records(publications, references, return_meta=True)

    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
    assert meta == {
        "deduplicate_mode": "single_pass_sorted",
        "input_record_count": 4,
        "duplicate_bibcode_count": 1,
        "max_records_per_bibcode": 2,
    }
