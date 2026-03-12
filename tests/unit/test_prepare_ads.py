from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from author_name_disambiguation.data.prepare_ads import prepare_ads_source_data


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
    assert result["runtime"]["explode_mentions_seconds"] >= 0.0
