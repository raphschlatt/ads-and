from __future__ import annotations

from pathlib import Path

import pandas as pd

from author_name_disambiguation.embedding_contract import CANONICAL_TEXT_EMBEDDING_FIELD
from author_name_disambiguation.source_subset import build_contract_valid_source_subset


def test_build_contract_valid_source_subset_writes_raw_like_minimal_contract(monkeypatch, tmp_path: Path):
    publications = pd.DataFrame(
        [
            {
                "bibcode": "pub1",
                "authors": ["Doe J", "Roe A"],
                "title": "Paper 1",
                "abstract": "Abstract 1",
                "year": 2020,
                "aff": ["Inst A", "Inst B"],
                CANONICAL_TEXT_EMBEDDING_FIELD: [0.1, 0.2, 0.3],
                "AuthorUID": ["u1", "u2"],
            }
        ]
    )
    references = pd.DataFrame(
        [
            {
                "bibcode": "ref1",
                "authors": ["Ref X"],
                "title": "Ref Paper",
                "abstract": "Ref Abstract",
                "year": 2019,
                "aff": ["Inst R"],
                CANONICAL_TEXT_EMBEDDING_FIELD: [0.4, 0.5, 0.6],
                "AuthorDisplayName": ["Ref X"],
            }
        ]
    )

    def _fake_load(path, *, source_type):
        return publications.copy() if source_type == "publication" else references.copy()

    monkeypatch.setattr("author_name_disambiguation.source_subset.load_ads_records", _fake_load)

    result = build_contract_valid_source_subset(
        publications_path=tmp_path / "publications.parquet",
        references_path=tmp_path / "references.parquet",
        output_root=tmp_path / "subset",
        drop_precomputed_embeddings=True,
    )

    pubs = pd.read_parquet(result.publications_path)
    refs = pd.read_parquet(result.references_path)

    assert list(pubs.columns) == ["Bibcode", "Author", "Title_en", "Abstract_en", "Year", "Affiliation"]
    assert list(refs.columns) == ["Bibcode", "Author", "Title_en", "Abstract_en", "Year", "Affiliation"]
    assert pubs.iloc[0]["Bibcode"] == "pub1"
    assert refs.iloc[0]["Bibcode"] == "ref1"
    assert result.publications_rows == 1
    assert result.references_rows == 1


def test_build_contract_valid_source_subset_can_keep_precomputed_embeddings(monkeypatch, tmp_path: Path):
    publications = pd.DataFrame(
        [
            {
                "bibcode": "pub1",
                "authors": ["Doe J"],
                "title": "Paper 1",
                "abstract": "Abstract 1",
                "year": 2020,
                "aff": ["Inst A"],
                CANONICAL_TEXT_EMBEDDING_FIELD: [0.1, 0.2, 0.3],
            }
        ]
    )

    monkeypatch.setattr(
        "author_name_disambiguation.source_subset.load_ads_records",
        lambda path, *, source_type: publications.copy(),
    )

    result = build_contract_valid_source_subset(
        publications_path=tmp_path / "publications.parquet",
        output_root=tmp_path / "subset_keep",
        drop_precomputed_embeddings=False,
    )

    pubs = pd.read_parquet(result.publications_path)
    assert CANONICAL_TEXT_EMBEDDING_FIELD in pubs.columns
