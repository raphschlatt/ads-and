from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.approaches.nand.export import export_source_mirrored_outputs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_export_source_mirrored_outputs_adds_authoruid_lists(tmp_path: Path):
    pubs_path = tmp_path / "publications.jsonl"
    refs_path = tmp_path / "references.jsonl"
    pubs_out = tmp_path / "publications.disambiguated.jsonl"
    refs_out = tmp_path / "references.disambiguated.jsonl"

    _write_jsonl(
        pubs_path,
        [
            {"Bibcode": "bib1", "Author": ["Doe J", "Roe A"], "Title_en": "T1"},
            {"Bibcode": "bib2", "Author": ["Doe J"], "Title_en": "T2"},
        ],
    )
    _write_jsonl(
        refs_path,
        [
            {"Bibcode": "bib3", "Author": ["Ref X", "Ref Y"], "Title_en": "R1"},
        ],
    )

    clusters = pd.DataFrame(
        [
            {"mention_id": "bib1::0", "block_key": "blk.a", "author_uid": "blk.a::0"},
            {"mention_id": "bib1::1", "block_key": "blk.a", "author_uid": "blk.a::1"},
            {"mention_id": "bib2::0", "block_key": "blk.a", "author_uid": "blk.a::0"},
            {"mention_id": "bib3::0", "block_key": "blk.r", "author_uid": "blk.r::0"},
        ]
    )

    qc = export_source_mirrored_outputs(
        clusters=clusters,
        publications_path=pubs_path,
        references_path=refs_path,
        publications_output_path=pubs_out,
        references_output_path=refs_out,
    )

    pub_rows = [json.loads(line) for line in pubs_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    ref_rows = [json.loads(line) for line in refs_out.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(pub_rows) == 2
    assert len(ref_rows) == 1
    assert pub_rows[0]["AuthorUID"] == ["blk.a::0", "blk.a::1"]
    assert pub_rows[1]["AuthorUID"] == ["blk.a::0"]
    assert ref_rows[0]["AuthorUID"] == ["blk.r::0", None]
    assert qc["authors_total"] == 5
    assert qc["authors_mapped"] == 4
    assert qc["authors_unmapped"] == 1
    assert abs(float(qc["coverage_rate"]) - 0.8) < 1e-9


def test_export_source_mirrored_outputs_supports_parquet(tmp_path: Path):
    pubs_path = tmp_path / "publ_final.parquet"
    refs_path = tmp_path / "refs_final.parquet"
    pubs_out = tmp_path / "publ_final.parquet"
    refs_out = tmp_path / "refs_final.parquet"

    pubs = pd.DataFrame(
        [
            {"Bibcode": "bib1", "Author": ["Doe J", "Roe A"], "Title_en": "T1"},
            {"Bibcode": "bib2", "Author": ["Doe J"], "Title_en": "T2"},
        ]
    )
    refs = pd.DataFrame(
        [
            {"Bibcode": "bib3", "Author": ["Ref X", "Ref Y"], "Title_en": "R1"},
        ]
    )
    pubs.to_parquet(pubs_path, index=False)
    refs.to_parquet(refs_path, index=False)

    clusters = pd.DataFrame(
        [
            {"mention_id": "bib1::0", "block_key": "blk.a", "author_uid": "blk.a::0"},
            {"mention_id": "bib1::1", "block_key": "blk.a", "author_uid": "blk.a::1"},
            {"mention_id": "bib2::0", "block_key": "blk.a", "author_uid": "blk.a::0"},
            {"mention_id": "bib3::0", "block_key": "blk.r", "author_uid": "blk.r::0"},
        ]
    )

    qc = export_source_mirrored_outputs(
        clusters=clusters,
        publications_path=pubs_path,
        references_path=refs_path,
        publications_output_path=pubs_out,
        references_output_path=refs_out,
    )

    pubs_out_df = pd.read_parquet(pubs_out)
    refs_out_df = pd.read_parquet(refs_out)
    assert "AuthorUID" in pubs_out_df.columns
    assert "AuthorUID" in refs_out_df.columns
    assert list(pubs_out_df.loc[0, "AuthorUID"]) == ["blk.a::0", "blk.a::1"]
    assert list(pubs_out_df.loc[1, "AuthorUID"]) == ["blk.a::0"]
    assert list(refs_out_df.loc[0, "AuthorUID"]) == ["blk.r::0", None]
    assert qc["authors_total"] == 5
    assert qc["authors_mapped"] == 4
    assert qc["authors_unmapped"] == 1
