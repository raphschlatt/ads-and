from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from author_name_disambiguation.approaches.nand.export import build_author_entities, export_source_mirrored_outputs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _assignments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_type": "publication",
                "source_row_idx": 0,
                "bibcode": "bib1",
                "author_idx": 0,
                "author_raw": "Doe J",
                "author_uid": "set::blk.a.0",
                "author_uid_local": "blk.a.0",
                "author_display_name": "Doe J",
                "assignment_kind": "canonical",
                "canonical_mention_id": "bib1::0",
            },
            {
                "source_type": "publication",
                "source_row_idx": 0,
                "bibcode": "bib1",
                "author_idx": 1,
                "author_raw": "Roe A",
                "author_uid": "set::blk.a.1",
                "author_uid_local": "blk.a.1",
                "author_display_name": "Roe A",
                "assignment_kind": "canonical",
                "canonical_mention_id": "bib1::1",
            },
            {
                "source_type": "publication",
                "source_row_idx": 1,
                "bibcode": "bib2",
                "author_idx": 0,
                "author_raw": "Doe J.",
                "author_uid": "set::blk.a.0",
                "author_uid_local": "blk.a.0",
                "author_display_name": "Doe J",
                "assignment_kind": "projected_duplicate",
                "canonical_mention_id": "bib2::0",
            },
            {
                "source_type": "reference",
                "source_row_idx": 0,
                "bibcode": "bib3",
                "author_idx": 0,
                "author_raw": "Ref X",
                "author_uid": "set::blk.r.0",
                "author_uid_local": "blk.r.0",
                "author_display_name": "Ref X",
                "assignment_kind": "canonical",
                "canonical_mention_id": "bib3::0",
            },
            {
                "source_type": "reference",
                "source_row_idx": 0,
                "bibcode": "bib3",
                "author_idx": 1,
                "author_raw": "Ref Y",
                "author_uid": "set::src.reference.0.1",
                "author_uid_local": "src.reference.0.1",
                "author_display_name": "Ref Y",
                "assignment_kind": "fallback_unmatched",
                "canonical_mention_id": "src::reference::0::1",
            },
        ]
    )


def test_build_author_entities_uses_most_frequent_alias():
    assignments = _assignments()
    entities = build_author_entities(assignments)

    entity = entities.set_index("author_uid").loc["set::blk.a.0"]
    assert entity["author_display_name"] == "Doe J"
    assert list(entity["aliases"]) == ["Doe J", "Doe J."]
    assert entity["mention_count"] == 2
    assert entity["document_count"] == 2
    assert entity["unique_mention_count"] == 2
    assert entity["display_name_method"] == "most_frequent_alias"


def test_export_source_mirrored_outputs_adds_uid_and_display_name_lists(tmp_path: Path):
    pubs_path = tmp_path / "publications.jsonl"
    refs_path = tmp_path / "references.jsonl"
    pubs_out = tmp_path / "publications_disambiguated.jsonl"
    refs_out = tmp_path / "references_disambiguated.jsonl"

    _write_jsonl(
        pubs_path,
        [
            {"Bibcode": "bib1", "Author": ["Doe J", "Roe A"], "Title_en": "T1"},
            {"Bibcode": "bib2", "Author": ["Doe J."], "Title_en": "T2"},
        ],
    )
    _write_jsonl(
        refs_path,
        [
            {"Bibcode": "bib3", "Author": ["Ref X", "Ref Y"], "Title_en": "R1"},
        ],
    )

    qc = export_source_mirrored_outputs(
        assignments=_assignments(),
        publications_path=pubs_path,
        references_path=refs_path,
        publications_output_path=pubs_out,
        references_output_path=refs_out,
    )

    pub_rows = [json.loads(line) for line in pubs_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    ref_rows = [json.loads(line) for line in refs_out.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert pub_rows[0]["AuthorUID"] == ["set::blk.a.0", "set::blk.a.1"]
    assert pub_rows[0]["AuthorDisplayName"] == ["Doe J", "Roe A"]
    assert ref_rows[0]["AuthorUID"] == ["set::blk.r.0", "set::src.reference.0.1"]
    assert ref_rows[0]["AuthorDisplayName"] == ["Ref X", "Ref Y"]
    assert qc["authors_total"] == 5
    assert qc["authors_mapped"] == 5
    assert qc["authors_unmapped"] == 0
    assert qc["authors_fallback"] == 1
    assert float(qc["coverage_rate"]) == 1.0


def test_export_source_mirrored_outputs_supports_parquet(tmp_path: Path):
    pubs_path = tmp_path / "publications.parquet"
    refs_path = tmp_path / "references.parquet"
    pubs_out = tmp_path / "publications_disambiguated.parquet"
    refs_out = tmp_path / "references_disambiguated.parquet"

    pd.DataFrame(
        [
            {"Bibcode": "bib1", "Author": ["Doe J", "Roe A"], "Title_en": "T1"},
            {"Bibcode": "bib2", "Author": ["Doe J."], "Title_en": "T2"},
        ]
    ).to_parquet(pubs_path, index=False)
    pd.DataFrame(
        [
            {"Bibcode": "bib3", "Author": ["Ref X", "Ref Y"], "Title_en": "R1"},
        ]
    ).to_parquet(refs_path, index=False)

    export_source_mirrored_outputs(
        assignments=_assignments(),
        publications_path=pubs_path,
        references_path=refs_path,
        publications_output_path=pubs_out,
        references_output_path=refs_out,
    )

    pubs_out_df = pd.read_parquet(pubs_out)
    refs_out_df = pd.read_parquet(refs_out)
    assert list(pubs_out_df.loc[0, "AuthorUID"]) == ["set::blk.a.0", "set::blk.a.1"]
    assert list(pubs_out_df.loc[0, "AuthorDisplayName"]) == ["Doe J", "Roe A"]
    assert list(refs_out_df.loc[0, "AuthorUID"]) == ["set::blk.r.0", "set::src.reference.0.1"]


def test_export_source_mirrored_outputs_keeps_authorless_rows_with_empty_lists(tmp_path: Path):
    pubs_path = tmp_path / "publications.parquet"
    pubs_out = tmp_path / "publications_disambiguated.parquet"

    pd.DataFrame(
        [
            {"Bibcode": "bib1", "Author": ["Doe J", "Roe A"], "Title_en": "T1"},
            {"Bibcode": "bib2", "Author": [], "Title_en": "T2"},
        ]
    ).to_parquet(pubs_path, index=False)

    export_source_mirrored_outputs(
        assignments=_assignments().query("source_type == 'publication' and source_row_idx == 0"),
        publications_path=pubs_path,
        references_path=None,
        publications_output_path=pubs_out,
        references_output_path=None,
    )

    pubs_out_df = pd.read_parquet(pubs_out)
    assert list(pubs_out_df.loc[0, "AuthorUID"]) == ["set::blk.a.0", "set::blk.a.1"]
    assert list(pubs_out_df.loc[1, "AuthorUID"]) == []
    assert list(pubs_out_df.loc[1, "AuthorDisplayName"]) == []


def test_export_source_mirrored_outputs_still_fails_when_authored_row_lacks_assignments(tmp_path: Path):
    pubs_path = tmp_path / "publications.parquet"
    pubs_out = tmp_path / "publications_disambiguated.parquet"

    pd.DataFrame(
        [
            {"Bibcode": "bib1", "Author": ["Doe J"], "Title_en": "T1"},
        ]
    ).to_parquet(pubs_path, index=False)

    try:
        export_source_mirrored_outputs(
            assignments=pd.DataFrame(columns=_assignments().columns),
            publications_path=pubs_path,
            references_path=None,
            publications_output_path=pubs_out,
            references_output_path=None,
        )
    except RuntimeError as exc:
        assert "Missing source assignments for publication[0]" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for authored row without assignments.")
