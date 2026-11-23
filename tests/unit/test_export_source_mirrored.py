from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from author_name_disambiguation.approaches.nand.export import (
    build_author_entities,
    build_source_author_assignments,
    export_source_mirrored_outputs,
)
from author_name_disambiguation.data.prepare_ads import prepare_ads_source_data


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


def test_build_source_author_assignments_can_return_author_entities():
    publications = pd.DataFrame(
        [
            {"bibcode": "bib1", "source_type": "publication", "source_row_idx": 0, "authors": ["Doe J", "Roe A"]},
            {"bibcode": "bib2", "source_type": "publication", "source_row_idx": 1, "authors": ["Doe J."]},
        ]
    )
    references = pd.DataFrame(
        [
            {"bibcode": "bib3", "source_type": "reference", "source_row_idx": 0, "authors": ["Ref X", "Ref Y"]},
        ]
    )
    canonical_records = pd.DataFrame(
        [
            {"bibcode": "bib1", "canonical_source_type": "publication", "canonical_source_row_idx": 0},
            {"bibcode": "bib2", "canonical_source_type": "publication", "canonical_source_row_idx": 1},
            {"bibcode": "bib3", "canonical_source_type": "reference", "canonical_source_row_idx": 0},
        ]
    )
    clusters = pd.DataFrame(
        [
            {"mention_id": "bib1::0", "author_uid": "set::blk.a.0", "author_uid_local": "blk.a.0"},
            {"mention_id": "bib1::1", "author_uid": "set::blk.a.1", "author_uid_local": "blk.a.1"},
            {"mention_id": "bib2::0", "author_uid": "set::blk.a.0", "author_uid_local": "blk.a.0"},
            {"mention_id": "bib3::0", "author_uid": "set::blk.r.0", "author_uid_local": "blk.r.0"},
        ]
    )

    assignments, entities = build_source_author_assignments(
        publications=publications,
        references=references,
        canonical_records=canonical_records,
        clusters=clusters,
        uid_scope="dataset",
        uid_namespace="set",
        return_author_entities=True,
    )

    assert list(assignments["assignment_kind"]) == [
        "canonical",
        "canonical",
        "canonical",
        "canonical",
        "fallback_unmatched",
    ]
    assert set(entities["author_uid"]) == {
        "set::blk.a.0",
        "set::blk.a.1",
        "set::blk.r.0",
        "set::src.reference.0.1",
    }


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
            {"Bibcode": "bib1", "Author": ["Doe J", "Roe A"], "Title_en": "T1", "DOI": "10.1/a", "Keywords": ["x"]},
            {"Bibcode": "bib2", "Author": ["Doe J."], "Title_en": "T2", "DOI": "10.1/b", "Keywords": ["y"]},
        ]
    ).to_parquet(pubs_path, index=False)
    pd.DataFrame(
        [
            {"Bibcode": "bib3", "Author": ["Ref X", "Ref Y"], "Title_en": "R1", "PDF_URL": "https://example.test/ref.pdf"},
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
    assert pubs_out_df.loc[0, "DOI"] == "10.1/a"
    assert list(pubs_out_df.loc[0, "Keywords"]) == ["x"]
    assert list(refs_out_df.loc[0, "AuthorUID"]) == ["set::blk.r.0", "set::src.reference.0.1"]
    assert refs_out_df.loc[0, "PDF_URL"] == "https://example.test/ref.pdf"


def test_export_source_mirrored_outputs_reuses_in_memory_parquet_frames(tmp_path: Path):
    pubs_path = tmp_path / "publications.parquet"
    refs_path = tmp_path / "references.parquet"
    pubs_out = tmp_path / "publications_disambiguated.parquet"
    refs_out = tmp_path / "references_disambiguated.parquet"

    pubs_df = pd.DataFrame(
        [
            {"Bibcode": "bib1", "Author": ["Doe J", "Roe A"], "Title_en": "T1"},
            {"Bibcode": "bib2", "Author": ["Doe J."], "Title_en": "T2"},
        ]
    )
    refs_df = pd.DataFrame(
        [
            {"Bibcode": "bib3", "Author": ["Ref X", "Ref Y"], "Title_en": "R1"},
        ]
    )
    pubs_df.to_parquet(pubs_path, index=False)
    refs_df.to_parquet(refs_path, index=False)

    qc, runtime = export_source_mirrored_outputs(
        assignments=_assignments(),
        publications_path=pubs_path,
        references_path=refs_path,
        publications_output_path=pubs_out,
        references_output_path=refs_out,
        publications_frame=pubs_df,
        references_frame=refs_df,
        return_runtime_meta=True,
    )

    pubs_out_df = pd.read_parquet(pubs_out)
    refs_out_df = pd.read_parquet(refs_out)
    assert list(pubs_out_df.loc[0, "AuthorUID"]) == ["set::blk.a.0", "set::blk.a.1"]
    assert list(refs_out_df.loc[0, "AuthorUID"]) == ["set::blk.r.0", "set::src.reference.0.1"]
    assert qc["authors_total"] == 5
    assert runtime["mirror_mode"] == "parquet_frame_reuse"
    assert runtime["source_reread_seconds"] == 0.0


def test_export_source_mirrored_outputs_overwrites_existing_disambiguation_columns(tmp_path: Path):
    pubs_path = tmp_path / "publications_existing_uid.parquet"
    pubs_out = tmp_path / "publications_disambiguated.parquet"

    pd.DataFrame(
        [
            {
                "Bibcode": "bib1",
                "Author": ["Doe J", "Roe A"],
                "Title_en": "T1",
                "AuthorUID": ["old::0", "old::1"],
                "AuthorDisplayName": ["Old Doe", "Old Roe"],
            }
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
    assert list(pubs_out_df.loc[0, "AuthorDisplayName"]) == ["Doe J", "Roe A"]


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


@pytest.mark.parametrize("suffix", [".parquet", ".jsonl"])
def test_source_rows_with_authors_but_without_bibcode_are_skipped(tmp_path: Path, suffix: str):
    pubs_path = tmp_path / f"publications{suffix}"
    pubs_out = tmp_path / f"publications_disambiguated{suffix}"
    rows = [
        {"Bibcode": "bib1", "Author": ["Doe J", "Roe A"], "Title_en": "T1"},
        {"Bibcode": None, "Author": ["Lost A"], "Title_en": "T2"},
        {"Bibcode": "bib3", "Author": ["Doe J."], "Title_en": "T3"},
    ]
    if suffix == ".parquet":
        pd.DataFrame(rows).to_parquet(pubs_path, index=False)
    else:
        _write_jsonl(pubs_path, rows)

    prepared = prepare_ads_source_data(pubs_path)
    clusters = prepared["mentions"][["mention_id"]].copy()
    clusters["author_uid_local"] = ["blk.a.0", "blk.a.1", "blk.a.0"]
    clusters["author_uid"] = "set::" + clusters["author_uid_local"]
    assignments = build_source_author_assignments(
        publications=prepared["publications"],
        references=prepared["references"],
        canonical_records=prepared["canonical_records"],
        clusters=clusters,
        uid_scope="dataset",
        uid_namespace="set",
    )

    qc = export_source_mirrored_outputs(
        assignments=assignments,
        publications_path=pubs_path,
        references_path=None,
        publications_output_path=pubs_out,
        references_output_path=None,
    )

    if suffix == ".parquet":
        out_rows = pd.read_parquet(pubs_out).to_dict("records")
    else:
        out_rows = [json.loads(line) for line in pubs_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["Title_en"] for row in out_rows] == ["T1", "T2", "T3"]
    assert list(out_rows[0]["AuthorUID"]) == ["set::blk.a.0", "set::blk.a.1"]
    assert list(out_rows[1]["AuthorUID"]) == []
    assert list(out_rows[1]["AuthorDisplayName"]) == []
    assert list(out_rows[2]["AuthorUID"]) == ["set::blk.a.0"]
    assert qc["authors_total"] == 4
    assert qc["authors_mapped"] == 3
    assert qc["authors_unmapped"] == 1


def test_string_author_column_with_missing_values_is_skipped(tmp_path: Path):
    pubs_path = tmp_path / "publications.parquet"
    pubs_out = tmp_path / "publications_disambiguated.parquet"
    pd.DataFrame(
        [
            {"Bibcode": "bib1", "Author": "Cole, Shaun; Lacey, Cedric", "Title_en": "T1"},
            {"Bibcode": "bib2", "Author": None, "Title_en": "T2"},
            {"Bibcode": None, "Author": None, "Title_en": "T3"},
        ]
    ).to_parquet(pubs_path, index=False)

    prepared = prepare_ads_source_data(pubs_path)
    clusters = prepared["mentions"][["mention_id"]].copy()
    clusters["author_uid_local"] = ["s.cole::0", "c.lacey::0"]
    clusters["author_uid"] = "set::" + clusters["author_uid_local"]
    assignments = build_source_author_assignments(
        publications=prepared["publications"],
        references=prepared["references"],
        canonical_records=prepared["canonical_records"],
        clusters=clusters,
        uid_scope="dataset",
        uid_namespace="set",
    )
    qc = export_source_mirrored_outputs(
        assignments=assignments,
        publications_path=pubs_path,
        references_path=None,
        publications_output_path=pubs_out,
        references_output_path=None,
    )

    out = pd.read_parquet(pubs_out)
    assert list(out.loc[0, "AuthorUID"]) == ["set::s.cole::0", "set::c.lacey::0"]
    assert list(out.loc[1, "AuthorUID"]) == []
    assert list(out.loc[2, "AuthorUID"]) == []
    assert qc["authors_total"] == 2
    assert qc["authors_unmapped"] == 0


def test_duplicate_record_with_reordered_authors_does_not_inherit_wrong_identities(tmp_path: Path):
    pubs_path = tmp_path / "publications.parquet"
    refs_path = tmp_path / "references.parquet"
    refs_out = tmp_path / "references_disambiguated.parquet"
    pd.DataFrame(
        [{"Bibcode": "bib1", "Author": ["Doe, John", "Roe, Anna"], "Title_en": "T1"}]
    ).to_parquet(pubs_path, index=False)
    pd.DataFrame(
        [
            {"Bibcode": "bib1", "Author": ["Roe, Anna", "Doe, John"], "Title_en": "T1"},
            {"Bibcode": "bib1", "Author": ["Doe, J.", "Roe, A."], "Title_en": "T1"},
        ]
    ).to_parquet(refs_path, index=False)

    prepared = prepare_ads_source_data(pubs_path, refs_path)
    uid_by_mention = {"bib1::0": "j.doe::0", "bib1::1": "a.roe::0"}
    clusters = prepared["mentions"][["mention_id"]].copy()
    clusters["author_uid_local"] = clusters["mention_id"].map(uid_by_mention)
    clusters["author_uid"] = "set::" + clusters["author_uid_local"]
    assignments = build_source_author_assignments(
        publications=prepared["publications"],
        references=prepared["references"],
        canonical_records=prepared["canonical_records"],
        clusters=clusters,
        uid_scope="dataset",
        uid_namespace="set",
    )
    qc = export_source_mirrored_outputs(
        assignments=assignments,
        publications_path=pubs_path,
        references_path=refs_path,
        publications_output_path=tmp_path / "publications_disambiguated.parquet",
        references_output_path=refs_out,
    )

    refs_out_df = pd.read_parquet(refs_out)
    # Same order with abbreviated names: the identities of the first record carry over.
    assert list(refs_out_df.loc[1, "AuthorUID"]) == ["set::j.doe::0", "set::a.roe::0"]
    # Reversed order: position alone would swap the two people, so both stay unresolved.
    assert list(refs_out_df.loc[0, "AuthorUID"]) == ["set::src.reference.0.0", "set::src.reference.0.1"]
    reversed_rows = assignments[(assignments["source_type"] == "reference") & (assignments["source_row_idx"] == 0)]
    assert set(reversed_rows["assignment_kind"]) == {"fallback_name_mismatch"}
    assert qc["references"]["authors_fallback"] == 2
    assert qc["authors_name_mismatch"] == 2


def test_reordered_duplicate_with_two_letter_surnames_does_not_inherit_wrong_identities():
    publications = pd.DataFrame(
        [{"bibcode": "bib1", "source_type": "publication", "source_row_idx": 0, "authors": ["Li, W.", "Wu, W."]}]
    )
    references = pd.DataFrame(
        [{"bibcode": "bib1", "source_type": "reference", "source_row_idx": 0, "authors": ["Wu, W.", "Li, W."]}]
    )
    canonical_records = pd.DataFrame(
        [{"bibcode": "bib1", "canonical_source_type": "publication", "canonical_source_row_idx": 0}]
    )
    clusters = pd.DataFrame(
        [
            {"mention_id": "bib1::0", "author_uid": "set::w.li::0", "author_uid_local": "w.li::0"},
            {"mention_id": "bib1::1", "author_uid": "set::w.wu::0", "author_uid_local": "w.wu::0"},
        ]
    )

    assignments = build_source_author_assignments(
        publications=publications,
        references=references,
        canonical_records=canonical_records,
        clusters=clusters,
        uid_scope="dataset",
        uid_namespace="set",
    )

    reference_rows = assignments[assignments["source_type"] == "reference"]
    assert reference_rows["assignment_kind"].tolist() == ["fallback_name_mismatch", "fallback_name_mismatch"]
    assert reference_rows["author_uid"].tolist() == ["set::src.reference.0.0", "set::src.reference.0.1"]


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
