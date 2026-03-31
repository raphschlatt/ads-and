from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from author_name_disambiguation import cli
from author_name_disambiguation.infer_sources import InferSourcesResult


def _make_infer_result(root: Path, *, mention_rows: list[dict[str, str]], go: bool = True) -> InferSourcesResult:
    root.mkdir(parents=True, exist_ok=True)
    mention_clusters_path = root / "mention_clusters.parquet"
    source_author_assignments_path = root / "source_author_assignments.parquet"
    author_entities_path = root / "author_entities.parquet"
    publications_disambiguated_path = root / "publications_disambiguated.parquet"
    stage_metrics_path = root / "05_stage_metrics_infer_sources.json"
    go_no_go_path = root / "05_go_no_go_infer_sources.json"

    pd.DataFrame(mention_rows).to_parquet(mention_clusters_path, index=False)
    pd.DataFrame(mention_rows).to_parquet(source_author_assignments_path, index=False)
    pd.DataFrame([{"author_uid": row["author_uid"]} for row in mention_rows]).to_parquet(author_entities_path, index=False)
    pd.DataFrame([{"Bibcode": "bib1", "AuthorUID": [row["author_uid"] for row in mention_rows]}]).to_parquet(
        publications_disambiguated_path,
        index=False,
    )
    stage_metrics_path.write_text(json.dumps({"counts": {"ads_mentions": len(mention_rows)}}), encoding="utf-8")
    go_no_go_path.write_text(json.dumps({"go": go}), encoding="utf-8")

    return InferSourcesResult(
        run_id=root.name,
        go=go,
        output_root=root,
        publications_disambiguated_path=publications_disambiguated_path,
        references_disambiguated_path=None,
        source_author_assignments_path=source_author_assignments_path,
        author_entities_path=author_entities_path,
        mention_clusters_path=mention_clusters_path,
        stage_metrics_path=stage_metrics_path,
        go_no_go_path=go_no_go_path,
    )


def test_cli_run_hf_compatibility_report_writes_json_and_markdown(monkeypatch, tmp_path: Path):
    sample_frame = pd.DataFrame(
        [
            {
                "bibcode": "bib1",
                "authors": ["Doe J", "Roe A"],
                "title": "Paper 1",
                "abstract": "Abstract 1",
                "year": 2020,
                "aff": ["Inst A", "Inst B"],
            },
            {
                "bibcode": "bib2",
                "authors": ["Doe J"],
                "title": "Paper 2",
                "abstract": "Abstract 2",
                "year": 2021,
                "aff": ["Inst C"],
            },
        ]
    )

    monkeypatch.setattr(
        "author_name_disambiguation.hf_compatibility_report._resolve_model_bundle",
        lambda _bundle: {
            "manifest": {
                "embedding_contract": {
                    "text": {
                        "model_name": "allenai/specter",
                        "text_backend": "transformers",
                        "text_adapter_name": None,
                        "text_adapter_alias": "specter2",
                        "tokenization": {"max_length": 256},
                    }
                }
            },
            "model_cfg": {"representation": {"text_model_name": "allenai/specter", "max_length": 256}},
        },
    )
    monkeypatch.setattr(
        "author_name_disambiguation.hf_compatibility_report._load_normalized_source",
        lambda _path, source_type: sample_frame.copy() if source_type == "publication" else sample_frame.iloc[0:0].copy(),
    )
    monkeypatch.setattr(
        "author_name_disambiguation.hf_compatibility_report.generate_specter_embeddings",
        lambda mentions, **_kwargs: (
            np.vstack([np.full((1, 768), fill_value=float(idx + 1), dtype=np.float32) for idx in range(len(mentions))]),
            {"generation_mode": "local_mock"},
        ),
    )
    monkeypatch.setattr(
        "author_name_disambiguation.hf_compatibility_report._embed_texts_via_hf",
        lambda **_kwargs: (
            np.vstack([np.full((1, 768), fill_value=float(idx + 1), dtype=np.float32) for idx in range(2)]),
            {"generation_mode": "hf_mock"},
        ),
    )

    calls: list[str] = []

    def _fake_run_infer_sources(request):
        calls.append(str(request.output_root))
        return _make_infer_result(
            Path(request.output_root),
            mention_rows=[
                {"mention_id": "m1", "author_uid": "u1", "block_key": "blk_d"},
                {"mention_id": "m2", "author_uid": "u2", "block_key": "blk_r"},
            ],
            go=True,
        )

    monkeypatch.setattr("author_name_disambiguation.hf_compatibility_report.run_infer_sources", _fake_run_infer_sources)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run-hf-compatibility-report",
            "--publications-path",
            str(tmp_path / "publications.parquet"),
            "--output-root",
            str(tmp_path / "compat"),
            "--dataset-id",
            "my_ads_2026",
            "--model-bundle",
            str(tmp_path / "bundle"),
            "--no-progress",
        ]
    )
    payload = args.func(args)

    report_json = json.loads((tmp_path / "compat" / "hf_compatibility_report.json").read_text(encoding="utf-8"))
    report_md = (tmp_path / "compat" / "hf_compatibility_report.md").read_text(encoding="utf-8")

    assert payload["compatible"] is True
    assert report_json["compatible"] is True
    assert report_json["downstream_smoke"]["changed_assignments"] == 0
    assert report_json["mini_cpu_infer"]["go"] is True
    assert "Status: `PASS`" in report_md
    assert len(calls) == 3
